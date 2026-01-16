import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

class BarAngleFitter:
    """
    MCMC fitter for bar angle and parallax zero-point using analytic model.
    
    Parameters:
    -----------
    parallax_data : array-like
        Measured parallax values (mas)
    parallax_error : array-like
        Measurement errors on parallax (mas)
    l_values : array-like
        Galactic longitude values (degrees)
    b_values : array-like or float
        Galactic latitude values (degrees). Can be single value or array.
    model_params : dict
        Bar model fixed parameters (sigma_x, sigma_y, sigma_z, r_E, s_max)
    """
    
    def __init__(self, parallax_data, parallax_error, l_values, b_values, model_params):
        self.parallax_data = np.array(parallax_data)
        self.parallax_error = np.array(parallax_error)
        self.l_values = np.array(l_values)
        
        # Handle single b value or array
        if np.isscalar(b_values):
            self.b_values = np.full_like(l_values, b_values, dtype=float)
        else:
            self.b_values = np.array(b_values)
            
        self.model_params = model_params
        self.sampler = None
        self.samples = None
        
        # Import the model function
        from bar_parallax_analytic_model import bar_parallax3D
        self.bar_parallax3D = bar_parallax3D
        
    def model_parallax(self, bar_angle, zp):
        model = [ self.bar_parallax3D(l, b, bar_angle, **self.model_params)+zp for l, b in zip(self.l_values, self.b_values)]
        return np.array(model)
    
    def log_likelihood(self, theta):
        bar_angle, zp = theta
        model = self.model_parallax(bar_angle, zp)
        
        # Chi-squared
        chi2 = np.sum(((self.parallax_data - model) / self.parallax_error) ** 2)
        return -0.5 * chi2
    
    def log_prior(self, theta):
        bar_angle, zp = theta
        if self.priors_range is not None:
            ba_range, zp_range = self.priors_range
        else:
            ba_range, zp_range = [(0, 90), (-0.02, 0.02)]
        # Uniform priors
        if ba_range[0] <= bar_angle <= ba_range[1] and zp_range[0] <= zp <= zp_range[1]:
            return 0.0
        return -np.inf
    
    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
        
    def run_mcmc(self, n_walkers=5, n_steps=5000, n_burn=1000, thin=1, initial_guess=None, priors_range=None):
        self.priors_range = priors_range
        self.n_burn = n_burn
        self.thin = thin
        n_dim = 2  # bar_angle, zp
        
        pos = initial_guess + 1e-2 * np.random.randn(n_walkers, n_dim)
        
        '''
        pos = []
        for i in range(n_walkers):
            # Perturbation factors for each parameter
            perturbations = [
                1e-1,   # bar_angle ( % variation)
                1e-1,   # zp ( %variation)
            ]            
            walker_pos = []
            for j, (param, pert) in enumerate(zip(initial_guess, perturbations)):
                walker_pos.append(param +  np.random.normal(0, pert * param))
            pos.append(walker_pos)
        
        pos = np.array(pos)
        '''

        self.sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.log_probability)
        self.sampler.run_mcmc(pos, n_steps, progress=True)
        
        self.samples = self.sampler.get_chain(discard=n_burn, thin=thin, flat=True)
        #tau = self.sampler.get_autocorr_time()
        #print(tau)
        return self.samples
        
    def get_results(self):
        if self.samples is None:
            raise ValueError("Must run MCMC first!")
        
        results = {}
        labels = ['bar_angle', 'zp']
        
        for i, label in enumerate(labels):
            samples_i = self.samples[:, i]
            results[label] = {
                'median': np.median(samples_i),
                'mean': np.mean(samples_i),
                'std': np.std(samples_i),
                'p16': np.percentile(samples_i, 16),
                'p84': np.percentile(samples_i, 84)
            }
        
        return results
    
    def plot_chains(self, filename=None):
        if self.sampler is None:
            raise ValueError("Must run MCMC first!")
        
        chain = self.sampler.get_chain()
        labels = ['Bar Angle (°)', 'Zero-point (mas)']
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        for i in range(2):
            ax = axes[i]
            ax.plot(chain[:, :, i], "k", alpha=0.3)
            ax.axvline(self.n_burn, color='red', linestyle='--', 
                      label='Burn-in' if i == 0 else '')
            ax.set_ylabel(labels[i])
            if i == 0:
                ax.legend()
        
        axes[-1].set_xlabel("Step Number")
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300)
        plt.show()
    
    def plot_corner(self, truths=None, filename=None):
        if self.samples is None:
            raise ValueError("Must run MCMC first!")
        
        labels = ['Bar Angle (°)', 'Zero-point (mas)']
        
        fig = corner.corner(
            self.samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            truths=truths
        )
        
        if filename:
            plt.savefig(filename, dpi=300)
        plt.show()
    
    def plot_fit(self, filename=None):
        if self.samples is None:
            raise ValueError("Must run MCMC first!")
        
        # Get best-fit parameters
        bar_angle_fit = np.median(self.samples[:, 0])
        zp_fit = np.median(self.samples[:, 1])
        
        # Compute best-fit model
        model_fit = self.model_parallax(bar_angle_fit, zp_fit)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(self.l_values, self.parallax_data, 
                   yerr=self.parallax_error,
                   fmt='o-', color='green', alpha=0.8, label='Data', 
                   capsize=5, markersize=6)
        
        ax.plot(self.l_values, model_fit, 
               's--', color='blue', alpha=0.7, markersize=6,
               label=f'Best fit: α={bar_angle_fit:.2f}°, zp={zp_fit:.4f} mas')
        
        ax.set_xlabel('Galactic Longitude (degrees)')
        ax.set_ylabel('Parallax (mas)')
        ax.set_title('Data vs Best-Fit Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(1/8.2, color='gray', linestyle='--', alpha=0.5)

        axsec = ax.secondary_yaxis('right', functions=(lambda x: 1/x, lambda x: 1/x))
        axsec.set_ylabel('Distance (kpc)')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300)
        plt.show()
    
    def plot_results(self, truths=None):
        """Plot all diagnostic plots."""
        self.plot_chains()
        self.plot_corner(truths=truths)
        self.plot_fit()