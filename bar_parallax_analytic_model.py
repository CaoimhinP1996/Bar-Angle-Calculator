"""
Analytic Bar Distance/Parallax Model

This module implements computing line of sight average distance/ average parallax
to bar as a function of galactic coordinates (l, b) and bar parameters.

(Note: average parallax is not simply 1/average distance)

Based on a rotated and shifted 3D Gaussian density profile:
- 3D Gaussian density in bar frame
- Rotation by bar angle alpha_deg (the angle made wrt to +ve x-axis is 90-alpha_deg and +ve x-axis is towards l=0, b=0 and +ve y-axis is towards l>0 (left))
- Translation by r_E (distance to bar center from us)
- Line-of-sight integration to compute <s>(l,b) i.e. the average distance

Mathematical formulation:
rho_bar(x', y', z) = rho0 exp(-x'²/(2sigma_x^2) - y'²/(2sigma_y^2) - z²/(2sigma_z^2))

Where (x', y', z) are coordinates in the bar frame (i.e. coordiante axes centered at the bar center and the x' axis aligned with the bar major axis).

Author: Himanshu Verma  
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

class BarDistanceModel:    
    def __init__(self, rho0=1.0, sigma_x=1.0, sigma_y=0.5, sigma_z=0.3, r_E=8.0):
        """
        Initialize bar model parameters
        
        Args:
            rho0: Central density normalization
            sigma_x: Scale length along bar major axis (kpc)
            sigma_y: Scale length along bar minor axis (kpc) 
            sigma_z: Scale length perpendicular to bar plane (kpc)
            r_E: Distance from Sun to bar center (kpc)
        """
        self.rho0 = rho0
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.r_E = r_E
        
    def coordinate_transform(self, s, l_deg, b_deg, alpha_deg):
        """
        Transform from line-of-sight coordinates to bar frame
        
        Args:
            s: Distance along line of sight (kpc)
            l_deg: Galactic longitude (degrees)
            b_deg: Galactic latitude (degrees)
            alpha_deg: Bar angle (degrees)
            
        Returns:
            x_prime, y_prime, z_prime: Coordinates in bar frame
        """
        # Convert to radians
        l = np.radians(l_deg)
        b = np.radians(b_deg)
        alpha = np.radians(-alpha_deg)#angle with +ve x-axis
        
        # Galactic coordinates
        x = s * np.cos(l) * np.cos(b)
        y = s * np.sin(l) * np.cos(b)
        z = s * np.sin(b)
        
        # Transform to bar frame (rotation + translation)
        x_prime = (x - self.r_E) * np.cos(alpha) + y * np.sin(alpha)
        y_prime = -(x - self.r_E) * np.sin(alpha) + y * np.cos(alpha)
        z_prime = z
        
        return x_prime, y_prime, z_prime
    
    def density(self, s, l_deg, b_deg, alpha_deg):
        """
        Compute bar density at distance s along line of sight
        
        Args:
            s: Distance along line of sight (kpc)
            l_deg: Galactic longitude (degrees)
            b_deg: Galactic latitude (degrees)
            alpha_deg: Bar angle (degrees)
            
        Returns:
            rho: Density value
        """
        x_prime, y_prime, z_prime = self.coordinate_transform(s, l_deg, b_deg, alpha_deg)
        
        # 3D Gaussian density in bar frame
        exponent = -(x_prime**2)/(2*self.sigma_x**2) - \
                   (y_prime**2)/(2*self.sigma_y**2) - \
                   (z_prime**2)/(2*self.sigma_z**2)
        
        return self.rho0 * np.exp(exponent)
    
    def average_distance(self, l_deg, b_deg, alpha_deg, s_max=30.0, epsrel=1e-6):
        """
        Compute average distance <s> along line of sight
        
        Args:
            l_deg: Galactic longitude (degrees)
            b_deg: Galactic latitude (degrees)
            alpha_deg: Bar angle (degrees)
            s_max: Maximum integration distance (kpc)
            epsrel: Relative tolerance for integration
            
        Returns:
            avg_distance: Average distance <s> (kpc)
        """
        # Define integrands
        def numerator_integrand(s):
            return s * self.density(s, l_deg, b_deg, alpha_deg) * s**2
        
        def denominator_integrand(s):
            return self.density(s, l_deg, b_deg, alpha_deg) * s**2
        
        # Numerical integration
        try:
            numerator, _ = integrate.quad(numerator_integrand, 0, s_max, epsrel=epsrel)
            denominator, _ = integrate.quad(denominator_integrand, 0, s_max, epsrel=epsrel)
            
            if denominator > 0:
                return numerator / denominator
            else:
                return np.nan
                
        except Exception as e:
            print(f"Integration failed for l={l_deg}, b={b_deg}: {e}")
            return np.nan

    def average_parallax(self, l_deg, b_deg, alpha_deg, s_max=30.0, epsrel=1e-6):
        """
        Compute average parallax <p> along line of sight
        
        Args:
            l_deg: Galactic longitude (degrees)
            b_deg: Galactic latitude (degrees)
            alpha_deg: Bar angle (degrees)
            s_max: Maximum integration distance (kpc)
            epsrel: Relative tolerance for integration
            
        Returns:
            avg_parallax: Average parallax <p> (mas)
        """
        # Define integrands
        def numerator_integrand(s):#p=kpc/s in mas
            p = 1.0 / s  # parallax in kpc^-1
            return p * self.density(s, l_deg, b_deg, alpha_deg) * (1/(p**2))
        
        def denominator_integrand(s):
            p = 1.0 / s  # parallax in kpc^-1
            return self.density(s, l_deg, b_deg, alpha_deg) * (1/(p**2))
        
        # Numerical integration
        try:
            numerator, _ = integrate.quad(numerator_integrand, 0, s_max, epsrel=epsrel)
            denominator, _ = integrate.quad(denominator_integrand, 0, s_max, epsrel=epsrel)
            
            if denominator > 0:
                return numerator / denominator
            else:
                return np.nan
                
        except Exception as e:
            print(f"Integration failed for l={l_deg}, b={b_deg}: {e}")
            return np.nan

# Convenient wrapper functions
def bar_distance3D(l_deg, b_deg, bar_angle_deg, 
                        sigma_x=3.0, sigma_y=1.0, sigma_z=0.5, 
                        rho0=1.0, r_E=8.0, s_max=30.0, epsrel=1e-6):
    """
    Compute theoretical average distance to bar stars for given parameters        
    Returns:
        average_distance: Theoretical average distance (kpc)
    """
    model = BarDistanceModel(rho0, sigma_x, sigma_y, sigma_z, r_E)
    return model.average_distance(l_deg, b_deg, bar_angle_deg, s_max=s_max, epsrel=epsrel)

def bar_parallax3D(l_deg, b_deg, bar_angle_deg, 
                        sigma_x=3.0, sigma_y=1.0, sigma_z=0.5, 
                        rho0=1.0, r_E=8.0, s_max=30.0, epsrel=1e-6):
    """
    Compute theoretical average parallax to bar stars for given parameters    
    Returns:
        average_parallax: Theoretical average parallax (mas)    
    """
    model = BarDistanceModel(rho0, sigma_x, sigma_y, sigma_z, r_E)
    return model.average_parallax(l_deg, b_deg, bar_angle_deg, s_max=s_max, epsrel=epsrel)

def bar_parallax1D(l_deg, bar_angle_deg=30, rE=8.2):
    l = np.radians(l_deg)
    alpha = np.radians(bar_angle_deg)

    d = rE*np.tan(alpha) / ( np.sin(l) + np.cos(l)*np.tan(alpha) )

    return (1/d)#*(d**2)