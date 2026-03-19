import matplotlib.pyplot as plt # type: ignore
import numpy as np
import scipy.optimize as spopt # type: ignore
import fitting_routines as fit
from bar_parallax_analytic_model import bar_parallax3D

def CMD(df1,df2,df3,plotpath=None,l=2.0,b=2.0):
    largest_area = df1
    medium_area = df2
    smallest_area = df3

    fig = plt.figure(figsize=(20,20))
    
    ax = fig.add_subplot(2,3,1)
    ax.scatter(largest_area["bp_rp"],largest_area["phot_g_mean_mag"],s=0.5, alpha = 0.2)
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram of Largest Area')

    ax = fig.add_subplot(2,3,2)
    ax.scatter(medium_area["bp_rp"],medium_area["phot_g_mean_mag"],s=0.5, alpha = 0.2)
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_title('Color-Magnitude Diagram of Medium Area')
    
    ax = fig.add_subplot(2,3,3)
    ax.scatter(smallest_area["bp_rp"],smallest_area["phot_g_mean_mag"],s=0.5, alpha = 0.2)
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_title('Color-Magnitude Diagram of Smallest Area')

    ax = fig.add_subplot(2,3,4)
    ax.scatter(largest_area["bp_rp"],largest_area["phot_g_mean_mag"],s=0.5, alpha = 0.2, c = 'c', label = 'Largest Area')
    ax.scatter(medium_area["bp_rp"],medium_area["phot_g_mean_mag"],s=0.5, alpha = 0.2, c = 'y', label = 'Medium Area')
    ax.scatter(smallest_area["bp_rp"],smallest_area["phot_g_mean_mag"],s=0.5, alpha = 0.2, c='k', label = 'Smallest Area')
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram of Each Area')

    plt.tight_layout()
    plt.savefig(plotpath/f'l_{l:0.2f}_b_{b:0.2f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def SRedClumpPlot(redclump, parallax, df, doubleclump, doubleparallax, plotpath=None, l=None, b=None,  s= 0.33): # plotting routines
    
    xrc,yrc,sxrc,syrc = redclump["Params"]
    initial = redclump["initial cut"]
    C,D,rccol,cNrc,scrc = redclump["cpar"]
    chist = redclump["chist"]
    cbc = redclump["cbc"]
    A,B,rcmag,mNrc,smrc = redclump["mpar"]
    mhist = redclump["mhist"]
    mbc = redclump["mbc"]
    rcp,pNrc,sprc = parallax["ppar"]
    phist = parallax["phist"]
    pbc = parallax["pbc"]
    final_data = parallax["dataframe"]

    fig = plt.figure(figsize=(20,20))

    #2d histogram
    ax = fig.add_subplot(4,3,1)
    ax.hist2d(df['bp_rp'],df['phot_g_mean_mag'],bins = 50)
    ax.scatter(xrc,yrc)
    ax.axvline(xrc-0.55*sxrc)
    ax.axvline(xrc+sxrc)
    ax.axhline(yrc-syrc)
    ax.axhline(yrc+1.45*syrc)
    ax.set_xlim(0.5,4)
    ax.set_ylim(20.5,14)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('2D Histogram')

    # initial CMD cut plotted over original CMD
    ax = fig.add_subplot(4,3,2)
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2) #original data
    ax.scatter(initial['bp_rp'],initial['phot_g_mean_mag'],s=0.5, c='yellow', alpha=0.5) # data after guesses
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram')

    # single clump magnitude histogram
    ax = fig.add_subplot(4,3,3)
    ax.errorbar(mbc,mhist,yerr=np.sqrt(mhist),fmt='o')
    G = np.linspace(min(initial['phot_g_mean_mag']),max(initial['phot_g_mean_mag']),num=200)
    NG = fit.rcmmodel(G,A,B,rcmag,mNrc,smrc)
    ax.plot(G,NG,label='Fitted')
    NGg = fit.gaussian(G,rcmag,mNrc,smrc)
    ax.plot(G,NGg, label = 'Gaussian')
    NGe = fit.exponential(G,A,B,rcmag)
    ax.plot(G,NGe, label = 'Exponential')
    ax.axvline(rcmag, color ='k', label= 'magpeak')
    ax.axvline(rcmag+smrc, color ='gray', linestyle = '--')
    ax.axvline(rcmag-smrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Magnitude Histogram')

    # color histogram
    ax = fig.add_subplot(4,3,5)
    ax.errorbar(cbc,chist,yerr=np.sqrt(chist),fmt='o')
    Col = np.linspace(min(initial['bp_rp']),max(initial['bp_rp']),num=200)
    NC = fit.rccmodel(Col,C,D,rccol,cNrc,scrc)
    ax.plot(Col,NC,label='Fitted')
    NCg = fit.gaussian(Col,rccol,cNrc,scrc)
    ax.plot(Col,NCg, label = 'Gaussian')
    NCe = fit.exponential(Col,C,D,rccol)
    ax.plot(Col,NCe, label = 'Exponential')
    ax.axvline(rccol, color ='k', label= 'magpeak')
    ax.axvline(rccol+scrc, color ='gray', linestyle = '--')
    ax.axvline(rccol-scrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Color Histogram')

    # single clump parallax distribution of final cut
    ax = fig.add_subplot(4,3,6)
    ax.errorbar(pbc,phist,yerr=np.sqrt(phist),fmt='o')
    P = np.linspace(-1,1,num=200)
    NP = fit.rcpmodel(P,rcp,pNrc,sprc)
    ax.plot(P,NP,label='Fitted')
    ax.axvline(rcp, color ='k', label= 'ppeak')
    ax.axvline(rcp+sprc, color ='gray', linestyle = '--')
    ax.axvline(rcp-sprc, color ='gray', linestyle = '--')
    ax.axvline(0.12195, color ='b', label = 'Galactic Center')
    ax.legend()
    ax.set_title('Single Clump Parallax Histogram')

    # final RC cut plotted over original CMD
    ax = fig.add_subplot(4,3,8)
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2)
    ax.scatter(final_data['bp_rp'],final_data['phot_g_mean_mag'],s=0.5, alpha=0.75, cmap='viridis')
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_ylim(20.5,14)
    ax.set_xlim(.5,4)
    ax.set_title('Final Fit Color-Magnitude Selection')

    # magnitude vs. parallax
    ax = fig.add_subplot(4,3,9)
    ax.scatter(final_data["phot_g_mean_mag"], final_data["parallax"], s=2, alpha = 0.5)
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Parallax')
    ax.axhline(rcp, color ='b', label = 'single clump parallax mean')
    ax.set_ylim(-0.2,0.4)
    ax.set_xlim(14,20)

    # color vs. parallax
    ax = fig.add_subplot(4,3,10)
    ax.scatter(final_data["bp_rp"],final_data["parallax"], s=2, alpha = 0.5)
    ax.set_xlabel('Color')
    ax.set_ylabel('Parallax')
    ax.axhline(rcp, color ='b', label = 'single clump parallax mean')
    ax.set_ylim(-0.2,0.4)
    ax.set_xlim(0.5,4)

    plt.tight_layout()
    plt.savefig(plotpath/f'Plots_l_{l:0.4f}_b_{b:0.4f}_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def SRedClumpPlot_break(redclump, df, doubleclump, plotpath=None, l=None, b=None, s= 0.33): # plotting routines
    
    xrc,yrc,sxrc,syrc = redclump["Params"]
    initial = redclump["initial cut"]
    C,D,rccol,cNrc,scrc = redclump["cpar"]
    chist = redclump["chist"]
    cbc = redclump["cbc"]
    A,B,rcmag,mNrc,smrc = redclump["mpar"]
    mhist = redclump["mhist"]
    mbc = redclump["mbc"]

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(2,3,1)

    # 2d histogram
    ax.hist2d(df['bp_rp'],df['phot_g_mean_mag'],bins = 50)
    ax.scatter(xrc,yrc)
    ax.axvline(xrc-0.55*sxrc)
    ax.axvline(xrc+sxrc)
    ax.axhline(yrc-syrc)
    ax.axhline(yrc+1.45*syrc)
    ax.set_xlim(0.5,4)
    ax.set_ylim(20.5,14)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('2D Histogram')

    # initial CMD cut plotted over original CMD
    ax = fig.add_subplot(2,3,2)
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2) #original data
    ax.scatter(initial['bp_rp'],initial['phot_g_mean_mag'],s=0.5, c='yellow', alpha=0.5) # data after guesses
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram')

    # single clump magnitude histogram
    ax = fig.add_subplot(2,3,4)
    ax.errorbar(mbc,mhist,yerr=np.sqrt(mhist),fmt='o')
    G = np.linspace(min(initial['phot_g_mean_mag']),max(initial['phot_g_mean_mag']),num=200)
    NG = fit.rcmmodel(G,A,B,rcmag,mNrc,smrc)
    ax.plot(G,NG,label='Fitted')
    NGg = fit.gaussian(G,rcmag,mNrc,smrc)
    ax.plot(G,NGg, label = 'Gaussian')
    NGe = fit.exponential(G,A,B,rcmag)
    ax.plot(G,NGe, label = 'Exponential')
    ax.axvline(rcmag, color ='k', label= 'magpeak')
    ax.axvline(rcmag+smrc, color ='gray', linestyle = '--')
    ax.axvline(rcmag-smrc, color ='gray', linestyle = '--')
    ax.set_xlim(14,20)
    ax.set_ylim(0,500)
    ax.legend()
    ax.set_title('Magnitude Histogram')

    # color histogram
    ax = fig.add_subplot(2,3,6)
    ax.errorbar(cbc,chist,yerr=np.sqrt(chist),fmt='o')
    Col = np.linspace(min(initial['bp_rp']),max(initial['bp_rp']),num=200)
    NC = fit.rccmodel(Col,C,D,rccol,cNrc,scrc)
    ax.plot(Col,NC,label='Fitted')
    NCg = fit.gaussian(Col,rccol,cNrc,scrc)
    ax.plot(Col,NCg, label = 'Gaussian')
    NCe = fit.exponential(Col,C,D,rccol)
    ax.plot(Col,NCe, label = 'Exponential')
    ax.axvline(rccol, color ='k', label= 'magpeak')
    ax.axvline(rccol+scrc, color ='gray', linestyle = '--')
    ax.axvline(rccol-scrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Color Histogram')

    plt.tight_layout()
    plt.savefig(plotpath/f'Plots_l_{l:0.2f}_b_{b:0.2f}_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def SRedClumpPlot_runtime(redclump, df, plotpath=None, l=None, b=None, s= 0.33): # plotting routines
    
    xrc,yrc,sxrc,syrc = redclump["Params"] # import parameters from dictionary
    initial = redclump["initial cut"]
    
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(2,3,1)

    #2d histogram
    ax.hist2d(df['bp_rp'],df['phot_g_mean_mag'],bins = 50)
    ax.scatter(xrc,yrc)
    ax.axvline(xrc-0.55*sxrc)
    ax.axvline(xrc+sxrc)
    ax.axhline(yrc-syrc)
    ax.axhline(yrc+1.45*syrc)
    ax.set_xlim(0.5,4)
    ax.set_ylim(20.5,14)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('2D Histogram')

    #
    ax = fig.add_subplot(2,3,2) # initial CMD cut plotted over original CMD
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2) #original data
    ax.scatter(initial['bp_rp'],initial['phot_g_mean_mag'],s=0.5, c='yellow', alpha=0.5) # data after guesses
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram')

    plt.tight_layout()
    plt.savefig(plotpath/f'Plots_l_{l:0.2f}_b_{b:0.2f}_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def SRedClumpPlot_type(redclump, df, doubleclump, plotpath=None, l=None, b=None, s= 0.33): # plotting routines
    
    xrc,yrc,sxrc,syrc = redclump["Params"]
    initial = redclump["initial cut"]
    C,D,rccol,cNrc,scrc = redclump["cpar"]
    chist = redclump["chist"]
    cbc = redclump["cbc"]
    A,B,rcmag,mNrc,smrc = redclump["mpar"]
    mhist = redclump["mhist"]
    mbc = redclump["mbc"]

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(2,3,1)

    #2d histogram
    ax.hist2d(df['bp_rp'],df['phot_g_mean_mag'],bins = 50)
    ax.scatter(xrc,yrc)
    ax.axvline(xrc-0.55*sxrc)
    ax.axvline(xrc+sxrc)
    ax.axhline(yrc-syrc)
    ax.axhline(yrc+1.45*syrc)
    ax.set_xlim(0.5,4)
    ax.set_ylim(20.5,14)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('2D Histogram')

    # initial CMD cut plotted over original CMD
    ax = fig.add_subplot(2,3,2)
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2) #original data
    ax.scatter(initial['bp_rp'],initial['phot_g_mean_mag'],s=0.5, c='yellow', alpha=0.5) # data after guesses
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram')

    # single clump magnitude histogram
    ax = fig.add_subplot(2,3,3)
    ax.errorbar(mbc,mhist,yerr=np.sqrt(mhist),fmt='o')
    G = np.linspace(min(initial['phot_g_mean_mag']),max(initial['phot_g_mean_mag']),num=200)
    NG = fit.rcmmodel(G,A,B,rcmag,mNrc,smrc)
    ax.plot(G,NG,label='Fitted')
    NGg = fit.gaussian(G,rcmag,mNrc,smrc)
    ax.plot(G,NGg, label = 'Gaussian')
    NGe = fit.exponential(G,A,B,rcmag)
    ax.plot(G,NGe, label = 'Exponential')
    ax.axvline(rcmag, color ='k', label= 'magpeak')
    ax.axvline(rcmag+smrc, color ='gray', linestyle = '--')
    ax.axvline(rcmag-smrc, color ='gray', linestyle = '--')
    ax.set_xlim(14,20)
    ax.set_ylim(0,500)
    ax.legend()
    ax.set_title('Magnitude Histogram')

    # color histogram
    ax = fig.add_subplot(2,3,5)
    ax.errorbar(cbc,chist,yerr=np.sqrt(chist),fmt='o')
    Col = np.linspace(min(initial['bp_rp']),max(initial['bp_rp']),num=200)
    NC = fit.rccmodel(Col,C,D,rccol,cNrc,scrc)
    ax.plot(Col,NC,label='Fitted')
    NCg = fit.gaussian(Col,rccol,cNrc,scrc)
    ax.plot(Col,NCg, label = 'Gaussian')
    NCe = fit.exponential(Col,C,D,rccol)
    ax.plot(Col,NCe, label = 'Exponential')
    ax.axvline(rccol, color ='k', label= 'magpeak')
    ax.axvline(rccol+scrc, color ='gray', linestyle = '--')
    ax.axvline(rccol-scrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Color Histogram')

    plt.tight_layout()
    plt.savefig(plotpath/f'Plots_l_{l:0.2f}_b_{b:0.2f}_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def Smeanplot(meanplot, Al, b=2.0, plotpath=None, s= 0.33):
    Longitude = meanplot["Long"]
    Para = meanplot["Mean Parallax"]
    Paraerr = meanplot["Mean Parallax Error"]
    MeanMag = meanplot["Mean Magnitude"]
    MeanCol = meanplot["Mean Color"]
    count = meanplot["Count"]
    modulus = meanplot["Distance Modulus"]
    fraction = meanplot["Red Clump Fraction"]
    width = meanplot["Red Clump Width"]

    #for comparison with analytic model
    # Test with different bar structure parameters
    custom_params = {
        'sigma_x': 0.67,  # Longer bar major axis
        'sigma_y': 0.29,  # Shorter bar minor axis
        'sigma_z': 0.27,  # Thinner bar vertically
        'r_E': 8.2,       # Closer bar center
        's_max': 30.0,    # Max distance from us for averaging
        #'epsrel': 1e-6    # Relative error tolerance for integration
    }
    bar_angles = [0, 15, 20, 25, 29.4, 35, 40, 45]  # Different bar angles to compare
    colors = plt.cm.viridis(np.linspace(0, 1, len(bar_angles)))

    def quadratic(L,a,e,c,):
        return a*L**2 + e*L + c

    # best fit
    pparam, pparam_cov = spopt.curve_fit(quadratic, Longitude, Para, sigma=Paraerr, absolute_sigma = True)
    a,e,c, = pparam
    a_err, e_err, c_err = np.sqrt(np.diag(pparam_cov))

    Residuals = quadratic(Longitude, *pparam) - Para

    # generating plots; first is from analytic model
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(3,2,1)
    for i, bar_angle in enumerate(bar_angles):
        Aparallax = []

        for l in Al:
            try:
                parallax = bar_parallax3D(l, b, bar_angle, **custom_params)
                Aparallax.append(parallax if parallax else np.nan)
            except:
                Aparallax.append(np.nan)
            
        ax.plot(Al, Aparallax, 'o-', 
                color=colors[i], 
                label=f'{bar_angle}°' if bar_angle != 29.4 else f'{bar_angle}° (default)',
                linewidth=2, markersize=4, alpha=0.8)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, title='Bar Angle', bbox_to_anchor=(0.97, 1), loc='upper left', borderaxespad=0.)
    
    plt.ylim(0.08,0.20)

    # plots of Gaia data means w/ errors
    ax.errorbar(Longitude, Para, fmt = 'o', markersize=5, 
                yerr = Paraerr)
    #ax.errorbar(Longitude, final_dataPara, fmt = 'o', markersize=5, 
    #             yerr = final_dataParaerr)
    # quadratic regression
    model = np.poly1d(np.polyfit(Longitude, Para, 4))
    coefficients = np.polyfit(Longitude, Para, 4)
    polyline = np.arange(Longitude.min(),Longitude.max(),s)
    ax.plot(polyline, model(polyline), '--', color = 'b')
    ax.set_ylabel('Parallax')
    ax.set_xlabel('Longitude')
    ax.set_xlim(13,-13)
    ax.set_title(f'Mean Parallax vs. Galactic Longitude b={b:0.2f}')

    # residuals plots
    ax = fig.add_subplot(3,2,2)
    ax.scatter(Longitude,fraction)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('RC Fraction')
    ax.set_title('RC Fraction vs. Longitude')

    ax = fig.add_subplot(3,2,3)
    ax.scatter(Longitude,width)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('RC Width')
    ax.set_title('RC Width vs. Longitude')

    ax = fig.add_subplot(3,2,4)
    ax.scatter(Para,Residuals)
    ax.set_xlabel('Parallax')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs. Parallax')

    # count vs. longitude
    ax = fig.add_subplot(3,2,5)
    ax.scatter(Longitude, count)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Count of RC Stars")
    ax.set_title("Count of RC Stars vs. Longitude")

    # distance modulus vs longitude
    ax = fig.add_subplot(3,2,6)
    ax.scatter(Longitude,modulus)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Distance Modulus")
    ax.set_title("Distance Modulus vs. Longitude")

    plt.savefig(plotpath / f'mean_b_{b:0.2f}_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def Stotalplot(TotalGood,plotpath=None, s= 0.33):
    Longitude = TotalGood["Long"]
    Latitude = TotalGood["Latitude"]
    MeanMag = TotalGood["Mean Magnitude"]
    MeanCol = TotalGood["Mean Color"]
    Number_density = TotalGood["Count"]/0.33
    modulus = TotalGood["Distance Modulus"]
    fraction = TotalGood["Red Clump Fraction"]
    width = TotalGood["Red Clump Width"]
    mag_sigma = TotalGood["Magnitude Dispersion"]
    w_sigma = TotalGood["Parallax Dispersion"]
    

    fig = plt.figure(figsize=(20,20))

    #Number Density vs. longitude for each latitude
    ax = fig.add_subplot(3,3,1)
    sc = ax.scatter(Longitude, Number_density, c = Latitude)
    fig.colorbar(sc, ax=ax, label="Latitude")
    ax.set_xlim(10,-10)
    ax.set_ylim(2,5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("N_RC/deg^2")
    ax.set_title("Number Density vs. Longitude")

    #Distance Modulus vs. longitude for each latitude
    ax = fig.add_subplot(3,3,2)
    sc = ax.scatter(Longitude, modulus, c = Latitude)
    fig.colorbar(sc, ax=ax, label="Latitude")
    ax.set_xlim(10,-10)
    ax.set_ylim(-1,3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("mu")
    ax.set_title("Distance Modulus vs. Longitude")

    #fraction vs. longitude for each latitude
    ax = fig.add_subplot(3,3,3)
    sc = ax.scatter(Longitude, fraction, c = Latitude)
    fig.colorbar(sc, ax=ax, label="Latitude")
    ax.set_xlim(10,-10)
    ax.set_ylim(0,0.5)
    ax.set_xlabel("Longitude")
    ax.set_title("Red Clump Fraction vs. Longitude")

    #RC mag width vs. longitude for each latitude
    ax = fig.add_subplot(3,3,4)
    sc = ax.scatter(Longitude, Latitude, c = width)
    fig.colorbar(sc, ax=ax, label="Magnitude Dispersion")
    ax.set_xlim(10,-10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("RC width")

    #Mean Magnitude vs. longitude for each latitude
    ax = fig.add_subplot(3,3,5)
    sc = ax.scatter(Longitude, MeanMag, c = Latitude)
    fig.colorbar(sc, ax=ax, label="Latitude")
    ax.set_xlim(10,-10)
    ax.set_ylim(20.5,15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("G")
    ax.set_title("Mean Magnitude vs. Longitude")

    #Mean Color vs. longitude for each latitude
    ax = fig.add_subplot(3,3,6)
    sc = ax.scatter(Longitude, MeanCol, c = Latitude)
    fig.colorbar(sc, ax=ax, label="Latitude")
    ax.set_xlim(10,-10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("BP_RP")
    ax.set_title("Mean Color vs. Longitude")

    #Red Clump parallax dispersion by sightline
    ax = fig.add_subplot(3,3,7)
    sc = ax.scatter(Longitude, Latitude, c = w_sigma)
    fig.colorbar(sc, ax=ax, label="Parallax Dispersion")
    ax.set_xlim(10,-10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Parallax dispersion")

    plt.savefig(plotpath / f'totalplots_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def DRedClumpPlot(redclump, parallax, df, doubleclump, doubleparallax, plotpath=None, l=None, b=None, s= 0.33): # plotting routines
    
    xrc,yrc,sxrc,syrc = redclump["Params"]
    initial = redclump["initial cut"]
    C,D,rccol,cNrc,scrc = redclump["cpar"]
    chist = redclump["chist"]
    cbc = redclump["cbc"]
    A,B,rcmag,mNrc,smrc = redclump["mpar"]
    mhist = redclump["mhist"]
    mbc = redclump["mbc"]
    rcp,pNrc,sprc = parallax["ppar"]
    phist = parallax["phist"]
    pbc = parallax["pbc"]
    final_data = parallax["dataframe"]

    rcp1,pNrc1,sprc1,rcp2,pNrc2,sprc2 = doubleparallax["ppar"]
    dphist = doubleparallax["phist"]
    dpbc = doubleparallax["pbc"]
    double1 = doubleparallax["dataframe 1"]
    double2 = doubleparallax["dataframe 2"]
    E,F,rcmag1,mNrc1,smrc1,rcmag2,mNrc2,smrc2 = doubleclump["mpar"]
    dmbc = doubleclump["mbc"]
    dmhist = doubleclump["mhist"]

    fig = plt.figure(figsize=(20,20))

    #2d histogram
    ax = fig.add_subplot(4,3,1)
    ax.hist2d(df['bp_rp'],df['phot_g_mean_mag'],bins = 50)
    ax.scatter(xrc,yrc)
    ax.axvline(xrc-0.55*sxrc)
    ax.axvline(xrc+sxrc)
    ax.axhline(yrc-syrc)
    ax.axhline(yrc+1.45*syrc)
    ax.set_xlim(0.5,4)
    ax.set_ylim(20.5,14)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('2D Histogram')

    # initial CMD cut plotted over original CMD
    ax = fig.add_subplot(4,3,2)
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2) #original data
    ax.scatter(initial['bp_rp'],initial['phot_g_mean_mag'],s=0.5, c='yellow', alpha=0.5) # data after guesses
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram')

    # single clump magnitude histogram
    ax = fig.add_subplot(4,3,3)
    ax.errorbar(mbc,mhist,yerr=np.sqrt(mhist),fmt='o')
    G = np.linspace(min(initial['phot_g_mean_mag']),max(initial['phot_g_mean_mag']),num=200)
    NG = fit.rcmmodel(G,A,B,rcmag,mNrc,smrc)
    ax.plot(G,NG,label='Fitted')
    NGg = fit.gaussian(G,rcmag,mNrc,smrc)
    ax.plot(G,NGg, label = 'Gaussian')
    NGe = fit.exponential(G,A,B,rcmag)
    ax.plot(G,NGe, label = 'Exponential')
    ax.axvline(rcmag, color ='k', label= 'magpeak')
    ax.axvline(rcmag+smrc, color ='gray', linestyle = '--')
    ax.axvline(rcmag-smrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Magnitude Histogram')

    # double clump magnitude histogram
    ax = fig.add_subplot(4,3,4)
    ax.errorbar(dmbc,dmhist,yerr=np.sqrt(dmhist),fmt='o')
    G = np.linspace(min(initial['phot_g_mean_mag']),max(initial['phot_g_mean_mag']),num=200)
    NG = fit.doubleclumpmodel(G,E,F,rcmag1,mNrc1,smrc1,rcmag2,mNrc2,smrc2)
    ax.plot(G,NG,label='Fitted')
    NGg1 = fit.gaussian(G,rcmag1,mNrc1,smrc1)
    ax.plot(G,NGg1, label = 'Gaussian RC 1')
    NGg2 = fit.gaussian(G,rcmag2,mNrc2,smrc2)
    ax.plot(G,NGg2, label = 'Gaussian RC 2')
    NGe1 = fit.exponential(G,E,F,rcmag1)
    ax.plot(G,NGe1, label = 'Exponential')
    ax.axvline(rcmag1, color ='gray', label= 'magpeak 1')
    ax.axvline(rcmag1+smrc1, color ='gray', linestyle = '--')
    ax.axvline(rcmag1-smrc1, color ='gray', linestyle = '--')
    ax.axvline(rcmag2, color ='b', label= 'magpeak 2')
    ax.axvline(rcmag2+smrc2, color ='b', linestyle = '--')
    ax.axvline(rcmag2-smrc2, color ='b', linestyle = '--')
    ax.legend()
    ax.set_title('Magnitude Histogram')

    # color histogram
    ax = fig.add_subplot(4,3,5)
    ax.errorbar(cbc,chist,yerr=np.sqrt(chist),fmt='o')
    Col = np.linspace(min(initial['bp_rp']),max(initial['bp_rp']),num=200)
    NC = fit.rccmodel(Col,C,D,rccol,cNrc,scrc)
    ax.plot(Col,NC,label='Fitted')
    NCg = fit.gaussian(Col,rccol,cNrc,scrc)
    ax.plot(Col,NCg, label = 'Gaussian')
    NCe = fit.exponential(Col,C,D,rccol)
    ax.plot(Col,NCe, label = 'Exponential')
    ax.axvline(rccol, color ='k', label= 'magpeak')
    ax.axvline(rccol+scrc, color ='gray', linestyle = '--')
    ax.axvline(rccol-scrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Color Histogram')

    # single clump parallax distribution of final cut
    ax = fig.add_subplot(4,3,6)
    ax.errorbar(pbc,phist,yerr=np.sqrt(phist),fmt='o')
    P = np.linspace(-1,1,num=200)
    NP = fit.rcpmodel(P,rcp,pNrc,sprc)
    ax.plot(P,NP,label='Fitted')
    ax.axvline(rcp, color ='k', label= 'ppeak')
    ax.axvline(rcp+sprc, color ='gray', linestyle = '--')
    ax.axvline(rcp-sprc, color ='gray', linestyle = '--')
    ax.axvline(0.12195, color ='b', label = 'Galactic Center')
    ax.legend()
    ax.set_title('Single Clump Parallax Histogram')
    
    # double clump parallax distribution of final cut
    ax = fig.add_subplot(4,3,7)
    ax.errorbar(dpbc,dphist,yerr=np.sqrt(dphist),fmt='o')
    P = np.linspace(-1,1,num=200)
    NP = fit.doublepmodel(P,rcp1,pNrc1,sprc1,rcp2,pNrc2,sprc2)
    ax.plot(P,NP,label='Double Fitted')
    NP1 = fit.rcpmodel(P,rcp1,pNrc1,sprc1)
    ax.plot(P,NP1,label='Fitted 1')
    ax.axvline(rcp1, color ='gray', label= 'ppeak 1')
    ax.axvline(rcp1+sprc1, color ='gray', linestyle = '--')
    ax.axvline(rcp1-sprc1, color ='gray', linestyle = '--')
    NP2 = fit.rcpmodel(P,rcp2,pNrc2,sprc2)
    ax.plot(P,NP2,label='Fitted 2')
    ax.axvline(rcp2, color ='b', label= 'ppeak 2')
    ax.axvline(rcp2+sprc2, color ='b', linestyle = '--')
    ax.axvline(rcp2-sprc2, color ='b', linestyle = '--')
    ax.axvline(0.12195, color ='b', label = 'Galactic Center')
    ax.legend()
    ax.set_title('Double Clump Parallax Histogram')

    # final RC cut plotted over original CMD
    ax = fig.add_subplot(4,3,8)
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2)
    ax.scatter(double1['bp_rp'],double1['phot_g_mean_mag'],s=0.5, alpha=0.75, cmap='viridis')
    ax.scatter(double2['bp_rp'],double2['phot_g_mean_mag'],s=0.5, alpha=0.75, cmap ='plasma')
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_ylim(20.5,14)
    ax.set_xlim(.5,4)
    ax.set_title('Final Fit Color-Magnitude Selection')

    # magnitude vs. parallax
    ax = fig.add_subplot(4,3,9)
    ax.scatter(final_data["phot_g_mean_mag"], final_data["parallax"], s=2, alpha = 0.5)
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Parallax')
    ax.axhline(rcp, color ='b', label = 'single clump parallax mean')
    ax.axhline(rcp1, color ='k', label = 'double clump parallax mean 1', linestyle = '--')
    ax.axhline(rcp2, color ='g', label = 'double clump parallax mean 2', linestyle = '--')
    ax.set_ylim(-0.2,0.4)
    ax.set_xlim(14,20)

    # color vs. parallax
    ax = fig.add_subplot(4,3,10)
    ax.scatter(final_data["bp_rp"],final_data["parallax"], s=2, alpha = 0.5)
    ax.set_xlabel('Color')
    ax.set_ylabel('Parallax')
    ax.axhline(rcp, color ='b', label = 'single clump parallax mean')
    ax.axhline(rcp1, color ='k', label = 'double clump parallax mean 1', linestyle = '--')
    ax.axhline(rcp2, color ='g', label = 'double clump parallax mean 2', linestyle = '--')
    ax.set_ylim(-0.2,0.4)
    ax.set_xlim(0.5,4)

    plt.tight_layout()
    plt.savefig(plotpath/f'Plots_l_{l:0.2f}_b_{b:0.2f}_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def DRedClumpPlot_break(redclump, df, doubleclump, plotpath=None, l=None, b=None, s= 0.33): # plotting routines
    
    xrc,yrc,sxrc,syrc = redclump["Params"]
    initial = redclump["initial cut"]
    C,D,rccol,cNrc,scrc = redclump["cpar"]
    chist = redclump["chist"]
    cbc = redclump["cbc"]
    A,B,rcmag,mNrc,smrc = redclump["mpar"]
    mhist = redclump["mhist"]
    mbc = redclump["mbc"]

    E,F,rcmag1,mNrc1,smrc1,rcmag2,mNrc2,smrc2 = doubleclump["mpar"]
    dmbc = doubleclump["mbc"]
    dmhist = doubleclump["mhist"]

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(2,3,1)

    # 2d histogram
    ax.hist2d(df['bp_rp'],df['phot_g_mean_mag'],bins = 50)
    ax.scatter(xrc,yrc)
    ax.axvline(xrc-0.55*sxrc)
    ax.axvline(xrc+sxrc)
    ax.axhline(yrc-syrc)
    ax.axhline(yrc+1.45*syrc)
    ax.set_xlim(0.5,4)
    ax.set_ylim(20.5,14)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('2D Histogram')

    # initial CMD cut plotted over original CMD
    ax = fig.add_subplot(2,3,2)
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2) #original data
    ax.scatter(initial['bp_rp'],initial['phot_g_mean_mag'],s=0.5, c='yellow', alpha=0.5) # data after guesses
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram')

    # single clump magnitude histogram
    ax = fig.add_subplot(2,3,4)
    ax.errorbar(mbc,mhist,yerr=np.sqrt(mhist),fmt='o')
    G = np.linspace(min(initial['phot_g_mean_mag']),max(initial['phot_g_mean_mag']),num=200)
    NG = fit.rcmmodel(G,A,B,rcmag,mNrc,smrc)
    ax.plot(G,NG,label='Fitted')
    NGg = fit.gaussian(G,rcmag,mNrc,smrc)
    ax.plot(G,NGg, label = 'Gaussian')
    NGe = fit.exponential(G,A,B,rcmag)
    ax.plot(G,NGe, label = 'Exponential')
    ax.axvline(rcmag, color ='k', label= 'magpeak')
    ax.axvline(rcmag+smrc, color ='gray', linestyle = '--')
    ax.axvline(rcmag-smrc, color ='gray', linestyle = '--')
    ax.set_xlim(14,20)
    ax.set_ylim(0,500)
    ax.legend()
    ax.set_title('Magnitude Histogram')

    # double clump magnitude histogram
    ax = fig.add_subplot(2,3,5)
    ax.errorbar(dmbc,dmhist,yerr=np.sqrt(dmhist),fmt='o')
    G = np.linspace(min(initial['phot_g_mean_mag']),max(initial['phot_g_mean_mag']),num=200)
    NG = fit.doubleclumpmodel(G,E,F,rcmag1,mNrc1,smrc1,rcmag2,mNrc2,smrc2)
    ax.plot(G,NG,label='Fitted')
    NGg1 = fit.gaussian(G,rcmag1,mNrc1,smrc1)
    ax.plot(G,NGg1, label = 'Gaussian RC 1')
    NGg2 = fit.gaussian(G,rcmag2,mNrc2,smrc2)
    ax.plot(G,NGg2, label = 'Gaussian RC 2')
    NGe1 = fit.exponential(G,E,F,rcmag1)
    ax.plot(G,NGe1, label = 'Exponential')
    ax.axvline(rcmag1, color ='gray', label= 'magpeak 1')
    ax.axvline(rcmag1+smrc1, color ='gray', linestyle = '--')
    ax.axvline(rcmag1-smrc1, color ='gray', linestyle = '--')
    ax.axvline(rcmag2, color ='b', label= 'magpeak 1')
    ax.axvline(rcmag2+smrc2, color ='b', linestyle = '--')
    ax.axvline(rcmag2-smrc2, color ='b', linestyle = '--')
    ax.legend()
    ax.set_title('Magnitude Histogram')

    # color histogram
    ax = fig.add_subplot(2,3,6)
    ax.errorbar(cbc,chist,yerr=np.sqrt(chist),fmt='o')
    Col = np.linspace(min(initial['bp_rp']),max(initial['bp_rp']),num=200)
    NC = fit.rccmodel(Col,C,D,rccol,cNrc,scrc)
    ax.plot(Col,NC,label='Fitted')
    NCg = fit.gaussian(Col,rccol,cNrc,scrc)
    ax.plot(Col,NCg, label = 'Gaussian')
    NCe = fit.exponential(Col,C,D,rccol)
    ax.plot(Col,NCe, label = 'Exponential')
    ax.axvline(rccol, color ='k', label= 'magpeak')
    ax.axvline(rccol+scrc, color ='gray', linestyle = '--')
    ax.axvline(rccol-scrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Color Histogram')

    plt.tight_layout()
    plt.savefig(plotpath/f'Plots_l_{l:0.2f}_b_{b:0.2f}_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def DRedClumpPlot_runtime(redclump, df, plotpath=None, l=None, b=None, s= 0.33): # plotting routines
    
    xrc,yrc,sxrc,syrc = redclump["Params"] # import parameters from dictionary
    initial = redclump["initial cut"]
    

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(2,3,1)

    #2d histogram
    ax.hist2d(df['bp_rp'],df['phot_g_mean_mag'],bins = 50)
    ax.scatter(xrc,yrc)
    ax.axvline(xrc-0.55*sxrc)
    ax.axvline(xrc+sxrc)
    ax.axhline(yrc-syrc)
    ax.axhline(yrc+1.45*syrc)
    ax.set_xlim(0.5,4)
    ax.set_ylim(20.5,14)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('2D Histogram')

    #
    ax = fig.add_subplot(2,3,2) # initial CMD cut plotted over original CMD
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2) #original data
    ax.scatter(initial['bp_rp'],initial['phot_g_mean_mag'],s=0.5, c='yellow', alpha=0.5) # data after guesses
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram')

    plt.tight_layout()
    plt.savefig(plotpath/f'Plots_l_{l:0.2f}_b_{b:0.2f}_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def DRedClumpPlot_type(redclump, df, doubleclump, plotpath=None, l=None, b=None, s= 0.33): # plotting routines
    
    xrc,yrc,sxrc,syrc = redclump["Params"]
    initial = redclump["initial cut"]
    C,D,rccol,cNrc,scrc = redclump["cpar"]
    chist = redclump["chist"]
    cbc = redclump["cbc"]
    A,B,rcmag,mNrc,smrc = redclump["mpar"]
    mhist = redclump["mhist"]
    mbc = redclump["mbc"]
    
    E,F,rcmag1,mNrc1,smrc1,rcmag2,mNrc2,smrc2 = doubleclump["mpar"]
    dmbc = doubleclump["mbc"]
    dmhist = doubleclump["mhist"]
    

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(2,3,1)

    #2d histogram
    ax.hist2d(df['bp_rp'],df['phot_g_mean_mag'],bins = 50)
    ax.scatter(xrc,yrc)
    ax.axvline(xrc-0.55*sxrc)
    ax.axvline(xrc+sxrc)
    ax.axhline(yrc-syrc)
    ax.axhline(yrc+1.45*syrc)
    ax.set_xlim(0.5,4)
    ax.set_ylim(20.5,14)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('2D Histogram')

    # initial CMD cut plotted over original CMD
    ax = fig.add_subplot(2,3,2)
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2) #original data
    ax.scatter(initial['bp_rp'],initial['phot_g_mean_mag'],s=0.5, c='yellow', alpha=0.5) # data after guesses
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram')

    # single clump magnitude histogram
    ax = fig.add_subplot(2,3,3)
    ax.errorbar(mbc,mhist,yerr=np.sqrt(mhist),fmt='o')
    G = np.linspace(min(initial['phot_g_mean_mag']),max(initial['phot_g_mean_mag']),num=200)
    NG = fit.rcmmodel(G,A,B,rcmag,mNrc,smrc)
    ax.plot(G,NG,label='Fitted')
    NGg = fit.gaussian(G,rcmag,mNrc,smrc)
    ax.plot(G,NGg, label = 'Gaussian')
    NGe = fit.exponential(G,A,B,rcmag)
    ax.plot(G,NGe, label = 'Exponential')
    ax.axvline(rcmag, color ='k', label= 'magpeak')
    ax.axvline(rcmag+smrc, color ='gray', linestyle = '--')
    ax.axvline(rcmag-smrc, color ='gray', linestyle = '--')
    ax.set_xlim(14,20)
    ax.set_ylim(0,500)
    ax.legend()
    ax.set_title('Magnitude Histogram')

    # double clump magnitude histogram
    ax = fig.add_subplot(2,3,4)
    ax.errorbar(dmbc,dmhist,yerr=np.sqrt(dmhist),fmt='o')
    G = np.linspace(min(initial['phot_g_mean_mag']),max(initial['phot_g_mean_mag']),num=200)
    NG = fit.doubleclumpmodel(G,E,F,rcmag1,mNrc1,smrc1,rcmag2,mNrc2,smrc2)
    ax.plot(G,NG,label='Fitted')
    NGg1 = fit.gaussian(G,rcmag1,mNrc1,smrc1)
    ax.plot(G,NGg1, label = 'Gaussian RC 1')
    NGg2 = fit.gaussian(G,rcmag2,mNrc2,smrc2)
    ax.plot(G,NGg2, label = 'Gaussian RC 2')
    NGe1 = fit.exponential(G,E,F,rcmag1)
    ax.plot(G,NGe1, label = 'Exponential')
    ax.axvline(rcmag1, color ='gray', label= 'magpeak 1')
    ax.axvline(rcmag1+smrc1, color ='gray', linestyle = '--')
    ax.axvline(rcmag1-smrc1, color ='gray', linestyle = '--')
    ax.axvline(rcmag2, color ='b', label= 'magpeak 1')
    ax.axvline(rcmag2+smrc2, color ='b', linestyle = '--')
    ax.axvline(rcmag2-smrc2, color ='b', linestyle = '--')
    ax.legend()
    ax.set_title('Magnitude Histogram')

    # color histogram
    ax = fig.add_subplot(2,3,5)
    ax.errorbar(cbc,chist,yerr=np.sqrt(chist),fmt='o')
    Col = np.linspace(min(initial['bp_rp']),max(initial['bp_rp']),num=200)
    NC = fit.rccmodel(Col,C,D,rccol,cNrc,scrc)
    ax.plot(Col,NC,label='Fitted')
    NCg = fit.gaussian(Col,rccol,cNrc,scrc)
    ax.plot(Col,NCg, label = 'Gaussian')
    NCe = fit.exponential(Col,C,D,rccol)
    ax.plot(Col,NCe, label = 'Exponential')
    ax.axvline(rccol, color ='k', label= 'magpeak')
    ax.axvline(rccol+scrc, color ='gray', linestyle = '--')
    ax.axvline(rccol-scrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Color Histogram')

    plt.tight_layout()
    plt.savefig(plotpath/f'Plots_l_{l:0.2f}_b_{b:0.2f}_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def Dmeanplot(meanplot, Al, b=2.0, plotpath=None, s= 0.33):
    Longitude = meanplot["Long"]
    Para = meanplot["Mean Parallax"]
    Paraerr = meanplot["Mean Parallax Error"]
    MeanMag = meanplot["Mean Magnitude"]
    MeanCol = meanplot["Mean Color"]
    count = meanplot["Count"]
    modulus = meanplot["Distance Modulus"]
    fraction = meanplot["Red Clump Fraction"]
    width = meanplot["Red Clump Width"]

    #for comparison with analytic model
    # Test with different bar structure parameters
    custom_params = {
        'sigma_x': 0.67,  # Longer bar major axis
        'sigma_y': 0.29,  # Shorter bar minor axis
        'sigma_z': 0.27,  # Thinner bar vertically
        'r_E': 8.2,       # Closer bar center
        's_max': 30.0,    # Max distance from us for averaging
        #'epsrel': 1e-6    # Relative error tolerance for integration
    }
    bar_angles = [0, 15, 20, 25, 29.4, 35, 40, 45]  # Different bar angles to compare
    colors = plt.cm.viridis(np.linspace(0, 1, len(bar_angles)))

    def quadratic(L,a,e,c,):
        return a*L**2 + e*L + c
    
    # best fit
    pparam, pparam_cov = spopt.curve_fit(quadratic, Longitude, Para, sigma=Paraerr, absolute_sigma = True)
    a,e,c, = pparam
    a_err, e_err, c_err = np.sqrt(np.diag(pparam_cov))

    Residuals = quadratic(Longitude, *pparam) - Para

    # generating plots; first is from analytic model
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(3,2,1)
    for i, bar_angle in enumerate(bar_angles):
        Aparallax = []

        for l in Al:
            try:
                parallax = bar_parallax3D(l, b, bar_angle, **custom_params)
                Aparallax.append(parallax if parallax else np.nan)
            except:
                Aparallax.append(np.nan)
            
        ax.plot(Al, Aparallax, 'o-', 
                color=colors[i], 
                label=f'{bar_angle}°' if bar_angle != 29.4 else f'{bar_angle}° (default)',
                linewidth=2, markersize=4, alpha=0.8)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, title='Bar Angle', bbox_to_anchor=(0.97, 1), loc='upper left', borderaxespad=0.)
    
    plt.ylim(0.08,0.20)

    # plots of Gaia data means w/ errors
    ax.errorbar(Longitude, Para, fmt = 'o', markersize=5, 
                yerr = Paraerr)
    #ax.errorbar(Longitude, final_dataPara, fmt = 'o', markersize=5, 
    #             yerr = final_dataParaerr)
    # quadratic regression
    model = np.poly1d(np.polyfit(Longitude, Para, 4))
    coefficients = np.polyfit(Longitude, Para, 4)
    polyline = np.arange(-10,10,0.33)
    ax.plot(polyline, model(polyline), '--', color = 'b')
    ax.set_ylabel('Parallax')
    ax.set_xlabel('Longitude')
    ax.set_xlim(13,-13)
    ax.set_title(f'Mean Parallax vs. Galactic Longitude b={b:0.2f}')

    # residuals plots
    ax = fig.add_subplot(3,2,2)
    ax.scatter(Longitude,fraction)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('RC Fraction')
    ax.set_title('RC Fraction vs. Longitude')

    ax = fig.add_subplot(3,2,3)
    ax.scatter(Longitude,width)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('RC Width')
    ax.set_title('RC Width vs. Longitude')

    ax = fig.add_subplot(3,2,4)
    ax.scatter(Para,Residuals)
    ax.set_xlabel('Parallax')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs. Parallax')

    # count vs. longitude
    ax = fig.add_subplot(3,2,5)
    ax.scatter(Longitude, count)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Count of RC Stars")
    ax.set_title("Count of RC Stars vs. Longitude")

    # distance modulus vs longitude
    ax = fig.add_subplot(3,2,6)
    ax.scatter(Longitude,modulus)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Distance Modulus")
    ax.set_title("Distance Modulus vs. Longitude")

    plt.savefig(plotpath / f'mean_b_{b:0.2f}_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def Dtotalplot(TotalGood,plotpath=None, s= 0.33):
    Longitude = TotalGood["Long"]
    Latitude = TotalGood["Latitude"]
    MeanMag = TotalGood["Mean Magnitude"]
    MeanCol = TotalGood["Mean Color"]
    Number_density = TotalGood["Count"]/0.33
    modulus = TotalGood["Distance Modulus"]
    fraction = TotalGood["Red Clump Fraction"]
    width = TotalGood["Red Clump Width"]
    mag_sigma = TotalGood["Magnitude Dispersion"]
    w_sigma = TotalGood["Parallax Dispersion"]
    

    fig = plt.figure(figsize=(20,20))

    #Number Density vs. longitude for each latitude
    ax = fig.add_subplot(3,3,1)
    sc = ax.scatter(Longitude, Number_density, c = Latitude)
    fig.colorbar(sc, ax=ax, label="Latitude")
    ax.set_xlim(10,-10)
    ax.set_ylim(2,5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("N_RC/deg^2")
    ax.set_title("Number Density vs. Longitude")

    #Distance Modulus vs. longitude for each latitude
    ax = fig.add_subplot(3,3,2)
    sc = ax.scatter(Longitude, modulus, c = Latitude)
    fig.colorbar(sc, ax=ax, label="Latitude")
    ax.set_xlim(10,-10)
    ax.set_ylim(-1,3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("mu")
    ax.set_title("Distance Modulus vs. Longitude")

    #fraction vs. longitude for each latitude
    ax = fig.add_subplot(3,3,3)
    sc = ax.scatter(Longitude, fraction, c = Latitude)
    fig.colorbar(sc, ax=ax, label="Latitude")
    ax.set_xlim(10,-10)
    ax.set_ylim(0,0.5)
    ax.set_xlabel("Longitude")
    ax.set_title("Red Clump Fraction vs. Longitude")

    #RC mag width vs. longitude for each latitude
    ax = fig.add_subplot(3,3,4)
    sc = ax.scatter(Longitude, Latitude, c = width)
    fig.colorbar(sc, ax=ax, label="Magnitude Dispersion")
    ax.set_xlim(10,-10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("RC width")

    #Mean Magnitude vs. longitude for each latitude
    ax = fig.add_subplot(3,3,5)
    sc = ax.scatter(Longitude, MeanMag, c = Latitude)
    fig.colorbar(sc, ax=ax, label="Latitude")
    ax.set_xlim(10,-10)
    ax.set_ylim(20.5,15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("G")
    ax.set_title("Mean Magnitude vs. Longitude")

    #Mean Color vs. longitude for each latitude
    ax = fig.add_subplot(3,3,6)
    sc = ax.scatter(Longitude, MeanCol, c = Latitude)
    fig.colorbar(sc, ax=ax, label="Latitude")
    ax.set_xlim(10,-10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("BP_RP")
    ax.set_title("Mean Color vs. Longitude")

    #Red Clump parallax dispersion by sightline
    ax = fig.add_subplot(3,3,7)
    sc = ax.scatter(Longitude, Latitude, c = w_sigma)
    fig.colorbar(sc, ax=ax, label="Parallax Dispersion")
    ax.set_xlim(10,-10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Parallax dispersion")

    plt.savefig(plotpath / f'totalplots_s_{s:0.4f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()
