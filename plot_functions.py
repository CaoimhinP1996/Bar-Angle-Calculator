import astropy.units as u # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import pytz # type: ignore
from astropy.stats import sigma_clip # type: ignore
import scipy.optimize as spopt # type: ignore
import os
from pathlib import Path
import fit_functions_copy2 as fit_functions
from bar_distance_analytic_model import compute_bar_distance

def RedClumpPlot(redclump, parallax, df, zero_point, plotpath=None, l=None, b=None): # plotting routines
    
    xrc,yrc,sxrc,syrc = redclump["Params"] # import parameters from dictionary
    initial = redclump["initial cut"]
    A,B,rcmag,mNrc,smrc = redclump["mpar"]
    C,D,rccol,cNrc,scrc = redclump["cpar"]
    mhist = redclump["mhist"]
    chist = redclump["chist"]
    mbc = redclump["mbc"]
    cbc = redclump["cbc"]
    mguess = redclump["mguess"]
    cguess = redclump["cguess"]
    E,F,rczp,zpNrc,szprc = parallax["zppar"]
    H,I,rcp,pNrc,sprc = parallax["ppar"]
    zphist = parallax["zphist"]
    phist = parallax["phist"]
    zpbc = parallax["zpbc"]
    pbc = parallax["pbc"]
    zpguess = parallax["zpguess"]
    pguess = parallax["pguess"]
    prezp = parallax["pre-zp pre-fit dataframe"]
    

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(3,3,1)

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
    ax = fig.add_subplot(3,3,4) # initial CMD cut plotted over original CMD
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2) #original data
    ax.scatter(initial['bp_rp'],initial['phot_g_mean_mag'],s=0.5, c='yellow', alpha=0.5) # data after guesses
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram')

    # magnitude histogram
    ax = fig.add_subplot(3,3,5)
    ax.errorbar(mbc,mhist,yerr=np.sqrt(mhist),fmt='o')
    G = np.linspace(min(initial['phot_g_mean_mag']),max(initial['phot_g_mean_mag']),num=200)
    NG = fit_functions.rcmmodel(G,A,B,rcmag,mNrc,smrc)
    ax.plot(G,NG,label='Fitted')
    NG_init = fit_functions.rcmmodel(G,*mguess)
    ax.plot(G,NG_init, label = 'Guess')
    ax.axvline(rcmag, color ='k', label= 'magpeak')
    ax.axvline(rcmag+smrc, color ='gray', linestyle = '--')
    ax.axvline(rcmag-smrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Magnitude Histogram')

    # color histogram
    ax = fig.add_subplot(3,3,6)
    ax.errorbar(cbc,chist,yerr=np.sqrt(chist),fmt='o')
    Col = np.linspace(min(initial['bp_rp']),max(initial['bp_rp']),num=200)
    NC = fit_functions.rccmodel(Col,C,D,rccol,cNrc,scrc)
    ax.plot(Col,NC,label='Fitted')
    NC_init = fit_functions.rccmodel(Col,*cguess)
    ax.plot(Col,NC_init, label = 'Guess')
    ax.axvline(rccol, color ='k', label= 'magpeak')
    ax.axvline(rccol+scrc, color ='gray', linestyle = '--')
    ax.axvline(rccol-scrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Color Histogram')

    # final RC cut plotted over original CMD
    ax = fig.add_subplot(3,3,7)
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2)
    ax.scatter(prezp['bp_rp'],prezp['phot_g_mean_mag'],s=0.5, alpha=0.75, c= zero_point, cmap='autumn')
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_ylim(20.5,14)
    ax.set_xlim(.5,4)
    ax.set_title('1\u03C3 Color-Magnitude Selection')

    # parallax distribution of final cut w/ zero point correction
    ax = fig.add_subplot(3,3,8)
    ax.errorbar(zpbc,zphist,yerr=np.sqrt(zphist),fmt='o')
    ZP = np.linspace(-1,1,num=200)
    NZP = fit_functions.zpcut(ZP,E,F,rczp,zpNrc,szprc)
    ax.plot(ZP,NZP,label='Fitted')
    NZP_init = fit_functions.zpcut(ZP,*zpguess)
    ax.plot(ZP,NZP_init, label = 'Guess')
    ax.axvline(rczp, color ='k', label= 'zppeak')
    ax.axvline(rczp+szprc, color ='gray', linestyle = '--')
    ax.axvline(rczp-szprc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Parallax Histogram w/ Zero Point')

    # parallax distribution of final cut w/o zero point correction
    ax = fig.add_subplot(3,3,9)
    ax.errorbar(pbc,phist,yerr=np.sqrt(phist),fmt='o')
    P = np.linspace(-1,1,num=200)
    NP = fit_functions.pcut(P,H,I,rcp,pNrc,sprc)
    ax.plot(P,NP,label='Fitted')
    NP_init = fit_functions.pcut(P,*pguess)
    ax.plot(P,NP_init, label = 'Guess')
    ax.axvline(rcp, color ='k', label= 'ppeak')
    ax.axvline(rcp+sprc, color ='gray', linestyle = '--')
    ax.axvline(rcp-sprc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Parallax Histogram w/o Zero Point')


    plt.tight_layout()
    plt.savefig(plotpath/f'Plots_l_{l:0.2f}_b_{b:0.2f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def RedClumpPlot_break(redclump, df, plotpath=None, l=None, b=None): # plotting routines
    
    xrc,yrc,sxrc,syrc = redclump["Params"] # import parameters from dictionary
    initial = redclump["initial cut"]
    A,B,rcmag,mNrc,smrc = redclump["mpar"]
    C,D,rccol,cNrc,scrc = redclump["cpar"]
    mhist = redclump["mhist"]
    chist = redclump["chist"]
    mbc = redclump["mbc"]
    cbc = redclump["cbc"]
    mguess = redclump["mguess"]
    cguess = redclump["cguess"]

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
    ax = fig.add_subplot(2,3,4) # initial CMD cut plotted over original CMD
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2) #original data
    ax.scatter(initial['bp_rp'],initial['phot_g_mean_mag'],s=0.5, c='yellow', alpha=0.5) # data after guesses
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram')

    # magnitude histogram
    ax = fig.add_subplot(2,3,5)
    ax.errorbar(mbc,mhist,yerr=np.sqrt(mhist),fmt='o')
    G = np.linspace(min(initial['phot_g_mean_mag']),max(initial['phot_g_mean_mag']),num=200)
    NG = fit_functions.rcmmodel(G,A,B,rcmag,mNrc,smrc)
    ax.plot(G,NG,label='Fitted')
    NG_init = fit_functions.rcmmodel(G,*mguess)
    ax.plot(G,NG_init, label = 'Guess')
    ax.axvline(rcmag, color ='k', label= 'magpeak')
    ax.axvline(rcmag+smrc, color ='gray', linestyle = '--')
    ax.axvline(rcmag-smrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Magnitude Histogram')

    # color histogram
    ax = fig.add_subplot(2,3,6)
    ax.errorbar(cbc,chist,yerr=np.sqrt(chist),fmt='o')
    Col = np.linspace(min(initial['bp_rp']),max(initial['bp_rp']),num=200)
    NC = fit_functions.rccmodel(Col,C,D,rccol,cNrc,scrc)
    ax.plot(Col,NC,label='Fitted')
    NC_init = fit_functions.rccmodel(Col,*cguess)
    ax.plot(Col,NC_init, label = 'Guess')
    ax.axvline(rccol, color ='k', label= 'magpeak')
    ax.axvline(rccol+scrc, color ='gray', linestyle = '--')
    ax.axvline(rccol-scrc, color ='gray', linestyle = '--')
    ax.legend()
    ax.set_title('Color Histogram')

    plt.tight_layout()
    plt.savefig(plotpath/f'Plots_l_{l:0.2f}_b_{b:0.2f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def RedClumpPlot_runtime(redclump, df, plotpath=None, l=None, b=None): # plotting routines
    
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
    ax = fig.add_subplot(2,3,4) # initial CMD cut plotted over original CMD
    ax.scatter(df['bp_rp'],df['phot_g_mean_mag'],s=0.5, alpha = 0.2) #original data
    ax.scatter(initial['bp_rp'],initial['phot_g_mean_mag'],s=0.5, c='yellow', alpha=0.5) # data after guesses
    ax.set_ylim(20.5,14)
    ax.set_xlim(0.5,4)
    ax.set_xlabel('Color')
    ax.set_ylabel('G')
    ax.set_title('Color-Magnitude Diagram')

    plt.tight_layout()
    plt.savefig(plotpath/f'Plots_l_{l:0.2f}_b_{b:0.2f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()

def meanplot(meanplot, Al, b=2.0, plotpath=None):
    Longitude = meanplot["Long"]
    ZpPara = meanplot["Mean Parallax w/ Zero Point"]
    ZpParaerr = meanplot["Mean Parallax Error w/ Zero Point"]
    PreZpPara = meanplot["Mean Parallax w/o Zero Point"]
    PreZpParaerr = meanplot["Mean Parallax Error w/o Zero Point"]

    #for comparison with analytic model
    # Test with different bar structure parameters
    custom_params = {
        'sigma_x': 0.67,  # Longer bar major axis
        'sigma_y': 0.29,  # Shorter bar minor axis
        'sigma_z': 0.27,  # Thinner bar vertically
        'r_E': 8.2,       # Closer bar center
        's_max': 30.0,     # Max distance from us for averaging
        #'epsrel': 1e-6    # Relative error tolerance for integration
    }
    bar_angles = [0, 15, 20, 25, 29.4, 35, 40, 45]  # Different bar angles to compare
    colors = plt.cm.viridis(np.linspace(0, 1, len(bar_angles)))

    for i, bar_angle in enumerate(bar_angles):
        distances = []

        for l in Al:
            try:
                distance = compute_bar_distance(l, b, bar_angle, **custom_params)
                distances.append(distance)
            except:
                distances.append(np.nan)
            
        plt.plot(Al, 1/np.array(distances), 'o-', 
                color=colors[i], 
                label=f'{bar_angle}°' if bar_angle != 29.4 else f'{bar_angle}°',
                linewidth=2, markersize=4, alpha=0.8)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, title='Bar Angle')
    
    plt.ylim(0.08,0.20)

    plt.errorbar(Longitude, ZpPara, fmt = 'o', markersize=5, 
                 yerr = ZpParaerr)
    plt.errorbar(Longitude, PreZpPara, fmt = 'o', markersize=5, 
                 yerr = PreZpParaerr)
    plt.ylabel('Parallax')
    plt.xlabel('Longitude')
    plt.gca().invert_xaxis()
    plt.title(f'Mean Parallax vs. Galactic Longitude b={b:0.2f}')
    plt.savefig(plotpath / f'mean_b_{b:0.2f}.jpg')
    plt.cla()
    plt.clf()
    plt.close()