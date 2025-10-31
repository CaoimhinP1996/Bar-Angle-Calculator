import astropy.units as u # type: ignore
from zero_point import zpt
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from astropy.stats import sigma_clip # type: ignore
import scipy.optimize as spopt # type: ignore
import os
from pathlib import Path
from sklearn.cluster import KMeans # type: ignore
zpt.load_tables()


def findrc(df): #automate finding the RC from original CMD using a 2D histogram that locates two maxima, 
                # then assign the top right maxima as RC
    # Your 2D data
    X = np.array([df['bp_rp'],df['phot_g_mean_mag']]).transpose()  # shape (n_samples, 2)

    # Find 2 cluster centers
    kmeans = KMeans(n_clusters=2, random_state=42, n_init = 2, algorithm = "elkan")
    labels = kmeans.fit_predict(X)
    spreads = [(np.std(X[labels==i], axis=0)) for i in range(2)]
    centers = kmeans.cluster_centers_
    x1,y1 = centers[0][0],centers[0][1]
    x2,y2 = centers[1][0],centers[1][1]
    sx1,sy1 = spreads[0][0],spreads[0][1]
    sx2,sy2 = spreads[1][0],spreads[1][1]

    if (y1 < y2): # automates selection of the RC from the two maxima; RC will always be brighter
        xrc = x1
        yrc = y1
        sxrc = sx1
        syrc = sy1
    else:
        xrc = x2
        yrc = y2
        sxrc = sx2
        syrc = sy2
    return xrc,yrc,sxrc,syrc

def extractrc_v1(df,xrc,yrc,sxrc,syrc): # make initial cuts from original dataframe using coordinates and spread from histogram
    df1 = (
        df.loc[(df['bp_rp']>xrc-0.5*sxrc) & (df['bp_rp']<xrc+sxrc) & (df['phot_g_mean_mag']>yrc-syrc) & (df['phot_g_mean_mag']<yrc+1.4*syrc)]
    )
    return df1

def extractrc_v2(df,xrc,yrc,sxrc,syrc): # make initial cuts from original dataframe using coordinates and spread from histogram
    df1 = (
        df.loc[(df['bp_rp']>xrc-0.8*sxrc) & (df['bp_rp']<xrc+1.3*sxrc) & (df['phot_g_mean_mag']>yrc-syrc) 
               & (df['phot_g_mean_mag']<yrc+1.6*syrc)])
    return df1

def rcmmodel(x,A,B,rcmag,Nrc,smrc): # function for fitting magnitude according to initial cut
    return A*np.exp(B*(x-rcmag)) + \
           Nrc/(np.sqrt(2*np.pi*smrc**2))*np.exp(-0.5*(x-rcmag)**2/smrc**2)

def rccmodel(x,C,D,rccol,Nrc,scrc): # function for fitting color according to initial cut
    return C*np.exp(D*(x-rccol)) + \
           Nrc/(np.sqrt(2*np.pi*scrc**2))*np.exp(-0.5*(x-rccol)**2/scrc**2)

def redclumpfinder_v1(df): # 1D histogram of magnitude and color in the dataframe to be matched over the fits to check accuracy of fits
    xrc,yrc,sxrc,syrc = findrc(df)

    df1 = extractrc_v1(df,xrc,yrc,sxrc,syrc)

    mhist,mbe = np.histogram(df1['phot_g_mean_mag'], bins=40)
    mbc = 0.5*(mbe[0:-1]+mbe[1:])
    mguess = [(mhist[0]+mhist[-1])/2,0.5,(np.max(mbc)+np.min(mbc))/2,(mhist[0]+mhist[-1])/2,0.3]
    mpar,mcov = spopt.curve_fit(rcmmodel,mbc,mhist,p0=mguess,sigma=np.sqrt(mhist))

    chist,cbe = np.histogram(df1['bp_rp'], bins=40)
    cbc = 0.5*(cbe[0:-1]+cbe[1:])
    cguess = [(chist[0]+chist[-1])/2,0.5,(np.max(cbc)+np.min(cbc))/2,(chist[0]+chist[-1])/2,0.3]
    cpar,ccov = spopt.curve_fit(rccmodel,cbc,chist,p0=cguess,sigma=np.sqrt(chist))

    A,B,rcmag,mNrc,smrc = mpar
    C,D,rccol,cNrc,scrc = cpar
    
    df2 = ( # make final cuts using mean and sigma found from fitting
    df.loc[(df['bp_rp']> rccol - scrc) & (df['bp_rp']< rccol + scrc) & 
    (df['phot_g_mean_mag']< rcmag + smrc) & (df['phot_g_mean_mag']>rcmag - smrc)]
    )

    output = {"Params": [xrc,yrc,sxrc,syrc], #input parameters for plotting into a directory
              "initial cut": df1,
              "fit cut": df2,
              "mhist": mhist,
              "mbc": mbc,
              "mbe": mbe,
              "mguess": mguess,
              "mpar": mpar,
              "mcov": mcov,
              "chist": chist,
              "cbe": cbe,
              "cbc": cbc,
              "cguess": cguess,
              "cpar": cpar,
              "ccov": ccov,}
    return output

def redclumpfinder_v2(df): # 1D histogram of magnitude and color in the dataframe to be matched over the fits to check accuracy of fits
    xrc,yrc,sxrc,syrc = findrc(df)

    df1 = extractrc_v2(df,xrc,yrc,sxrc,syrc)

    mhist,mbe = np.histogram(df1['phot_g_mean_mag'], bins=40)
    mbc = 0.5*(mbe[0:-1]+mbe[1:])
    mguess = [(mhist[0]+mhist[-1])/2,0.5,(np.max(mbc)+np.min(mbc))/2,(mhist[0]+mhist[-1])/2,0.3]
    mpar,mcov = spopt.curve_fit(rcmmodel,mbc,mhist,p0=mguess,sigma=np.sqrt(mhist))

    chist,cbe = np.histogram(df1['bp_rp'], bins=40)
    cbc = 0.5*(cbe[0:-1]+cbe[1:])
    cguess = [(chist[0]+chist[-1])/2,0.5,(np.max(cbc)+np.min(cbc))/2,(chist[0]+chist[-1])/2,0.3]
    cpar,ccov = spopt.curve_fit(rccmodel,cbc,chist,p0=cguess,sigma=np.sqrt(chist))

    A,B,rcmag,mNrc,smrc = mpar
    C,D,rccol,cNrc,scrc = cpar
    
    df2 = ( # make final cuts using mean and sigma found from fitting
    df.loc[(df['bp_rp']> rccol - scrc) & (df['bp_rp']< rccol + scrc) & 
    (df['phot_g_mean_mag']< rcmag + smrc) & (df['phot_g_mean_mag']>rcmag - smrc)]
    )

    output = {"Params": [xrc,yrc,sxrc,syrc], #input parameters for plotting into a directory
              "initial cut": df1,
              "fit cut": df2,
              "mhist": mhist,
              "mbc": mbc,
              "mbe": mbe,
              "mguess": mguess,
              "mpar": mpar,
              "mcov": mcov,
              "chist": chist,
              "cbe": cbe,
              "cbc": cbc,
              "cguess": cguess,
              "cpar": cpar,
              "ccov": ccov,}
    return output

def zeropoint(redclump): # calculate zero point for each star using individual parameters
    df = redclump["fit cut"]
    zero_point = zpt.get_zpt(df["phot_g_mean_mag"], df["nu_eff_used_in_astrometry"], df["pseudocolour"], df["ecl_lat"], 
                df["astrometric_params_solved"])
    return zero_point

#def zeropointavg(redclump): # calculate zero point for each star using average parameters
    #df = redclump["fit cut"]
    #df = pd.DataFrame(data=df)
    #phot = df["phot_g_mean_mag"].shape
    #nu = df["nu_eff_used_in_astrometry"].shape
    #pseudo = df["pseudocolour"].shape
    #ecl = df["ecl_lat"].shape
    #astro_params = df["astrometric_params_solved"].shape
    #clump_average_g_mean_mag = np.array([df["phot_g_mean_mag"].mean()*np.ones([phot[0]])])
    #nu_eff_average_g_mean_mag = np.array([df["nu_eff_used_in_astrometry"].mean()*np.ones([nu[0]])])
    #pseudocolor_average_g_mean_mag = np.array([df["pseudocolour"].mean()*np.ones([pseudo[0]])])
    #ecl_lat_average_g_mean_mag = np.array([df["ecl_lat"].mean()*np.ones([ecl[0]])])
    #params_average_g_mean_mag = np.array([df["astrometric_params_solved"].mean()*np.ones([astro_params[0]])])
    #zero_point = zpt.get_zpt(clump_average_g_mean_mag, nu_eff_average_g_mean_mag, pseudocolor_average_g_mean_mag, 
    #            ecl_lat_average_g_mean_mag, df["astrometric_params_solved"]);
    #return zero_point

def rczmodel(x,rcz,zNrc,szrc): # function for fitting parallax w/ zero point
    return zNrc/(np.sqrt(2*np.pi*szrc**2))*np.exp(-0.5*(x-rcz)**2/szrc**2)

def rcpmodel(x,rcp,pNrc,sprc): # function for fitting parallax w/ zero point
    return pNrc/(np.sqrt(2*np.pi*sprc**2))*np.exp(-0.5*(x-rcp)**2/sprc**2)

def finalfit(redclump, zero_point): # find peak parallax along sightline from fitting distribution of parallax before 
                                    # and after applying zero point
    prezp = redclump["fit cut"]
    prezp["zero point"] = zero_point
    postzp = prezp["parallax"] - zero_point
    zp = {"parallax": postzp, "parallax error": prezp["parallax_error"], "zero point": zero_point}
    zp = pd.DataFrame(data=zp)
    clipped = sigma_clip(np.array(prezp["parallax"]), sigma=3, maxiters=5)
    prezp = prezp[~clipped.mask]
    clipped = sigma_clip(np.array(zp["parallax"]), sigma=3, maxiters=5)
    zp = zp[~clipped.mask]

    zhist,zbe = np.histogram(zp["parallax"], bins=40)
    zbc = 0.5*(zbe[0:-1]+zbe[1:])
    zguess = [np.mean(zp["parallax"]),np.max(zhist)/np.sqrt(2*np.pi*np.std(zp["parallax"])**2),np.std(zp["parallax"])]
    zpar,zcov = spopt.curve_fit(rczmodel,zbc,zhist,p0=zguess,sigma=np.sqrt(zhist))

    phist,pbe = np.histogram(prezp["parallax"], bins=40)
    pbc = 0.5*(pbe[0:-1]+pbe[1:])
    pguess = [np.mean(prezp["parallax"]),np.max(phist)/np.sqrt(2*np.pi*np.std(prezp["parallax"])**2),np.std(prezp["parallax"])]
    ppar,pcov = spopt.curve_fit(rcpmodel,pbc,phist,p0=pguess,sigma=np.sqrt(phist))

    rcz,zNrc,szrc = zpar
    rcp,pNrc,sprc = ppar

    fitzp = (zp.loc[(zp["parallax"]> rcz - szrc) & (zp["parallax"]< rcz + szrc)])

    fitprezp = (prezp.loc[(prezp["parallax"]> rcp - sprc) & (prezp["parallax"]< rcp + sprc)])

    output = {"zhist": zhist,
              "zbc": zbc,
              "zbe": zbe,
              "zguess": zguess,
              "zpar": zpar,
              "zcov": zcov,
              "phist": phist,
              "pbc": pbc,
              "pbe": pbe,
              "pguess": pguess,
              "ppar": ppar,
              "pcov": pcov,
              "post-zp pre-fit dataframe": zp,
              "post-zp dataframe": fitzp,
              "pre-zp dataframe": fitprezp,
              "pre-zp pre-fit dataframe": prezp,
              "zp parallax peak": rcz,
              "prezp parallax peak": rcp,
              "zp parallax error": szrc,
              "prezp parallax error": sprc,
              "mean error w/ zero point": szrc/np.sqrt(len(fitzp["parallax"])),
              "mean error w/o zero point": sprc/np.sqrt(len(fitprezp["parallax"]))}
    return output

def finalcut(redclump, zero_point): # sigma clip based on parallax to further isolate the bar red clump and find mean parallax
                                    # for the sightline
    prezp = redclump["fit cut"]
    postzp = prezp["parallax"] - zero_point
    zp = {"parallax": postzp, "parallax error": prezp["parallax_error"]}
    # need to properly save parallax error here before sigma clipping so that the associated values for each
    # star stay properly associated after sigma clipping, zero point and pre-zp sigma clipping will not 
    # necessarily clip the same stars

    zp = pd.DataFrame(data=zp)
    clipped = sigma_clip(np.array(prezp["parallax"]), sigma=3, maxiters=5)
    prezp = prezp[~clipped.mask]
    clipped = sigma_clip(np.array(zp["parallax"]), sigma=3, maxiters=5)
    zp = zp[~clipped.mask]
    zparerr = 1/zp["parallax error"]**2
    zparerrmean = np.sqrt(1/np.sum(zparerr))
    zparmean = np.sum(zparerr*zp["parallax"]/np.sum(zparerr))
    prezparerr = 1/prezp["parallax_error"]**2
    prezparerrmean = np.sqrt(1/np.sum(prezparerr))
    prezparmean = np.sum(prezparerr*prezp["parallax"]/np.sum(prezparerr))
    output = {"zp parallax mean error": zparerrmean,
              "zp parallax mean": zparmean,
              "post-zp dataframe": zp,
              "pre-zp parallax mean error": prezparerrmean,
              "pre-zp parallax mean": prezparmean,
              "pre-zp dataframe": prezp}
    return output
