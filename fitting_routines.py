import astropy.units as u # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from astropy.stats import sigma_clip # type: ignore
import scipy.optimize as spopt # type: ignore
from sklearn.cluster import KMeans # type: ignore
from bar_parallax_analytic_model import parallax_model
from scipy import integrate
from astropy.stats import knuth_bin_width

def findrc(df): # automate finding the RC from original CMD using a 2D histogram that locates two maxima, 
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

def rcmmodel(x,A,B,rcmag,mNrc,smrc): # function for fitting magnitude according to initial cut
    return A*np.exp(B*(x-rcmag)) + \
           mNrc/(np.sqrt(2*np.pi*smrc**2))*np.exp(-0.5*(x-rcmag)**2/smrc**2)

def rccmodel(x,A,B,rccol,cNrc,scrc): # function for fitting magnitude according to initial cut
    return A*np.exp(B*(x-rccol)) + \
           cNrc/(np.sqrt(2*np.pi*scrc**2))*np.exp(-0.5*(x-rccol)**2/scrc**2)

def gaussian(x,rccol,Nrc,scrc):
    return Nrc/(np.sqrt(2*np.pi*scrc**2))*np.exp(-0.5*(x-rccol)**2/scrc**2)

def exponential(x,A,B,rccol):
    return A*np.exp(B*(x-rccol))

def redclumpfinder_v1(df): # 1D histogram of magnitude and color in the dataframe to be matched over the fits to check accuracy of fits
    xrc,yrc,sxrc,syrc = findrc(df)

    df1 = extractrc_v1(df,xrc,yrc,sxrc,syrc)

    mbin_width, mbin_edges = knuth_bin_width(df1["phot_g_mean_mag"], return_bins=True)
    cbin_width, cbin_edges = knuth_bin_width(df1["bp_rp"], return_bins=True)

    mhist,mbe = np.histogram(df1['phot_g_mean_mag'], bins=mbin_edges)
    mbc = 0.5*(mbe[0:-1]+mbe[1:])
    mguess = [(mhist[0]+mhist[-1])/2,0.5,(np.max(mbc)+np.min(mbc))/2,(mhist[0]+mhist[-1])/2,0.3]
    bounds = ([-np.inf,-np.inf,14,0,0],[np.inf,np.inf,19,1000,1.5])
    mpar,mcov = spopt.curve_fit(rcmmodel,mbc,mhist,p0=mguess,sigma=np.sqrt(mhist+0.5),absolute_sigma=True, bounds = bounds)

    chist,cbe = np.histogram(df1['bp_rp'], bins=cbin_edges)
    cbc = 0.5*(cbe[0:-1]+cbe[1:])
    cguess = [(chist[0]+chist[-1])/2,0.5,(np.max(cbc)+np.min(cbc))/2,(chist[0]+chist[-1])/2,0.3]
    bounds = ([-np.inf,-np.inf,0,0,0],[np.inf,np.inf,4,1000,1.5])
    cpar,ccov = spopt.curve_fit(rccmodel,cbc,chist,p0=cguess,sigma=np.sqrt(chist+0.5),absolute_sigma=True, bounds=bounds)

    A,B,rcmag,mNrc,smrc = mpar
    C,D,rccol,cNrc,scrc = cpar

    def Maggaussian(x,rcmag=rcmag,mNrc=mNrc,smrc=smrc):
        return mNrc/(np.sqrt(2*np.pi*smrc**2))*np.exp(-0.5*(x-rcmag)**2/smrc**2)
    
    def Colgaussian(x,rccol=rccol,cNrc=cNrc,scrc=scrc):
        return cNrc/(np.sqrt(2*np.pi*scrc**2))*np.exp(-0.5*(x-rccol)**2/scrc**2)
    
    def Magexponential(x,A=A,B=B,rcmag=rcmag):
        return A*np.exp(B*(x-rcmag))
    
    def Colexponential(x,C=C,D=D,rccol=rccol):
        return C*np.exp(D*(x-rccol))
    
    def Magmodel(x,A=A,B=B,rcmag=rcmag,mNrc=mNrc,smrc=smrc): 
        return A*np.exp(B*(x-rcmag)) + mNrc/(np.sqrt(2*np.pi*smrc**2))*np.exp(-0.5*(x-rcmag)**2/smrc**2)

    mg1area, mg1error = integrate.quad(Maggaussian, rcmag - smrc, rcmag + smrc)
    mg2area, mg2error = integrate.quad(Maggaussian, rcmag - 3*smrc, rcmag + 3*smrc)
    mearea, meerror = integrate.quad(Magexponential, rcmag - smrc, rcmag + smrc)
    mfarea, mferror = integrate.quad(Magmodel, rcmag - 3*smrc, rcmag + 3*smrc)
    mag_distinctness = mg1area/mearea
    RC_fraction = mg2area/mfarea
    cgarea, cgerror = integrate.quad(Colgaussian, rccol - scrc, rccol + scrc)
    cearea, ceerror = integrate.quad(Colexponential, rccol - scrc, rccol + scrc)
    col_distinctness = cgarea/cearea
    
    df2 = ( # make final cuts using mean and sigma found from fitting
    df.loc[(df['bp_rp']> rccol - scrc) & (df['bp_rp']< rccol + scrc) & 
    (df['phot_g_mean_mag']< rcmag + smrc) & (df['phot_g_mean_mag']>rcmag - smrc)]
    )

    RC_width = (df2["phot_g_mean_mag"].max() - df2["phot_g_mean_mag"].min())/2

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
              "ccov": ccov,
              "mag": rcmag,
              "mag sigma": smrc,
              "Number at mag mean": mNrc,
              "color": rccol,
              "color sigma": scrc,
              "Number at color mean": cNrc,
              "Magnitude Distinctness": mag_distinctness,
              "Color Distinctness": col_distinctness,
              "Red Clump Fraction": RC_fraction,
              "Red Clump Width": RC_width}
    return output

def redclumpfinder_v2(df): # 1D histogram of magnitude and color in the dataframe to be matched over the fits to check accuracy of fits
    xrc,yrc,sxrc,syrc = findrc(df)

    df1 = extractrc_v2(df,xrc,yrc,sxrc,syrc)

    mbin_width, mbin_edges = knuth_bin_width(df1["phot_g_mean_mag"], return_bins=True)
    cbin_width, cbin_edges = knuth_bin_width(df1["bp_rp"], return_bins=True)

    mhist,mbe = np.histogram(df1['phot_g_mean_mag'], bins=mbin_edges)
    mbc = 0.5*(mbe[0:-1]+mbe[1:])
    mguess = [(mhist[0]+mhist[-1])/2,0.5,(np.max(mbc)+np.min(mbc))/2,(mhist[0]+mhist[-1])/2,0.3]
    bounds = ([-np.inf,-np.inf,14,0,0],[np.inf,np.inf,19,1000,1.5])
    mpar,mcov = spopt.curve_fit(rcmmodel,mbc,mhist,p0=mguess,sigma=np.sqrt(mhist+0.5),absolute_sigma=True, bounds = bounds)

    chist,cbe = np.histogram(df1['bp_rp'], bins=cbin_edges)
    cbc = 0.5*(cbe[0:-1]+cbe[1:])
    cguess = [(chist[0]+chist[-1])/2,0.5,(np.max(cbc)+np.min(cbc))/2,(chist[0]+chist[-1])/2,0.3]
    bounds = ([-np.inf,-np.inf,0,0,0],[np.inf,np.inf,4,1000,1.5])
    cpar,ccov = spopt.curve_fit(rccmodel,cbc,chist,p0=cguess,sigma=np.sqrt(chist+0.5),absolute_sigma=True, bounds=bounds)

    A,B,rcmag,mNrc,smrc = mpar
    C,D,rccol,cNrc,scrc = cpar

    def Maggaussian(x,rcmag=rcmag,mNrc=mNrc,smrc=smrc):
        return mNrc/(np.sqrt(2*np.pi*smrc**2))*np.exp(-0.5*(x-rcmag)**2/smrc**2)
    
    def Colgaussian(x,rccol=rccol,cNrc=cNrc,scrc=scrc):
        return cNrc/(np.sqrt(2*np.pi*scrc**2))*np.exp(-0.5*(x-rccol)**2/scrc**2)
    
    def Magexponential(x,A=A,B=B,rcmag=rcmag):
        return A*np.exp(B*(x-rcmag))
    
    def Colexponential(x,C=C,D=D,rccol=rccol):
        return C*np.exp(D*(x-rccol))
    
    def Magmodel(x,A=A,B=B,rcmag=rcmag,mNrc=mNrc,smrc=smrc): 
        return A*np.exp(B*(x-rcmag)) + mNrc/(np.sqrt(2*np.pi*smrc**2))*np.exp(-0.5*(x-rcmag)**2/smrc**2)

    mg1area, mg1error = integrate.quad(Maggaussian, rcmag - smrc, rcmag + smrc)
    mg2area, mg2error = integrate.quad(Maggaussian, rcmag - 3*smrc, rcmag + 3*smrc)
    mearea, meerror = integrate.quad(Magexponential, rcmag - smrc, rcmag + smrc)
    mfarea, mferror = integrate.quad(Magmodel, rcmag - 3*smrc, rcmag + 3*smrc)
    mag_distinctness = mg1area/mearea
    RC_fraction = mg2area/mfarea
    cgarea, cgerror = integrate.quad(Colgaussian, rccol - scrc, rccol + scrc)
    cearea, ceerror = integrate.quad(Colexponential, rccol - scrc, rccol + scrc)
    col_distinctness = cgarea/cearea
    
    df2 = ( # make final cuts using mean and sigma found from fitting
    df.loc[(df['bp_rp']> rccol - scrc) & (df['bp_rp']< rccol + scrc) & 
    (df['phot_g_mean_mag']< rcmag + smrc) & (df['phot_g_mean_mag']>rcmag - smrc)]
    )

    RC_width = (df2["phot_g_mean_mag"].max() - df2["phot_g_mean_mag"].min())/2

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
              "ccov": ccov,
              "mag": rcmag,
              "mag sigma": smrc,
              "Number at mag mean": mNrc,
              "color": rccol,
              "color sigma": scrc,
              "Number at color mean": cNrc,
              "Magnitude Distinctness": mag_distinctness,
              "Color Distinctness": col_distinctness,
              "Red Clump Fraction": RC_fraction,
              "Red Clump Width": RC_width}
    return output

def rcpmodel(x,rcp,pNrc,sprc): # function for fitting parallax w/o zero point
    return pNrc/(np.sqrt(2*np.pi*sprc**2))*np.exp(-0.5*(x-rcp)**2/sprc**2)

def zeropoint(redclump):
    df = redclump["fit cut"]
    G = df["phot_g_mean_mag"]
    C = df["bp_rp"]
    P = df["parallax"]
    Gref = 20.0
    Cref = 0.8
    H0 = -0.0118
    EH0 = 0.003
    H1 = -0.0131
    EH1 = 0.007
    H2 = -0.0128
    EH2 = 0.0089
    H3 = -0.0124
    EH3 = 0.0123
    H4 = -.0131
    EH4 = 0.0201

    zeropoint = pd.DataFrame(index=df.index)

    corr = np.zeros(len(df))

    mask1 = G >= 20.0
    corr[mask1] = H4 - .016*(G[mask1]-Gref) - 0.0035*(C[mask1]-Cref)

    mask2 = (G < 20.0) & (G >= 19.9)
    corr[mask2] = H4 - 0.0035*(C[mask2]-Cref)

    mask3 = (G < 19.9) & (G >= 17)
    corr[mask3] = H4 + .006*(G[mask3]-19.9) - 0.0035*(C[mask3]-Cref)

    mask4 = (G < 17) & (G >= 16.45)
    corr[mask4] = H4 + .006*(G[mask4]-19.9)

    mask5 = (G < 16.45) & (G >= 13.218)
    corr[mask5] = H4 + .00178*(G[mask5]-13.265) - 0.026372

    mask6 = (G < 13.218) & (G >= 12.761)
    corr[mask6] = H4 - .042*(G[mask6]-12.755) - 0.007823

    zeropoint["Parallax"] = P - corr
    zeropoint["Correction"] = corr
    zeropoint["Error"] = np.sqrt(EH4**2 + 0.001**2 + 0.0027**2)

    return zeropoint

def parallax(redclump,zeropoint): # find peak parallax along sightline from fitting distribution of parallax before 
                                    # and after applying zero point
    fitcut = redclump["fit cut"]
    zeropoint = zeropoint
    df = {"parallax": zeropoint["Parallax"], "parallax error": fitcut["parallax_error"], "zp error": zeropoint["Error"],
             "zp correction": zeropoint["Correction"], "bp_rp": fitcut["bp_rp"], "phot_g_mean_mag": fitcut["phot_g_mean_mag"]}
    df = pd.DataFrame(data=df)
    clipped = sigma_clip(np.array(df["parallax"]), sigma=3, maxiters=5)
    df = df[~clipped.mask]

    pbin_width, pbin_edges = knuth_bin_width(df["parallax"], return_bins=True)

    phist,pbe = np.histogram(df["parallax"], bins=pbin_edges)
    pbc = 0.5*(pbe[0:-1]+pbe[1:])
    mask = phist > 0.25*np.max(phist)
    phist = phist[mask]
    pbc = pbc[mask]
    pguess = [np.mean(df["parallax"]),np.max(phist)/np.sqrt(2*np.pi*np.std(df["parallax"])**2),np.std(df["parallax"])]
    ppar,pcov = spopt.curve_fit(rcpmodel,pbc,phist,p0=pguess,sigma=np.sqrt(phist+0.5),absolute_sigma=True)

    rcp,pNrc,sprc = ppar

    fit = (df.loc[(df["parallax"]> rcp - sprc) & (df["parallax"]< rcp + sprc)])

    output = {"phist": phist,
              "pbc": pbc,
              "pbe": pbe,
              "pguess": pguess,
              "ppar": ppar,
              "pcov": pcov,
              "dataframe": fit,
              "pre-fit dataframe": df,
              "parallax peak": rcp,
              "parallax sigma": sprc,
              "Number at parallax mean": pNrc,
              "mean error": np.sqrt(np.diag(pcov)[0])}
    return output

def doubleclumpmodel(x,A,B,rcmag1,mNrc1,smrc1,rcmag2,mNrc2,smrc2): # function for fitting magnitude according to initial cut
    return A*np.exp(B*(x-rcmag1)) + \
           mNrc1/(np.sqrt(2*np.pi*smrc1**2))*np.exp(-0.5*(x-rcmag1)**2/smrc1**2) + \
           mNrc2/(np.sqrt(2*np.pi*smrc2**2))*np.exp(-0.5*(x-rcmag2)**2/smrc2**2)

def doubleclump_finder_v1(df,redclump):
    xrc,yrc,sxrc,syrc = findrc(df)

    df = extractrc_v1(df,xrc,yrc,sxrc,syrc)

    mbin_width, mbin_edges = knuth_bin_width(df["phot_g_mean_mag"], return_bins=True)

    mhist,mbe = np.histogram(df['phot_g_mean_mag'], bins=mbin_edges)
    mbc = 0.5*(mbe[0:-1]+mbe[1:])
    mguess = [(mhist[0]+mhist[-1])/2,0.5,-0.1+(np.max(mbc)+np.min(mbc))/2,(mhist[0]+mhist[-1])/2,0.3,
              0.1+(np.max(mbc)+np.min(mbc))/2,(mhist[0]+mhist[-1])/2,0.3]
    bounds = ([-np.inf,-np.inf,14,0,0,16,0,0],[np.inf,np.inf,19,1000,1.5,19,1000,1.5])
    mpar,mcov = spopt.curve_fit(doubleclumpmodel,mbc,mhist,p0=mguess,sigma=np.sqrt(mhist+0.5), bounds=bounds, absolute_sigma=True)

    A,B,rcmag1,mNrc1,smrc1,rcmag2,mNrc2,smrc2 = mpar
    
    df1 = ( # make final cuts using mean and sigma found from fitting
    df.loc[(df['bp_rp']> redclump["color"] - redclump["color sigma"]) & (df['bp_rp']< redclump["color"] + redclump["color sigma"]) & 
    (df['phot_g_mean_mag']< rcmag1 + smrc1) & (df['phot_g_mean_mag']>rcmag1 - smrc1)]
    )

    df2 = ( # make final cuts using mean and sigma found from fitting
    df.loc[(df['bp_rp']> redclump["color"] - redclump["color sigma"]) & (df['bp_rp']< redclump["color"] + redclump["color sigma"]) & 
    (df['phot_g_mean_mag']< rcmag2 + smrc2) & (df['phot_g_mean_mag']>rcmag2 - smrc2)]
    )

    def Mag1gaussian(x,rcmag=rcmag1,mNrc=mNrc1,smrc=smrc1):
        return mNrc/(np.sqrt(2*np.pi*smrc**2))*np.exp(-0.5*(x-rcmag)**2/smrc**2)
    
    def Mag2gaussian(x,rcmag=rcmag2,mNrc=mNrc2,smrc=smrc2):
        return mNrc/(np.sqrt(2*np.pi*smrc**2))*np.exp(-0.5*(x-rcmag)**2/smrc**2)
    
    def Magexponential(x,A=A,B=B,rcmag=rcmag1):
        return A*np.exp(B*(x-rcmag))

    mg1area, mg1error = integrate.quad(Mag1gaussian, rcmag1 - smrc1, rcmag1 + smrc1)
    mg2area, mg2error = integrate.quad(Mag2gaussian, rcmag2 - smrc2, rcmag2 + smrc2)
    mearea, meerror = integrate.quad(Magexponential, rcmag1 - smrc1, rcmag1 + smrc1)
    mag1_distinctness = mg1area/mearea
    mag2_distinctness = mg2area/mearea

    RC1_width = rcmag1+smrc1
    RC2_width = rcmag2+smrc2
    peak_diff = np.abs(rcmag1-rcmag2)

    output = {"Params": [xrc,yrc,sxrc,syrc], #input parameters for plotting into a directory
              "initial cut": df,
              "RC 1 fit cut": df1,
              "RC 2 fit cut": df2,
              "mhist": mhist,
              "mbc": mbc,
              "mbe": mbe,
              "mguess": mguess,
              "mpar": mpar,
              "mcov": mcov,
              "mag1": rcmag1,
              "mag1 sigma": smrc1,
              "Number at mag1 mean": mNrc1,
              "Red Clump 1 Width": RC1_width,
              "mag2": rcmag2,
              "mag2 sigma": smrc2,
              "Number at mag2 mean": mNrc2,
              "Red Clump 2 Width": RC2_width,
              "peak difference": peak_diff,
              "clump 1 distinctness": mag1_distinctness,
              "clump 2 distinctness": mag2_distinctness}
    return output

def doubleclump_finder_v2(df,redclump):
    xrc,yrc,sxrc,syrc = findrc(df)

    df = extractrc_v2(df,xrc,yrc,sxrc,syrc)

    mbin_width, mbin_edges = knuth_bin_width(df["phot_g_mean_mag"], return_bins=True)

    mhist,mbe = np.histogram(df['phot_g_mean_mag'], bins=mbin_edges)
    mbc = 0.5*(mbe[0:-1]+mbe[1:])
    mguess = [(mhist[0]+mhist[-1])/2,0.5,-0.1+(np.max(mbc)+np.min(mbc))/2,(mhist[0]+mhist[-1])/2,0.3,
              0.1+(np.max(mbc)+np.min(mbc))/2,(mhist[0]+mhist[-1])/2,0.3]
    bounds = ([-np.inf,-np.inf,14,0,0,16,0,0],[np.inf,np.inf,19,1000,1.5,19,1000,1.5])
    mpar,mcov = spopt.curve_fit(doubleclumpmodel,mbc,mhist,p0=mguess,sigma=np.sqrt(mhist+0.5), bounds=bounds, absolute_sigma=True)

    A,B,rcmag1,mNrc1,smrc1,rcmag2,mNrc2,smrc2 = mpar
    
    df1 = ( # make final cuts using mean and sigma found from fitting
    df.loc[(df['bp_rp']> redclump["color"] - redclump["color sigma"]) & (df['bp_rp']< redclump["color"] + redclump["color sigma"]) & 
    (df['phot_g_mean_mag']< rcmag1 + smrc1) & (df['phot_g_mean_mag']>rcmag1 - smrc1)]
    )

    df2 = ( # make final cuts using mean and sigma found from fitting
    df.loc[(df['bp_rp']> redclump["color"] - redclump["color sigma"]) & (df['bp_rp']< redclump["color"] + redclump["color sigma"]) & 
    (df['phot_g_mean_mag']< rcmag2 + smrc2) & (df['phot_g_mean_mag']>rcmag2 - smrc2)]
    )

    def Mag1gaussian(x,rcmag=rcmag1,mNrc=mNrc1,smrc=smrc1):
        return mNrc/(np.sqrt(2*np.pi*smrc**2))*np.exp(-0.5*(x-rcmag)**2/smrc**2)
    
    def Mag2gaussian(x,rcmag=rcmag2,mNrc=mNrc2,smrc=smrc2):
        return mNrc/(np.sqrt(2*np.pi*smrc**2))*np.exp(-0.5*(x-rcmag)**2/smrc**2)
    
    def Magexponential(x,A=A,B=B,rcmag=rcmag1):
        return A*np.exp(B*(x-rcmag))

    mg1area, mg1error = integrate.quad(Mag1gaussian, rcmag1 - smrc1, rcmag1 + smrc1)
    mg2area, mg2error = integrate.quad(Mag2gaussian, rcmag2 - smrc2, rcmag2 + smrc2)
    mearea, meerror = integrate.quad(Magexponential, rcmag1 - smrc1, rcmag1 + smrc1)
    mag1_distinctness = mg1area/mearea
    mag2_distinctness = mg2area/mearea

    RC1_width = rcmag1+smrc1
    RC2_width = rcmag2+smrc2
    peak_diff = np.abs(rcmag1-rcmag2)

    output = {"Params": [xrc,yrc,sxrc,syrc], #input parameters for plotting into a directory
              "initial cut": df,
              "RC 1 fit cut": df1,
              "RC 2 fit cut": df2,
              "mhist": mhist,
              "mbc": mbc,
              "mbe": mbe,
              "mguess": mguess,
              "mpar": mpar,
              "mcov": mcov,
              "mag1": rcmag1,
              "mag1 sigma": smrc1,
              "Number at mag1 mean": mNrc1,
              "Red Clump 1 Width": RC1_width,
              "mag2": rcmag2,
              "mag2 sigma": smrc2,
              "Number at mag2 mean": mNrc2,
              "Red Clump 2 Width": RC2_width,
              "peak difference": peak_diff,
              "clump 1 distinctness": mag1_distinctness,
              "clump 2 distinctness": mag2_distinctness}
    return output

def doublepmodel(x,rcp1,pNrc1,sprc1,rcp2,pNrc2,sprc2): # function for fitting parallax w/o zero point
    return pNrc1/(np.sqrt(2*np.pi*sprc1**2))*np.exp(-0.5*(x-rcp1)**2/sprc1**2) + \
            pNrc2/(np.sqrt(2*np.pi*sprc2**2))*np.exp(-0.5*(x-rcp2)**2/sprc2**2)

def doublefinalfit(redclump,zeropoint): # find peak parallax along sightline from fitting distribution of parallax before 
                                    # and after applying zero point
    fitcut = redclump["fit cut"]
    zeropoint = zeropoint
    df = {"parallax": zeropoint["Parallax"], "parallax error": fitcut["parallax_error"], "zp error": zeropoint["Error"],
             "zp correction": zeropoint["Correction"], "bp_rp": fitcut["bp_rp"], "phot_g_mean_mag": fitcut["phot_g_mean_mag"]}
    df = pd.DataFrame(data=df)
    clipped = sigma_clip(np.array(df["parallax"]), sigma=3, maxiters=5)
    df = df[~clipped.mask]

    pbin_width, pbin_edges = knuth_bin_width(df["parallax"], return_bins=True)

    phist,pbe = np.histogram(df["parallax"], bins=pbin_edges)
    pbc = 0.5*(pbe[0:-1]+pbe[1:])
    mask = phist > 0.25*np.max(phist)
    phist = phist[mask]
    pbc = pbc[mask]
    pguess = [np.mean(df["parallax"]),np.max(phist)/np.sqrt(2*np.pi*np.std(df["parallax"])**2),np.std(df["parallax"]),
              np.mean(df["parallax"]),np.max(phist)/np.sqrt(2*np.pi*np.std(df["parallax"])**2),np.std(df["parallax"])]
    bounds = [(0,0,0,0,0,0),(0.2,1000,0.5,0.2,1000,0.5)]
    ppar,pcov = spopt.curve_fit(doublepmodel,pbc,phist,p0=pguess,sigma=np.sqrt(phist+0.5),absolute_sigma=True, bounds=bounds)

    rcp1,pNrc1,sprc1,rcp2,pNrc2,sprc2 = ppar

    fit1 = (df.loc[(df["parallax"]> rcp1 - sprc1) & (df["parallax"]< rcp1 + sprc1)])

    fit2 = (df.loc[(df["parallax"]> rcp2 - sprc2) & (df["parallax"]< rcp2 + sprc2)])

    output = {"phist": phist,
              "pbc": pbc,
              "pbe": pbe,
              "pguess": pguess,
              "ppar": ppar,
              "pcov": pcov,
              "dataframe 1": fit1,
              "dataframe 2": fit2,
              "pre-fit dataframe": df,
              "parallax peak 1": rcp1,
              "parallax sigma 1": sprc1,
              "Number at parallax mean 1": pNrc1,
              "parallax peak 2": rcp2,
              "parallax sigma 2": sprc2,
              "Number at parallax mean 2": pNrc2}
    return output
