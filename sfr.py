"""Make some plots for the asterix paper."""
import os
import glob
import sys
import numpy as np
import scipy.stats
from multiprocessing import Pool
import h5py
import corecon as crc
import matplotlib.pyplot as plt
from bigfile import BigFile

plt.rcParams['axes.linewidth'] = 1.8
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 16
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams['figure.constrained_layout.use'] = True

def find_snapshot(reds, snaptxt="Snapshots.txt"):
    """Find a snapshot at the desired redshift"""
    snaps = np.loadtxt(snaptxt)
    atimes = 1/(reds +1)
    snapshots = np.array([int(snaps[np.argmin(np.abs(snaps[:,1] - atime)),0]) for atime in atimes])
    return snapshots

def get_ssfr(pig):
    """Get the specific SFR (SFR per unit stellar mass) for some different stellar masses."""
    bf = BigFile(pig)
    sfr = bf['FOFGroups/StarFormationRate'][:]
    stellarmasses = bf['FOFGroups/MassByType'][:][:,4]*1e10/bf["Header"].attrs["HubbleParam"]
    massbins = np.array([4e6, 1e7, 1e8, 1e9, 1e10])
    #massbins = np.array([1e7, 1e8, 1e9])
    avgssfr = np.zeros((np.size(massbins),3))
    for i, mm in enumerate(massbins):
        ii = np.where((stellarmasses < mm*1.3)*(stellarmasses > mm / 1.3))
        if np.size(ii) == 0:
            print("no halos for m* %g pig %s" % (mm, pig))
            continue
        avgssfr[i] = np.percentile(sfr[ii]/stellarmasses[ii], [16, 50, 84])
    return avgssfr

def plot_ssfr(reds, outdir):
    """Average SFR over time"""
    snaps = find_snapshot(reds, snaptxt=os.path.join(outdir, "Snapshots.txt"))
    pigs = [os.path.join(outdir, "PIG_%03d") % ss for ss in snaps]
    colors = ["black", "red", "blue", "brown", "green"]
    lss = ["-", "--", ":", "-."]
    labels = [r"$M_* = 4\times 10^6 M_\odot$", r"$M_* = 10^7 M_\odot$", r"$M_* = 10^8 M_\odot$", r"$M_* = 10^{9} M_\odot$", r"$M_* = 10^{10} M_\odot$"]

    #From arxiv.org/abs/1407.6012 table 3.
    zz = np.array([4,5,6])
    #1e9:
    logsfr9 = np.array([0.71, 0.88, 0.92])
    #Monte Carlo sigma
    sigma9 = np.array([0.26, 0.25, 0.43])
    yerr9 = (10**logsfr9/1e9 - 10**(logsfr9 - sigma9)/1e9, 10**(logsfr9 + sigma9)/1e9 - 10**logsfr9/1e9)
    plt.errorbar(zz, 10**logsfr9/1e9, yerr=yerr9, fmt='^',color='black',ms=5, label="Salmon+14 $10^9$")
    #1e10:
    #logsfr10 = np.array([1.35, 1.46, 1.47])
    #sigma10 = np.array([0.27, 0.27, 0.27])
    #yerr10 = (10**logsfr10/1e10 - 10**(logsfr10 - sigma10)/1e10, 10**(logsfr10 + sigma10)/1e10 - 10**logsfr10/1e10)
    #plt.errorbar(zz, 10**logsfr10/1e10, yerr=yerr10, fmt='^',color='black',ms=5, label="Salmon+14 $10^9$")

    #labels = [r"$10^7$", r"$10^8$", r"$10^{9}$"]
    ssfrs = np.array([get_ssfr(pig) for pig in pigs])
    for i in range(np.shape(ssfrs)[1]):
        j = np.where(ssfrs[:,i,1] > 0)
        ls = lss[i % len(lss)]
        if i == 0:
        #if i == 2 or i == 3:
            plt.fill_between(reds[j], ssfrs[j[0],i,0]+1e-13, ssfrs[j[0],i,2], color=colors[i], alpha=0.15)
        plt.plot(reds[j], ssfrs[j[0],i,1], label=labels[i], color=colors[i], ls=ls)
        np.savetxt("data/ssfr-mstar%d.txt" % i, np.vstack((reds[j], ssfrs[j[0],i,:].T)).T, header="# M* = "+labels[i]+" (redshift : sSFR 16, 50, 84 percentiles)")
    plt.xlabel("z")
    plt.ylabel(r"sSFR (yr$^{-1}$)")
    plt.legend(loc="upper left")
    #plt.yscale('log')
    plt.ylim(1e-9,3e-8)
    #plt.tight_layout()
    plt.savefig("ssfr.pdf")

def get_avg_thing(pig, thing="StarFormationRate"):
    """Get the averaged SFR for some different halo masses."""
    bf = BigFile(pig)
    fofmasses = bf['FOFGroups/Mass'][:]*1e10/bf["Header"].attrs["HubbleParam"]
    sfr = bf['FOFGroups/'+thing][:]
    massbins = np.array([2e9, 1e10, 1e11, 1e12, 1e13])
    avgsfr = np.zeros((5,3))
    for i, mm in enumerate(massbins):
        ii = np.where((fofmasses < mm*1.3)*(fofmasses > mm / 1.3))
        if np.size(ii) == 0:
            print("no halos for mass %g pig %s" % (mm, pig))
            continue
        avgsfr[i] = np.percentile(sfr[ii], [16, 50, 84])
    del sfr
    del fofmasses
    return avgsfr

def plot_behroozi_sfr(mass):
    """Plot the SFR over time from Behroozi 2013."""
    sfrdata = np.loadtxt("sfr/sfr_corrected_%d.0.dat" % mass)
    zz = 1/sfrdata[:,0]-1
    ii = np.where((zz >= 3)*(zz <= 12))
    plt.plot(zz[ii], sfrdata[:,1][ii], ls="--", label=r"$10^{%d}$" % mass)
    plt.plot(zz[ii], sfrdata[:,1][ii] + sfrdata[:,2][ii], ls="--")
    plt.plot(zz[ii], sfrdata[:,1][ii] - sfrdata[:,3][ii], ls="--")

def plot_avg_sfr(reds, outdir):
    """Average SFR over time"""
    snaps = find_snapshot(reds, snaptxt=os.path.join(outdir, "Snapshots.txt"))
    pigs = [os.path.join(outdir, "PIG_%03d") % ss for ss in snaps]
    colors = ["black", "red", "blue", "brown", "green"]
    lss = ["-", "--", ":", "-."]
    labels = [r"$2\times 10^{9}$", r"$10^{10}$", r"$10^{11}$", r"$10^{12}$", r"$10^{13}$"]
    sfrs = np.array([get_avg_thing(pig, thing="StarFormationRate") for pig in pigs])
    for i in range(np.shape(sfrs)[1]):
        j = np.where(sfrs[:,i,2] > 0)
        plt.fill_between(reds[j], sfrs[j[0],i,0]+1e-14, sfrs[j[0],i,2], color=colors[i], alpha=0.15)
        j = np.where(sfrs[:,i,1] > 0)
        ls = lss[i % len(lss)]
        plt.plot(reds[j], sfrs[j[0],i,1], label=labels[i], color=colors[i], ls=ls)
        np.savetxt("data/asfr-mdm%d.txt" % i, np.vstack((reds[j], sfrs[j[0],i,:].T)).T, header="# MDM = "+labels[i]+" (redshift : SFR 16, 50, 84 percentiles )")
    #for mm in (11, 12, 13):
        #plot_behroozi_sfr(mm)
    plt.xlabel("z")
    plt.ylabel(r"SFR ($M_\odot$ yr$^{-1}$)")
    plt.legend(loc="upper right")
    plt.yscale('log')
    plt.ylim(ymin=1e-5)
    #plt.tight_layout()
    plt.savefig("avg_sfr.pdf")

def get_avg_sfr_reion(pig, zreion, nmesh):
    """Get the averaged SFR for some different halo masses."""
    bf = BigFile(pig)
    redshift = 1/bf["Header"].attrs["Time"]-1
    fofmasses = bf['FOFGroups/Mass'][:]*1e10/bf["Header"].attrs["HubbleParam"]
    box = bf['Header'].attrs["BoxSize"]
    fofpos = bf['FOFGroups/MassCenterPosition'][:]
    sfr = bf['FOFGroups/StarFormationRate'][:]
    massbins = np.array([2e9, 5e9, 1e10, 5e10])
    avgsfr_reion = np.zeros((len(massbins),3))
    avgsfr_no_reion = np.zeros((len(massbins),3))
    for i, mm in enumerate(massbins):
        ii = np.where((fofmasses < mm*1.3)*(fofmasses > mm / 1.3))
        if np.size(ii) == 0:
            print("no halos for mass %g pig %s" % (mm, pig))
            continue
        zpos = fofpos[ii] / box * nmesh
        fofzreion = np.array([zreion[int(zp[0]), int(zp[1]), int(zp[2])] for zp in zpos])
        jj = np.where(fofzreion > redshift)
        if np.size(jj) > 0:
            avgsfr_reion[i] = np.percentile(sfr[ii][jj], [16, 50, 84])
        jj = np.where(fofzreion < redshift)
        if np.size(jj) > 0:
            avgsfr_no_reion[i] = np.percentile(sfr[ii][jj], [16, 50, 84])
    del sfr
    del fofpos
    del fofmasses
    return avgsfr_reion, avgsfr_no_reion

def plot_avg_sfr_reion(reds, outdir):
    """Average SFR over time"""
    snaps = find_snapshot(reds, snaptxt=os.path.join(outdir, "Snapshots.txt"))
    pigs = [os.path.join(outdir, "PIG_%03d") % ss for ss in snaps]
    colors_reion = ["darkred", "navy", "brown", "black"]
    colors_no_reion = ["red", "blue", "orange", "grey"]
    labels = [r"$2\times 10^{9}$", r"$5\times 10^{9}$", r"$10^{10}$", r"$5\times 10^{10}$"]
    lss_ratio = ["-", "--", ":", "-."]
    uvf = BigFile(os.path.join(outdir, "../UVFluctuationFile"))
    zreion = uvf["Zreion_Table"][:]
    nmesh = uvf["Zreion_Table"].attrs["Nmesh"][0]
    zreion = zreion.reshape((nmesh, nmesh, nmesh))
    uvf.close()
    sfrs_reion, sfrs_no_reion = zip(*[get_avg_sfr_reion(pig, zreion, nmesh) for pig in pigs])
    sfrs_reion = np.array(sfrs_reion)
    sfrs_no_reion = np.array(sfrs_no_reion)
    fig = plt.figure()
    gs = fig.add_gridspec(2,1, hspace=0, height_ratios=[0.7,0.3])
    lower = fig.add_subplot(gs[1,0])
    lower.set_xlim(xmin=5.5, xmax=reds[0])
    lower.set_xlabel("z")
    lower.set_ylim(ymin=0, ymax=0.55)
    lower.set_ylabel("$\Delta$ SFR", fontsize=14)
    upper = fig.add_subplot(gs[0,0], sharex=lower)
    plt.setp(upper.get_xticklabels(), visible=False)
    upper.set_ylabel(r"SFR ($M_\odot$ yr$^{-1}$)",fontsize=14)
    upper.set_yscale('log')
    hatches = [None, '/', None, 'X']
    upper.set_ylim(ymin=1e-6, ymax=1)
    for i in range(np.shape(sfrs_reion)[1]):
        j = np.where(sfrs_reion[:,i,2] > 0)
        if hatches[i] is not None:
            upper.fill_between(reds[j], sfrs_reion[j[0],i,0]+1e-14, sfrs_reion[j[0],i,2], facecolor="none", edgecolor=colors_reion[i], alpha=0.15, hatch=hatches[i])
        else:
            upper.fill_between(reds[j], sfrs_reion[j[0],i,0]+1e-14, sfrs_reion[j[0],i,2], color=colors_reion[i], alpha=0.15)
        j = np.where(sfrs_reion[:,i,1] > 0)
        upper.plot(reds[j], sfrs_reion[j[0],i,1], color=colors_reion[i], ls="-", label="HII: "+labels[i])
        #j = np.where(sfrs_no_reion[:,i,2] > 0)
        #plt.fill_between(reds[j], sfrs_no_reion[j[0],i,0]+1e-14, sfrs_no_reion[j[0],i,2], color=colors_no_reion[i], alpha=0.15)
        j = np.where(sfrs_no_reion[:,i,1] > 0)
        upper.plot(reds[j], sfrs_no_reion[j[0],i,1], color=colors_no_reion[i], ls="--", label="HI: "+labels[i])
        #np.savetxt("asfr-mdm%d.txt" % i, np.vstack((reds[j], sfrs[j[0],i,:].T)).T, header="# MDM = "+labels[i]+" (redshift : SFR 16, 50, 84 percentiles )")
        lower.plot(reds[j], sfrs_no_reion[j[0],i,1]/sfrs_reion[j[0],i,1]-1, color=colors_reion[i], ls=lss_ratio[i], label=labels[i])
    lower.legend(loc="upper right", ncol=2, fontsize=12)
    upper.legend(loc="lower right", ncol=2, fontsize=12)
    #plt.tight_layout()
    plt.savefig("avg_sfr_reion.pdf")

def get_avg_sfr_heii_reion(pig):
    """Get the averaged SFR for some different halo masses."""
    bf = BigFile(pig)
    fofmasses = bf['FOFGroups/Mass'][:]*1e10/bf["Header"].attrs["HubbleParam"]
    sfr = bf['FOFGroups/StarFormationRate'][:]
    heiifrac = bf['FOFGroups/MassHeIonized'][:]/(1e-6+bf['FOFGroups/MassByType'][:][:,0])
    massbins = np.array([5e9, 1e10, 5e10])
    avgsfr_reion = np.zeros((len(massbins),3))
    avgsfr_no_reion = np.zeros((len(massbins),3))
    for i, mm in enumerate(massbins):
        ii = np.where((fofmasses < mm*1.3)*(fofmasses > mm / 1.3))
        if np.size(ii) == 0:
            print("no halos for mass %g pig %s" % (mm, pig))
            continue
        print("mass %.1f : %d" % (mm, np.size(ii)))
        jj = np.where(heiifrac[ii] > 0.6)
        if np.size(jj) > 0:
            avgsfr_reion[i] = np.percentile(sfr[ii][jj], [16, 50, 84])
        jj = np.where(heiifrac[ii] < 0.4)
        if np.size(jj) > 0:
            avgsfr_no_reion[i] = np.percentile(sfr[ii][jj], [16, 50, 84])
    del sfr
    del fofmasses
    return avgsfr_reion, avgsfr_no_reion

def plot_avg_sfr_heii_reion(reds, outdir):
    """Average SFR over time"""
    snaps = find_snapshot(reds, snaptxt=os.path.join(outdir, "Snapshots.txt"))
    pigs = [os.path.join(outdir, "PIG_%03d") % ss for ss in snaps]
    colors_reion = ["darkred", "navy", "brown", "black"]
    colors_no_reion = ["red", "blue", "orange", "grey"]
    labels = [r"$5\times 10^{9}$", r"$10^{10}$", r"$5\times 10^{10}$"]
    #labels = [r"$2\times 10^{9}$", r"$5\times 10^{9}$", r"$10^{10}$", r"$5\times 10^{10}$"]
    sfrs_reion, sfrs_no_reion = zip(*[get_avg_sfr_heii_reion(pig) for pig in pigs])
    sfrs_reion = np.array(sfrs_reion)
    sfrs_no_reion = np.array(sfrs_no_reion)
    for i in range(np.shape(sfrs_reion)[1]):
        j = np.where(sfrs_reion[:,i,2] > 0)
        plt.fill_between(reds[j], sfrs_reion[j[0],i,0]+1e-14, sfrs_reion[j[0],i,2], color=colors_reion[i], alpha=0.2)
        j = np.where(sfrs_reion[:,i,1] > 0)
        plt.plot(reds[j], sfrs_reion[j[0],i,1], label="HeIII: "+labels[i], color=colors_reion[i], ls="-")
        j = np.where(sfrs_no_reion[:,i,2] > 0)
        plt.fill_between(reds[j], sfrs_no_reion[j[0],i,0]+1e-14, sfrs_no_reion[j[0],i,2], color=colors_no_reion[i], alpha=0.2)
        j = np.where(sfrs_no_reion[:,i,1] > 0)
        plt.plot(reds[j], sfrs_no_reion[j[0],i,1], label="HeII: "+labels[i], color=colors_no_reion[i], ls="--")
        #np.savetxt("asfr-mdm%d.txt" % i, np.vstack((reds[j], sfrs[j[0],i,:].T)).T, header="# MDM = "+labels[i]+" (redshift : SFR 16, 50, 84 percentiles )")
    plt.xlabel("z")
    plt.ylabel(r"SFR ($M_\odot$ yr$^{-1}$)")
    plt.legend(loc="upper left")
    #plt.yscale('log')
    #plt.ylim(ymin=1e-6, ymax=2)
    plt.xlim(xmin=3, xmax=4)
    #plt.tight_layout()
    plt.savefig("avg_sfr_heii_reion.pdf")

def load_bigfile_data(pig, metal=False, star=True, allbar=False):
    """Load the stellar mass data from a bigfile."""
    bf = BigFile(pig)
    hh = bf["Header"].attrs["HubbleParam"]
    fofmasses = bf['FOFGroups/Mass'][:]*1e10/hh
    metals = None
    if star:
        starmass = bf['FOFGroups/MassByType'][:][:,4]*1e10/hh
        if metal:
            metals = bf["FOFGroups/StellarMetalElemMass"][:]*1e10/hh
    else:
        starmass = bf['FOFGroups/MassByType'][:][:,0]*1e10/hh
        if allbar:
            starmass += bf['FOFGroups/MassByType'][:][:,4]*1e10/hh
            starmass += bf['FOFGroups/MassByType'][:][:,5]*1e10/hh
        if metal:
            metals = bf["FOFGroups/GasMetalElemMass"][:]*1e10/hh
    omega0 = bf["Header"].attrs["Omega0"]
    omegab = bf["Header"].attrs["OmegaBaryon"]
    zz = 1/bf["Header"].attrs["Time"]-1
    return (omega0, omegab, zz, metals, starmass, fofmasses)

def load_h5py_data(fofpattern, metal=False, star=True):
    """Load the stellar mass data from a bigfile."""
    if metal:
        raise NotImplementedError("Metal loading not supported")
    pigs = glob.glob(fofpattern)
    bf = h5py.File(pigs[0])
    hh = bf["Header"].attrs["HubbleParam"]
    omega0 = bf["Header"].attrs["Omega0"]
    xb = 0.186667
    omegab = omega0 * (xb / (xb + 1))
    zz = 1/bf["Header"].attrs["Time"]-1
    metals = None
    fofmasses = bf['Group']['GroupMass'][:]*1e10/hh
    if star:
        starmass = bf['Group']['GroupMassType'][:,4]*1e10/hh
    else:
        starmass = bf['Group']['GroupMassType'][:,0]*1e10/hh
    for pig in pigs[1:]:
        bf = h5py.File(pig)
        fm = bf['Group']['GroupMass'][:]*1e10/hh
        if star:
            sm = bf['Group']['GroupMassType'][:,4]*1e10/hh
        else:
            sm = bf['Group']['GroupMassType'][:,0]*1e10/hh
        fofmasses = np.concatenate([fofmasses, fm])
        starmass = np.concatenate([starmass, sm])
    return (omega0, omegab, zz, metals, starmass, fofmasses)

def plot_thesan_smhm(datafile, red=None, color="black", ls="-"):
    """Plot the stellar-mass/halo-mass relation from THESAN"""
    data = np.loadtxt(datafile)
    #log(Halo Mass [Msun])  Median-log(Stellar mass [Msun]) 10 percentile-log(Stellar mass [Msun])  90 percentile-log(Stellar mass [Msun])
    ombyob = 0.3089/0.0486
    #plt.fill_between(10**data[:,0], 10**data[:,2]/10**data[:,0]*ombyob, 10**data[:,3]/10**data[:,0]*ombyob, alpha=0.15)
    plt.plot(10**data[:,0], 10**data[:,1]/10**data[:,0]*ombyob, color=color, ls=ls, label="TSN z=%d" % red)

def plot_smhm(pig, color=None, ls=None, star=True, metal=False, scatter=True, hdf5=False, allbar=False):
    """Plot the stellar/gas mass to halo mass. If star is True, stars, else gas."""
    if hdf5:
        (omega0, omegab, zz, metals, starmass, fofmasses) = load_h5py_data(pig, metal=metal, star=star)
    else:
        (omega0, omegab, zz, metals, starmass, fofmasses) = load_bigfile_data(pig, metal=metal, star=star, allbar=allbar)
    if metal:
        ii = np.where(starmass > 0)
        oxy = metals[:,4]
        smhm = 12 + np.log10(1e-10 + oxy[ii]/(starmass[ii]*0.76)/16.)
        #smhm = metals[ii]/starmass[ii]
        fofmasses = starmass[ii]
        del oxy
        del metals
    else:
        smhm = starmass/fofmasses
    del starmass
    if hdf5:
        label= "TNG z=%d" % zz
    else:
        label = "z=%.0f" % zz
    if metal:
        massbins = np.logspace(7, 12, 50)
    else:
        massbins = np.logspace(9, 14, 50)
    smhm_bin = np.zeros(np.size(massbins)-1)
    smhm_lower = np.zeros(np.size(massbins)-1)
    smhm_upper = np.zeros(np.size(massbins)-1)
    if metal:
        factor = 1
        #factor = 1/0.0142
    else:
        factor = omega0 / omegab
    for i in np.arange(np.size(massbins)-1):
        ii = np.where((fofmasses > massbins[i])*(fofmasses < massbins[i+1]))
        if np.size(ii) < 10:
            continue
        smhm_bin[i] = np.median(smhm[ii]) * factor
        smhm_lower[i] = np.percentile(smhm[ii], 16) * factor
        smhm_upper[i] = np.percentile(smhm[ii], 84) * factor
    masses = np.exp((np.log(massbins[1:]) + np.log(massbins[:-1]))/2.)
    ii = np.where(smhm_bin > 0)
    plt.semilogx(masses[ii], smhm_bin[ii], color=color, label=label, ls=ls)
    if scatter:
        plt.fill_between(masses[ii], smhm_lower[ii], smhm_upper[ii], color=color, alpha=0.3)
    fname = "hm-z-%.1f.txt" % zz
    if star:
        if metal:
            fname = "data/starmetal-" + fname
        else:
            fname = "data/starm-" + fname
    else:
        if metal:
            fname = "data/gasmetal-" + fname
        else:
            fname = "data/gasm-" + fname
    np.savetxt(fname, np.vstack((masses[ii], smhm_lower[ii], smhm_bin[ii], smhm_upper[ii])).T, header="# "+label+" (mass : SMHM (16, 50, 84))")

def plot_smhms(reds, outdir, star=True, metal=False, hdf5=False, allbar=False):
    """Plot several SMHM over time."""
    snaps = find_snapshot(reds, snaptxt=os.path.join(outdir, "Snapshots.txt"))
    pigs = [os.path.join(outdir, "PIG_%03d") % ss for ss in snaps]
    colors = ["red", "blue", "brown", "black", "grey"]
    lss = [":", "--", "--", "-"]
    if star and metal:
        #FIt from table 3 of https://arxiv.org/pdf/2009.07292.pdf
        m10 = np.logspace(9, 10.5, 50)
        #zobs = 0.29 * np.log10(m10/1e10) + 8.41
        zupper = 0.31 * np.log10(m10/1e10) + 8.44
        zlower = 0.27 * np.log10(m10/1e10) + 8.38
        ii = np.where(m10 < 1e10)
        zlower[ii] = 0.31 * np.log10(m10[ii]/1e10) + 8.38
        zupper[ii] = 0.27 * np.log10(m10[ii]/1e10) + 8.44
        #plt.plot(m10, zobs, ls="-",color="blue")
        plt.fill_between(m10, zupper, zlower, color="green", alpha=0.2, label="Sanders+21")
    for ii in np.arange(len(reds)):
        plot_smhm(pigs[ii], color=colors[ii], ls=lss[ii % len(lss)], star=star, metal=metal, scatter=(ii ==len(reds)-1), hdf5=False, allbar=allbar)
    if hdf5:
        snaps = [13, 25]
        pigs = ["/work/04808/tg841079/frontera/tng_fof/tng_fof_%03d/fof_subhalo_tab_%03d.*.hdf5" % (ss, ss) for ss in snaps]
        color = ["black", "navy"]
        for i, pp in enumerate(pigs):
            plot_smhm(pp, color=color[i], ls=":", star=star, metal=False, scatter=False, hdf5=True)
    plt.xlabel(r"$M_\mathrm{h} (M_\odot)$")
    #plt.tight_layout()
    if star:
        if metal:
            plt.legend(loc="upper left")
            #plt.ylabel(r"Z$_*$ (Z$_\odot$)")
            plt.ylabel(r"$12 + \log_{10}(O/H)$")
            plt.xlabel(r"$M_\mathrm{*} (M_\odot)$")
            #plt.ylim(ymin=3e-4)
            #plt.yscale('log')
            plt.savefig("starmetal_oh.pdf")
        else:
            plt.legend(loc="lower right")
            plt.yscale('log')
            plt.ylabel(r"$M_* / M_\mathrm{h} (\Omega_M / \Omega_b)$")
            plt.ylim(ymin=5e-4)
            plt.savefig("smhms.pdf")
    else:
        if metal:
            plt.legend(loc="lower center")
            #plt.ylim(ymax=1.1)
            #plt.yscale('log')
            plt.ylabel(r"$12 + \log_{10}(O/H)$")
            #plt.ylabel(r"Z$_\mathrm{g}$ (Z$_\odot$)")
            plt.xlabel(r"$M_\mathrm{*} (M_\odot)$")
            plt.savefig("gasmetal_oh.pdf")
        else:
            plt.legend(loc="lower center")
            plt.ylim(0.5, 1)
            if allbar:
                plt.ylabel(r"$M_b / M_\mathrm{h} (\Omega_M / \Omega_b)$")
                plt.savefig("gmhms-all.pdf")
            else:
                plt.ylabel(r"$M_g / M_\mathrm{h} (\Omega_M / \Omega_b)$")
                plt.savefig("gmhms.pdf")

def plot_smhm_he_reion(pig, color=None, star=True, metal=False):
    """Plot the stellar/gas mass to halo mass. If star is True, stars, else gas."""
    bf = BigFile(pig)
    hh = bf["Header"].attrs["HubbleParam"]
    fofmasses = bf['FOFGroups/Mass'][:]*1e10/hh
    if star:
        if metal:
            metals = bf["FOFGroups/StellarMetalMass"][:]
            starmass = bf['FOFGroups/MassByType'][:][:,4]
        else:
            stellarmasses = bf['FOFGroups/MassByType'][:][:,4]*1e10/hh
    else:
        if metal:
            metals = bf["FOFGroups/GasMetalMass"][:]
            starmass = bf['FOFGroups/MassByType'][:][:,0]
        else:
            stellarmasses = bf['FOFGroups/MassByType'][:][:,0]*1e10/hh
    omega0 = bf["Header"].attrs["Omega0"]
    #From the mass ratios
    omegab = bf["Header"].attrs["OmegaBaryon"]
    zz = 1/bf["Header"].attrs["Time"]-1
    heiifrac = bf['FOFGroups/MassHeIonized'][:]/(1e-6+bf['FOFGroups/MassByType'][:][:,0])
    if metal:
        ii = np.where(starmass > 0)
        smhm = metals[ii]/starmass[ii]
        fofmasses = fofmasses[ii]
        heiifrac = heiifrac[ii]
        del starmass
        del metals
    else:
        smhm = stellarmasses/fofmasses
        del stellarmasses
    label = "z=%d" % zz
    massbins = np.logspace(9, 14, 50)
    smhm_bin = np.zeros(np.size(massbins)-1)
    smhm_lower = np.zeros(np.size(massbins)-1)
    smhm_upper = np.zeros(np.size(massbins)-1)
    if metal:
        factor = 1/0.0122
    else:
        factor = omega0 / omegab
    for i in np.arange(np.size(massbins)-1):
        ii = np.where((fofmasses > massbins[i])*(fofmasses < massbins[i+1]))
        if np.size(ii) < 10:
            continue
        jj = np.where(heiifrac[ii] < 0.4)
        smhm_bin[i] = np.median(smhm[ii][jj]) * factor
        smhm_lower[i] = np.percentile(smhm[ii][jj], 16) * factor
        smhm_upper[i] = np.percentile(smhm[ii][jj], 84) * factor
    masses = np.exp((np.log(massbins[1:]) + np.log(massbins[:-1]))/2.)
    ii = np.where(smhm_bin > 0)
    plt.semilogx(masses[ii], smhm_bin[ii], color=color, label="HeII: "+label, ls="--")
    for i in np.arange(np.size(massbins)-1):
        ii = np.where((fofmasses > massbins[i])*(fofmasses < massbins[i+1]))
        if np.size(ii) < 10:
            continue
        jj = np.where(heiifrac[ii] > 0.6)
        smhm_bin[i] = np.median(smhm[ii][jj]) * factor
        smhm_lower[i] = np.percentile(smhm[ii][jj], 16) * factor
        smhm_upper[i] = np.percentile(smhm[ii][jj], 84) * factor
    masses = np.exp((np.log(massbins[1:]) + np.log(massbins[:-1]))/2.)
    ii = np.where(smhm_bin > 0)
    plt.semilogx(masses[ii], smhm_bin[ii], color=color, label="HeIII: "+label, ls="-")
    #if scatter:
        #plt.fill_between(masses[ii], smhm_lower[ii], smhm_upper[ii], color=color, alpha=0.3)
    fname = "hm-z-%.1f.txt" % zz
    if star:
        if metal:
            fname = "data/starmetal-" + fname
        else:
            fname = "data/starm-" + fname
    else:
        if metal:
            fname = "data/gasmetal-" + fname
        else:
            fname = "data/gasm-" + fname
    #np.savetxt(fname, np.vstack((masses[ii], smhm_lower[ii], smhm_bin[ii], smhm_upper[ii])).T, header="# "+label+" (mass : SMHM (16, 50, 84))")

def plot_smhms_he_reion(reds, outdir, star=True, metal=False):
    """Plot several SMHM over time."""
    snaps = find_snapshot(reds, snaptxt=os.path.join(outdir, "Snapshots.txt"))
    pigs = [os.path.join(outdir, "PIG_%03d") % ss for ss in snaps]
    colors = ["black", "red", "blue", "brown", "grey", "orange"]
    for ii in np.arange(len(reds)):
        plot_smhm_he_reion(pigs[ii], color=colors[ii], star=star, metal=metal)
    plt.xlabel(r"$M_\mathrm{h} (M_\odot)$")
    #plt.tight_layout()
    if star:
        if metal:
            plt.legend(loc="upper left")
            plt.ylabel(r"Z$_*$ (Z$_\odot$)")
            #plt.ylim(ymin=3e-4)
            plt.yscale('log')
            plt.savefig("starmetal_he_reion.pdf")
        else:
            plt.legend(loc="upper left")
            plt.yscale('log')
            plt.ylabel(r"$M_* / M_\mathrm{h} (\Omega_M / \Omega_b)$")
            plt.ylim(ymin=5e-4)
            plt.savefig("smhms_he_reion.pdf")
    else:
        if metal:
            plt.legend(loc="lower center")
            #plt.ylim(ymax=1.1)
            plt.yscale('log')
            plt.ylabel(r"Z$_\mathrm{g}$ (Z$_\odot$)")
            plt.savefig("gasmetal_he_reion.pdf")
        else:
            plt.ylabel(r"$M_g / M_\mathrm{h} (\Omega_M / \Omega_b)$")
            plt.legend(loc="lower center")
            plt.ylim(0.5, 1)
            plt.savefig("gmhms_he_reion.pdf")

def plot_smhm_reion(pig, color=None, ls=None, zreion=None, nmesh=0, star=True, metal=False, scatter=True):
    """Plot the stellar/gas mass to halo mass. If star is True, stars, else gas."""
    bf = BigFile(pig)
    hh = bf["Header"].attrs["HubbleParam"]
    fofmasses = bf['FOFGroups/Mass'][:]*1e10/hh
    ##
    fofpos = bf['FOFGroups/MassCenterPosition'][:]
    ##
    if star:
        if metal:
            metals = bf["FOFGroups/StellarMetalMass"][:]
            starmass = bf['FOFGroups/MassByType'][:][:,4]
        else:
            stellarmasses = bf['FOFGroups/MassByType'][:][:,4]*1e10/hh
    else:
        if metal:
            metals = bf["FOFGroups/GasMetalMass"][:]
            starmass = bf['FOFGroups/MassByType'][:][:,0]
        else:
            stellarmasses = bf['FOFGroups/MassByType'][:][:,0]*1e10/hh
    omega0 = bf["Header"].attrs["Omega0"]
    omegab = bf["Header"].attrs["OmegaBaryon"]
    zz = 1/bf["Header"].attrs["Time"]-1
    box = bf["Header"].attrs["BoxSize"]
    if metal:
        ii = np.where(starmass > 0)
        smhm = metals[ii]/starmass[ii]
        fofmasses = fofmasses[ii]
        fofpos = fofpos[ii]
        del starmass
        del metals
    else:
        smhm = stellarmasses/fofmasses
        del stellarmasses
    label = "z=%d" % zz
    massbins = np.logspace(9, 14, 50)
    smhm_bin = np.zeros(np.size(massbins)-1)
    #smhm_lower = np.zeros(np.size(massbins)-1)
    #smhm_upper = np.zeros(np.size(massbins)-1)
    if metal:
        factor = 1/0.0122
    else:
        factor = omega0 / omegab
    for i in np.arange(np.size(massbins)-1):
        ii = np.where((fofmasses > massbins[i])*(fofmasses < massbins[i+1]))
        if np.size(ii) < 10:
            continue
        zpos = fofpos[ii] / box * nmesh
        fofzreion = np.array([zreion[int(zp[0]), int(zp[1]), int(zp[2])] for zp in zpos])
        jj = np.where(fofzreion < 7.5)
        smhm_bin[i] = np.median(smhm[ii][jj]) * factor
        #smhm_lower[i] = np.percentile(smhm[ii][jj], 16) * factor
        #smhm_upper[i] = np.percentile(smhm[ii][jj], 84) * factor
    masses = np.exp((np.log(massbins[1:]) + np.log(massbins[:-1]))/2.)
    ii = np.where(smhm_bin > 0)
    plt.semilogx(masses[ii], smhm_bin[ii], color=color, label=r"$z_{re} < 7.5$ "+label, ls="--")
    for i in np.arange(np.size(massbins)-1):
        ii = np.where((fofmasses > massbins[i])*(fofmasses < massbins[i+1]))
        if np.size(ii) < 10:
            continue
        zpos = fofpos[ii] / box * nmesh
        fofzreion = np.array([zreion[int(zp[0]), int(zp[1]), int(zp[2])] for zp in zpos])
        jj = np.where(fofzreion > 7.5)
        smhm_bin[i] = np.median(smhm[ii][jj]) * factor
        #smhm_lower[i] = np.percentile(smhm[ii][jj], 16) * factor
        #smhm_upper[i] = np.percentile(smhm[ii][jj], 84) * factor
    masses = np.exp((np.log(massbins[1:]) + np.log(massbins[:-1]))/2.)
    ii = np.where(smhm_bin > 0)
    plt.semilogx(masses[ii], smhm_bin[ii], color=color, label=r"$z_{re} > 7.5$ "+label, ls="-")
    #if scatter:
        #plt.fill_between(masses[ii], smhm_lower[ii], smhm_upper[ii], color=color, alpha=0.3)
    fname = "hm-z-%.1f.txt" % zz
    if star:
        if metal:
            fname = "data/starmetal-" + fname
        else:
            fname = "data/starm-" + fname
    else:
        if metal:
            fname = "data/gasmetal-" + fname
        else:
            fname = "data/gasm-" + fname
    #np.savetxt(fname, np.vstack((masses[ii], smhm_lower[ii], smhm_bin[ii], smhm_upper[ii])).T, header="# "+label+" (mass : SMHM (16, 50, 84))")

def plot_smhm_conc(pig, color=None, ls=None, zreion=None, nmesh=0):
    """Plot the gas mass to density at fixed halo mass. """
    bf = BigFile(pig)
    hh = bf["Header"].attrs["HubbleParam"]
    fofmasses = bf['FOFGroups/Mass'][:]*1e10/hh
    ##
    fofpos = bf['FOFGroups/MassCenterPosition'][:]
    ##
    gasmasses = bf['FOFGroups/MassByType'][:][:,0]*1e10/hh
    smhm = gasmasses/fofmasses
    del gasmasses
    #This is sum(m_i r_i r_j), the moment of inertia tensor.
    imom = bf['FOFGroups/Imom'][:]*1e10/hh**3
    #Average the diagonal elements of the inertia tensor to get a rough radius (in Mpc).
    modr = np.sqrt((imom[:,0] + imom[:,3+1] + imom[:,6+2])/3. / fofmasses)
    density = fofmasses / (4/3.*np.pi * modr**3)
    omega0 = bf["Header"].attrs["Omega0"]
    omegab = bf["Header"].attrs["OmegaBaryon"]
    zz = 1/bf["Header"].attrs["Time"]-1
    box = bf["Header"].attrs["BoxSize"]
    label = "z=%d" % zz
    factor = omega0 / omegab
    massbin = 1e10
    ii = np.where((fofmasses > massbin / 1.3)*(fofmasses < massbin * 1.3))
    (binned_smhm, bind_edges, _) = scipy.stats.binned_statistic(np.log10(density[ii]), smhm[ii], statistic='median', bins=20)
    plt.semilogx(10**((bind_edges[1:]+ bind_edges[:-1])/2), binned_smhm, ls="-", color=color)
    zpos = fofpos[ii] / box * nmesh
    fofzreion = np.array([zreion[int(zp[0]), int(zp[1]), int(zp[2])] for zp in zpos])
    jj = np.where(fofzreion < 7.5)
    (binned_smhm, bind_edges, _) = scipy.stats.binned_statistic(np.log10(density[ii][jj]), smhm[ii][jj], statistic='median', bins=20)
    plt.semilogx(10**((bind_edges[1:]+ bind_edges[:-1])/2), binned_smhm, ls="--", label=r"$z_{re} > 7.5$", color="blue")
    jj = np.where(fofzreion > 7.5)
    (binned_smhm, bind_edges, _) = scipy.stats.binned_statistic(np.log10(density[ii][jj]), smhm[ii][jj], statistic='median', bins=20)
    plt.semilogx(10**((bind_edges[1:]+ bind_edges[:-1])/2), binned_smhm, ls="-.", label=r"$z_{re} < 7.5$", color="red")

def plot_smhms_conc(reds, outdir):
    """Plot several SMHM over time."""
    snaps = find_snapshot(reds, snaptxt=os.path.join(outdir, "Snapshots.txt"))
    pigs = [os.path.join(outdir, "PIG_%03d") % ss for ss in snaps]
    colors = ["black", "red", "blue", "brown", "grey", "orange"]
    lss = ["-", "-.", "--", ":"]
    uvf = BigFile(os.path.join(outdir, "../UVFluctuationFile"))
    zreion = uvf["Zreion_Table"][:]
    nmesh = uvf["Zreion_Table"].attrs["Nmesh"][0]
    zreion = zreion.reshape((nmesh, nmesh, nmesh))
    uvf.close()
    for ii in np.arange(len(reds)):
        plot_smhm_conc(pigs[ii], color=colors[ii], ls=lss[ii % 4], zreion=zreion, nmesh=nmesh)
    plt.xlabel(r"$\rho (M_\odot/\mathrm{Mpc}^3)$")
    #plt.tight_layout()
    plt.ylabel(r"$M_g / M_\mathrm{h} (\Omega_M / \Omega_b)$")
    plt.legend(loc="lower center", ncol=2)
    plt.savefig("gmhms_conc.pdf")

def plot_smhms_reion(reds, outdir, star=True, metal=False):
    """Plot several SMHM over time."""
    snaps = find_snapshot(reds, snaptxt=os.path.join(outdir, "Snapshots.txt"))
    pigs = [os.path.join(outdir, "PIG_%03d") % ss for ss in snaps]
    colors = ["black", "red", "blue", "brown", "grey", "orange"]
    lss = ["-", "-.", "--", ":"]
    uvf = BigFile(os.path.join(outdir, "../UVFluctuationFile"))
    zreion = uvf["Zreion_Table"][:]
    nmesh = uvf["Zreion_Table"].attrs["Nmesh"][0]
    zreion = zreion.reshape((nmesh, nmesh, nmesh))
    uvf.close()
    for ii in np.arange(len(reds)):
        plot_smhm_reion(pigs[ii], color=colors[ii], ls=lss[ii % 4], zreion=zreion, nmesh=nmesh, star=star, metal=metal, scatter=(ii ==len(reds)-1))
    plt.xlabel(r"$M_\mathrm{h} (M_\odot)$")
    #plt.tight_layout()
    if star:
        if metal:
            plt.legend(loc="upper left")
            plt.ylabel(r"Z$_*$ (Z$_\odot$)")
            #plt.ylim(ymin=3e-4)
            plt.yscale('log')
            plt.savefig("starmetal_reion.pdf")
        else:
            plt.legend(loc="upper left")
            plt.yscale('log')
            plt.ylabel(r"$M_* / M_\mathrm{h} (\Omega_M / \Omega_b)$")
            plt.ylim(ymin=5e-4)
            plt.savefig("smhms_reion.pdf")
    else:
        if metal:
            plt.legend(loc="lower center")
            #plt.ylim(ymax=1.1)
            plt.yscale('log')
            plt.ylabel(r"Z$_\mathrm{g}$ (Z$_\odot$)")
            plt.savefig("gasmetal_reion.pdf")
        else:
            plt.ylabel(r"$M_g / M_\mathrm{h} (\Omega_M / \Omega_b)$")
            plt.legend(loc="lower center", ncol=2)
            plt.ylim(0.5, 1)
            plt.savefig("gmhms_reion.pdf")

def get_sfrd(outdir,Lbox=250,hh=0.6774):
    """Get the star formation rate density"""
    sfiles = outdir+'/sfr.txt*'
    files = sorted(glob.glob(sfiles))
    ts = []
    sfrs = []
    for f in files:
        d = np.loadtxt(f)
        if np.size(d) == 0:
            continue
        ts = np.append(ts,d[:,0])
        sfrs = np.append(sfrs,d[:,2])
    ts=np.array(ts)
    zs = np.array(1./ts - 1)
    zbin = np.linspace(3,12,300)
    zmid = (zbin[:-1]+zbin[1:])/2
    sout = []
    zout = []
    for i in range(0,len(zbin)-1):
        zmask = zs>zbin[i]
        zmask &= zs<zbin[i+1]
        if np.size(sfrs[zmask]) > 0:
            sout.append(np.max(sfrs[zmask]))
            zout.append((zbin[i] + zbin[i+1])/2)
    sout = np.array(sout)/((Lbox/hh)**3)
    zout = np.array(zout)
    return zout,sout

def plot_sfrd():
    """Plot the SFRD"""
    sfrdir = '/scratch3/06431/yueyingn/pack-asterix/sfr-file'
    zmid, sout = get_sfrd(sfrdir)
    np.savetxt("sfrd.txt", (zmid, np.log10(sout)))
    #From Salpeter to Chabrier
    corrfac = -0.26
    #Bouwens 2015, table 7. https://arxiv.org/abs/1403.4295
    data = np.array([[3.8 ,  -1.00 ,  0.06 , 0.06],
    [4.9,     -1.26,   0.06 , 0.06],
    [5.9 ,    -1.55,  0.06 , 0.06],
    [6.8  ,  -1.69 , 0.06 , 0.06],
    [7.9   , -2.08,  0.07 , 0.07],
    [10.4  ,  -3.13 , 0.45 ,0.36]])
    plt.plot(zmid, np.log10(sout), color="black")
    #Coe 2013  1211.3663
    coedata = np.log10(np.array([1.1e-3,1.4e-3]))
    coelogerr = np.log10(np.array([[-0.9e-3, -1.1e-3], [1.0e-3, 1.3e-3]]) + 10**coedata)-coedata
    coelogerr[0,:]*=-1
    plt.errorbar([9.6, 11], coedata+corrfac, fmt='^', yerr=coelogerr, xerr=0.5, color="red")
    plt.errorbar(data[:,0], data[:,1]+corrfac, fmt='o', yerr=(data[:,2], data[:,3]), xerr=0.5, color="grey")
    #Oesch 2014: https://arxiv.org/abs/1409.1228 text above fig 7.
    plt.errorbar([10,], np.array([-2.8,])+corrfac, fmt='s', yerr=[[0.5,], [0.3,]], xerr=0.5, color="purple")
    plt.xlabel('z')
    plt.ylabel(r'log SFRD ($M_\odot\, \mathrm{yr}^{-1}\, \mathrm{Mpc}^{-3}$)')
    plt.savefig("sfrd.pdf")

def get_reion_data(keylist):
    """Print data for ionized fraction"""
    ioniz_data = crc.get("ionized_fraction")
    for key in keylist:
        data = ioniz_data[key]
        errors = np.vstack([data.err_down, data.err_up])
        plt.errorbar(data.axes, 1-data.values, yerr=errors, fmt='d', label=key, color="black")
    Bosman_reion_data()

def Bosman_reion_data():
    """Data from Bosman 2021, 2108.03699. Table 6 with self-shielding.
    Note derived from a Nyx simulation."""
    reds = np.array([5.0, 5.1, 5.2, 5.3])
    mean = np.array([3.020, 3.336, 3.636, 3.598])*1e-5
    lowerr = np.array([0.058, 0.164, 0.095, 0.145])*1e-5
    uperr = np.array([0.230, 0.064, 0.131, 0.566])*1e-5
    errors = np.vstack([lowerr, uperr])
    plt.errorbar(reds, mean, yerr=errors, fmt='d', label="Bosman 2021", color="black")

def get_neutral_frac_sim(pp, nsamp=10000):
    """Get the neutral fraction from a particle snapshot."""
    bf = bigfile.BigFile(pp)
    partsz = bf["0/NeutralHydrogenFraction"].size
    print("Starting %s" % pp, flush=True)
    inds = np.random.randint(0, partsz, size=nsamp)
    nhi = np.array([bf["0/NeutralHydrogenFraction"][ii] for ii in inds])
    vol = np.array([bf["0/SmoothingLength"][ii] for ii in inds])**3
    xhi = np.sum(nhi * vol) / np.sum(vol)
    return xhi

def plot_reionization_history_sim(reds, outdir):
    """Reionization history as a function of redshift."""
    snaps = find_snapshot(reds, snaptxt=os.path.join(outdir, "Snapshots.txt"))
    part = [os.path.join(outdir, "PART_%03d") % ss for ss in snaps]
    print(reds,flush=True)
    #bf = BigFile(os.path.join(outdir, "../UVFluctuationFile"))
    #zreion = bf["Zreion_Table"][:]
    #zreion[np.where(zreion > 10)] = 10
    #bf.close()
    #plt.hist(zreion, 50, cumulative=True, density=True, histtype='step')
    plt.yscale('log')
    #plt.tight_layout()
    plt.savefig("reion_hist_sim.pdf")
    #Pick some random particles.
    with Pool(20) as p:
        mapres = p.map(get_neutral_frac_sim, part)
        xhi = np.array(mapres)
#        xhi = np.array([get_neutral_frac_sim(pp) for pp in part])
        print(xhi, flush=True)
        plt.semilogy(reds, xhi, ls="-", color="blue")
    plt.xlabel('z')
    plt.ylabel(r'$x_{HI}$')
    keylist = ['Davies et al. 2018', 'Fan et al. 2006', 'Greig et al. 2019', 'Hoag et al. 2019', 'Mason et al. 2018', 'Ono et al. 2012', 'Ota et al. 2008', 'Wang et al. 2020 (subm.)', 'Yang et al. 2020']
    get_reion_data(keylist)
    plt.yscale('log')
    #plt.tight_layout()
    plt.savefig("reion_hist_sim.pdf")

def plot_reionization_history(outdir):
    """Reionization history as a function of redshift."""
    bf = BigFile(os.path.join(outdir, "../UVFluctuationFile"))
    zreion = bf["Zreion_Table"][:]
    zreion[np.where(zreion > 10)] = 10
    bf.close()
    plt.hist(zreion, 50, cumulative=True, density=True)
    plt.xlabel('z')
    plt.ylabel(r'$x_{HI}$')
    keylist = ['Davies et al. 2018', 'Fan et al. 2006', 'Greig et al. 2019', 'Hoag et al. 2019', 'Mason et al. 2018', 'Ono et al. 2012', 'Ota et al. 2008', 'Wang et al. 2020 (subm.)', 'Yang et al. 2020']
    get_reion_data(keylist)
    plt.yscale('log')
    #plt.tight_layout()
    plt.savefig("reion_hist.pdf")

def plot_zreion(outdir, zval=0):
    """Plot a slice of reionization redshifts"""
    bf = BigFile(os.path.join(outdir, "../UVFluctuationFile"))
    zreion = bf["Zreion_Table"][:]
    nmesh = bf["Zreion_Table"].attrs["Nmesh"][0]
    box = bf["Zreion_Table"].attrs["BoxSize"]
    bf.close()
    zmap = zreion.reshape((nmesh, nmesh, nmesh))
    color_map = plt.imshow(zmap[:, :, zval], extent=([0, box, 0, box]), vmax=11)
    color_map.set_cmap("Blues")
    plt.colorbar()
    plt.xlabel(r'x (Mpc/h)')
    plt.ylabel(r'y (Mpc/h)')
    #plt.tight_layout()
    plt.savefig("reion_slice.pdf")

def modecount_rebin(kk, pk, modes, minmodes=20, ndesired=200):
    """Rebins a power spectrum so that there are sufficient modes in each bin"""
    assert np.all(kk) > 0
    logkk=np.log10(kk)
    mdlogk = (np.max(logkk) - np.min(logkk))/ndesired
    istart=iend=1
    count=0
    k_list=[kk[0]]
    pk_list=[pk[0]]
    targetlogk=mdlogk+logkk[istart]
    while iend < np.size(logkk)-1:
        count+=modes[iend]
        iend+=1
        if count >= minmodes and logkk[iend-1] >= targetlogk:
            pk1 = np.sum(modes[istart:iend]*pk[istart:iend])/count
            kk1 = np.sum(modes[istart:iend]*kk[istart:iend])/count
            k_list.append(kk1)
            pk_list.append(pk1)
            istart=iend
            targetlogk=mdlogk+logkk[istart]
            count=0
    k_list = np.array(k_list)
    pk_list = np.array(pk_list)
    return (k_list, pk_list)

def get_power(matpow, rebin=True):
    """Plot the power spectrum from CAMB
    (or anything else where no changes are needed)"""
    data = np.loadtxt(matpow)
    kk = data[:,0]
    ii = np.where(kk > 0.)
    #Rebin power so that there are enough modes in each bin
    kk = kk[ii]
    pk = data[:,1][ii]
    if rebin:
        modes = data[:,2][ii]
        return modecount_rebin(kk, pk, modes)
    return (kk,pk)

def plot_power(reds, outdir):
    """Plot some matter power spectra"""
    aa = 1/(1+reds)
    colors = ["black", "red", "blue", "brown", "pink", "orange"]
    for zz,cc in zip(reds, colors):
        (kk, pk) = get_power(os.path.join(outdir, "../class-planck15/ics_matterpow_99.dat-%.1f" % zz), rebin=False)
        plt.loglog(kk, pk, ls = ":", color=cc)
    for a,cc in zip(aa, colors):
        (kk, pk) = get_power(os.path.join(outdir, "powerspectrum-%.4f.txt" % a))
        plt.loglog(kk, pk, label="z=%d" % (1/a-1), color=cc)
    plt.legend()
    plt.ylabel(r"P(k) (Mpc/h)$^3$")
    plt.xlabel(r"k (h/Mpc)")
    plt.xlim(0.01, 200)
    plt.ylim(1e-4, 2e3)
    #plt.tight_layout()
    plt.savefig("matterpower.pdf")

if __name__ == "__main__":
    simdir = sys.argv[1]
    red = np.array([9, 8.3, 8, 7.5, 7, 6.5, 6, 5])
    plt.figure()
    plot_avg_sfr_reion(red, outdir=simdir)
    plt.clf()
    #red = np.array([4,3.5,3.0])
    #plot_avg_sfr_heii_reion(red, outdir=simdir)
    #plt.clf()
    red = np.array([12,10,9,8,7,6,5,4.53,4,3.5,3.0])
    plot_avg_sfr(red, outdir=simdir)
    plt.clf()
    plot_ssfr(red, outdir=simdir)
    plt.clf()
    plot_power(np.array([10, 8, 6.12, 4, 3]), outdir=simdir)
    plt.clf()
    reds2 = np.array([8, 6, 4, 3])
    plot_smhms(reds2, outdir=simdir, metal=True)
    plt.clf()
    plot_smhms(reds2, outdir=simdir, star=False, metal=True)
    plt.clf()
    plot_smhms(reds2, outdir=simdir)
    plt.clf()
    plot_smhms(reds2, outdir=simdir, star=False, hdf5=True)
    plt.clf()
    reds2 = np.array([4, 3])
    plot_smhms_he_reion(reds2, outdir=simdir, metal=True)
    plt.clf()
    plot_smhms_he_reion(reds2, outdir=simdir, star=False, metal=True)
    plt.clf()
    plot_smhms_he_reion(reds2, outdir=simdir)
    plt.clf()
    plot_smhms_he_reion(reds2, outdir=simdir, star=False)
    plt.clf()

    red_reion = np.array([10,9,8.5,8,7.5,7,6.75,6.5,6.25,6,5.75,5.5,5])
    plot_reionization_history_sim(red_reion,outdir=simdir)
    plt.clf()
    #plot_zreion(outdir=simdir)
    #plt.clf()
