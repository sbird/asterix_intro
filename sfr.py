"""Make some plots for the asterix paper."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from bigfile import BigFile
import matplotlib.font_manager

plt.rcParams['axes.linewidth'] = 1.8
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 13
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
    stellarmasses = bf['FOFGroups/MassByType'][:][:,4]*1e10
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
    labels = [r"$4\times 10^6$", r"$10^7$", r"$10^8$", r"$10^{9}$", r"$10^{10}$"]
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
    plt.legend(loc="lower right")
    plt.yscale('log')
    plt.ylim(1e-9,4e-8)
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
    uvf = BigFile(os.path.join(outdir, "../UVFluctuationFile"))
    zreion = uvf["Zreion_Table"][:]
    nmesh = uvf["Zreion_Table"].attrs["Nmesh"][0]
    zreion = zreion.reshape((nmesh, nmesh, nmesh))
    uvf.close()
    sfrs_reion, sfrs_no_reion = zip(*[get_avg_sfr_reion(pig, zreion, nmesh) for pig in pigs])
    sfrs_reion = np.array(sfrs_reion)
    sfrs_no_reion = np.array(sfrs_no_reion)
    plt.xlabel("z")
    plt.ylabel(r"SFR ($M_\odot$ yr$^{-1}$)")
    plt.yscale('log')
    plt.ylim(ymin=1e-6)
    plt.xlim(xmin=5.5, xmax=reds[0])
    for i in range(np.shape(sfrs_reion)[1]):
        j = np.where(sfrs_reion[:,i,2] > 0)
        plt.fill_between(reds[j], sfrs_reion[j[0],i,0]+1e-14, sfrs_reion[j[0],i,2], color=colors_reion[i], alpha=0.15)
        j = np.where(sfrs_reion[:,i,1] > 0)
        plt.plot(reds[j], sfrs_reion[j[0],i,1], label="HII: "+labels[i], color=colors_reion[i], ls="-")
        #j = np.where(sfrs_no_reion[:,i,2] > 0)
        #plt.fill_between(reds[j], sfrs_no_reion[j[0],i,0]+1e-14, sfrs_no_reion[j[0],i,2], color=colors_no_reion[i], alpha=0.15)
        j = np.where(sfrs_no_reion[:,i,1] > 0)
        plt.plot(reds[j], sfrs_no_reion[j[0],i,1], label="HI: " + labels[i], color=colors_no_reion[i], ls="--")
        #np.savetxt("asfr-mdm%d.txt" % i, np.vstack((reds[j], sfrs[j[0],i,:].T)).T, header="# MDM = "+labels[i]+" (redshift : SFR 16, 50, 84 percentiles )")
    plt.ylim(ymin=1e-6, ymax=2)
    plt.xlim(xmin=5.5, xmax=reds[0])
    plt.legend(loc="lower left", ncol=2)
    #plt.tight_layout()
    plt.savefig("avg_sfr_reion.pdf")

def get_avg_sfr_heii_reion(pig):
    """Get the averaged SFR for some different halo masses."""
    bf = BigFile(pig)

    fofmasses = bf['FOFGroups/Mass'][:]*1e10/bf["Header"].attrs["HubbleParam"]
    sfr = bf['FOFGroups/StarFormationRate'][:]
    offsets = np.concatenate((np.zeros(1), np.cumsum(bf['FOFGroups/LengthByType'][:][:,0])))
    massbins = np.array([4e9, 1e10, 1e11])
    avgsfr_reion = np.zeros((len(massbins),3))
    avgsfr_no_reion = np.zeros((len(massbins),3))
    for i, mm in enumerate(massbins):
        ii = np.where((fofmasses < mm*1.3)*(fofmasses > mm / 1.3))
        if np.size(ii) == 0:
            print("no halos for mass %g pig %s" % (mm, pig))
            continue
        print("mass %.1f : %d" % (mm, np.size(ii)))
        heiifrac = np.array([np.mean(bf['0/HeIIIIonized'][int(offsets[i]):int(offsets[i]+10)]) for i in ii[0]])
        jj = np.where(heiifrac > 0.9)
        if np.size(jj) > 0:
            avgsfr_reion[i] = np.percentile(sfr[ii][jj], [16, 50, 84])
        jj = np.where(heiifrac < 0.1)
        if np.size(jj) > 0:
            avgsfr_no_reion[i] = np.percentile(sfr[ii][jj], [16, 50, 84])
    del offsets
    del sfr
    del fofmasses
    return avgsfr_reion, avgsfr_no_reion

def plot_avg_sfr_heii_reion(reds, outdir):
    """Average SFR over time"""
    snaps = find_snapshot(reds, snaptxt=os.path.join(outdir, "Snapshots.txt"))
    pigs = [os.path.join(outdir, "PIG_%03d") % ss for ss in snaps]
    colors_reion = ["darkred", "blue", "brown", "green"]
    colors_no_reion = ["red", "blue", "orange", "grey"]
    labels = [r"$5\times 10^{9}$", r"$10^{10}$", r"$5\times 10^{10}$"]
    sfrs_reion, sfrs_no_reion = zip(*[get_avg_sfr_heii_reion(pig) for pig in pigs])
    sfrs_reion = np.array(sfrs_reion)
    sfrs_no_reion = np.array(sfrs_no_reion)
    for i in range(np.shape(sfrs_reion)[1]):
        j = np.where(sfrs_reion[:,i,2] > 0)
        plt.fill_between(reds[j], sfrs_reion[j[0],i,0]+1e-14, sfrs_reion[j[0],i,2], color=colors_reion[i], alpha=0.2)
        j = np.where(sfrs_reion[:,i,1] > 0)
        plt.plot(reds[j], sfrs_reion[j[0],i,1], label=labels[i]+" reion", color=colors_reion[i], ls="-")
        j = np.where(sfrs_no_reion[:,i,2] > 0)
        plt.fill_between(reds[j], sfrs_no_reion[j[0],i,0]+1e-14, sfrs_no_reion[j[0],i,2], color=colors_no_reion[i], alpha=0.2)
        j = np.where(sfrs_no_reion[:,i,1] > 0)
        plt.plot(reds[j], sfrs_no_reion[j[0],i,1], label=labels[i]+" no reion", color=colors_no_reion[i], ls="--")
        #np.savetxt("asfr-mdm%d.txt" % i, np.vstack((reds[j], sfrs[j[0],i,:].T)).T, header="# MDM = "+labels[i]+" (redshift : SFR 16, 50, 84 percentiles )")
    plt.xlabel("z")
    plt.ylabel(r"SFR ($M_\odot$ yr$^{-1}$)")
    plt.legend(loc="upper left")
    plt.yscale('log')
    plt.ylim(ymin=1e-5)
    #plt.tight_layout()
    plt.savefig("avg_sfr_heii_reion.pdf")

def plot_smhm(pig, color=None, ls=None, star=True, metal=False):
    """Plot the stellar/gas mass to halo mass. If star is True, stars, else gas."""
    bf = BigFile(pig)
    fofmasses = bf['FOFGroups/Mass'][:]*1e10
    if star:
        if metal:
            metals = bf["FOFGroups/StellarMetalMass"][:]
            starmass = bf['FOFGroups/MassByType'][:][:,4]
        else:
            stellarmasses = bf['FOFGroups/MassByType'][:][:,4]*1e10
    else:
        if metal:
            metals = bf["FOFGroups/GasMetalMass"][:]
            starmass = bf['FOFGroups/MassByType'][:][:,0]
        else:
            stellarmasses = bf['FOFGroups/MassByType'][:][:,0]*1e10
    omega0 = bf["Header"].attrs["Omega0"]
    omegab = bf["Header"].attrs["OmegaBaryon"]
    zz = 1/bf["Header"].attrs["Time"]-1
    if metal:
        ii = np.where(starmass > 0)
        smhm = metals[ii]/starmass[ii]
        fofmasses = fofmasses[ii]
        del starmass
        del metals
    else:
        smhm = stellarmasses/fofmasses
        del stellarmasses
    label = "z=%.1f" % zz
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
        if np.size(ii) == 0:
            continue
        smhm_bin[i] = np.median(smhm[ii]) * factor
        smhm_lower[i] = np.percentile(smhm[ii], 16) * factor
        smhm_upper[i] = np.percentile(smhm[ii], 84) * factor
    masses = np.exp((np.log(massbins[1:]) + np.log(massbins[:-1]))/2.)
    ii = np.where(smhm_bin > 0)
    plt.semilogx(masses[ii], smhm_bin[ii], color=color, label=label, ls=ls)
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

def plot_smhms(reds, outdir, star=True, metal=False):
    """Plot several SMHM over time."""
    snaps = find_snapshot(reds, snaptxt=os.path.join(outdir, "Snapshots.txt"))
    pigs = [os.path.join(outdir, "PIG_%03d") % ss for ss in snaps]
    colors = ["black", "red", "blue", "brown", "pink", "orange"]
    lss = ["-", "-.", "--", ":"]
    for ii in np.arange(len(reds)):
        plot_smhm(pigs[ii], color=colors[ii], ls=lss[ii % 4], star=star, metal=metal)
    plt.xlabel(r"$M_\mathrm{h} (M_\odot)$")
    #plt.tight_layout()
    if star:
        if metal:
            plt.legend(loc="upper left")
            plt.ylabel(r"Z$_*$ (Z$_\odot$)")
            #plt.ylim(ymin=3e-4)
            plt.yscale('log')
            plt.savefig("starmetal.pdf")
        else:
            plt.legend(loc="upper left")
            plt.ylabel(r"$M_* / M_\mathrm{h} (\Omega_M / \Omega_b)$")
            plt.savefig("smhms.pdf")
    else:
        if metal:
            plt.legend(loc="lower center")
            #plt.ylim(ymax=1.1)
            plt.yscale('log')
            plt.ylabel(r"Z$_\mathrm{g}$ (Z$_\odot$)")
            plt.savefig("gasmetal.pdf")
        else:
            plt.ylabel(r"$M_g / M_\mathrm{h} (\Omega_M / \Omega_b)$")
            plt.legend(loc="lower center")
            plt.ylim(0.5, 1)
            plt.savefig("gmhms.pdf")

def plot_reionization_history(outdir):
    """Reionization history as a function of redshift."""
    bf = BigFile(os.path.join(outdir, "../UVFluctuationFile"))
    zreion = bf["Zreion_Table"][:]
    zreion[np.where(zreion > 11)] = 11
    bf.close()
    plt.hist(zreion, 50, cumulative=True, density=True)
    plt.xlabel('z')
    plt.ylabel(r'$x_{HI}$')
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
    reds2 = np.array([12, 10, 8, 6, 4, 3])
    plot_smhms(reds2, outdir=simdir, metal=True)
    plt.clf()
    plot_smhms(reds2, outdir=simdir, star=False, metal=True)
    plt.clf()
    plot_smhms(reds2, outdir=simdir)
    plt.clf()
    plot_smhms(reds2, outdir=simdir, star=False)
    plt.clf()

    plot_reionization_history(outdir=simdir)
    plt.clf()
    plot_zreion(outdir=simdir)
    plt.clf()
