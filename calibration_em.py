#! /usr/bin/env python

import numpy as np
import os.path,sys
np.set_printoptions(threshold=sys.maxsize)
#from Cosmology import *
import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
#from distinct_colours import get_distinct 
#from stats import *
#import mpl_style
#plt.style.use(mpl_style.style1)

# code to produce calibration plots in Elliot et al. paper
# (only 1 SAM run divided into 64 subvolumes)
# for more than 1 model

def percentiles(val,data,weights=None):
    """
    Examples
    --------
    >>> import numpy as np
    >>> import stats
    >>> data = np.array(np.arange(0.,100.,10.))
    >>> stats.percentiles(0.5,data)
    >>> 45.0
    """
    if (val <0 or val >1):
        sys.exit('STOP percentiles: 0<val<1')

    if (weights is None):
        ws = np.zeros(shape=(len(data))) ; ws.fill(1.)
    else:
        ws = weights
        
    data = np.array(data) ; ws = np.array(ws)
    ind_sorted = np.argsort(data)  # Median calculation from wquantiles
    sorted_data = data[ind_sorted] ; sorted_weights = ws[ind_sorted]
    num = np.cumsum(sorted_weights) - 0.5*sorted_weights
    den = np.sum(sorted_weights)
    if (den!=0):
        pn = num/den
        percentiles = np.interp(val, pn, sorted_data)
    else:
        sys.exit('STOP percentiles: problem with weights')
    return percentiles



path = '/home/chandro/Galform_Out/simulations/'
model = '/fnl_sam/emulator/n512_fid_G4/gp19.vimal.em.fnl_sam/'
nmodels = 1000
input_file = "/home/chandro/emulator_project/hypercupe_10p_1000.dat"
nvol = 64
end = '_fnl_sam_G4_em'
z = 0
z1 = 1.1
outpath = '/home/chandro/galform/elliott/'

# histogram K LF (m-5logh in AB)
mlow_K = -28
mupp_K = -15
dm_K = 0.25
mbins_K = np.arange(mlow_K,mupp_K,dm_K)
xmf_K = mbins_K + dm_K/2.0

# histogram r LF (m-5logh in AB)
mlow_r = -25
mupp_r = -12
dm_r = 0.1
mbins_r = np.arange(mlow_r,mupp_r,dm_r)
xmf_r = mbins_r + dm_r/2.0

# histogram r LF (m-5logh in AB)
# early and low tipes, zstar
mlow_r_lb = -25
mupp_r_lb = -12
dm_r_lb = 0.5
mbins_r_lb = np.arange(mlow_r_lb,mupp_r_lb,dm_r_lb)
xmf_r_lb = mbins_r_lb + dm_r_lb/2.0

# histogram HIMF (Msun/h^2)
mupp_HI = 12
mlow_HI = 5
dm_HI = 0.1
mbins_HI = np.arange(mlow_HI,mupp_HI,dm_HI)
xmf_HI = mbins_HI + dm_HI/2.0
    
# histogram Vc (km/s)
mlow_vc = 0
mupp_vc = 5
dm_vc = 0.1
mbins_vc = np.arange(mlow_vc,mupp_vc,dm_vc)
xmf_vc = mbins_vc + dm_vc/2.0

# histogram Mbulge (Msun/h)
mlow_BH = 7
mupp_BH = 14
dm_BH = 0.25
mbins_BH = np.arange(mlow_BH,mupp_BH,dm_BH)
xmf_BH = mbins_BH + dm_BH/2.0
    
# histogram mgas fraction (Msun)
mlow_g = 12
mupp_g = 16
dm_g = 0.2
mbins_g = np.arange(mlow_g,mupp_g,dm_g)
xmf_g = mbins_g + dm_g/2.0

index = 0

# Bands for K LF
#iband_K = 'UKIRT-K' ; iband6 = 'K' ;
iband_K = 'UK' ; iband6 = 'K' ;
# Bands for r LF
#iband_r = 'SDSS-r' ; iband1 = 'r' ; iband14 = 14
iband_r = 'SD' ; iband1 = 'r' ;
# Bands for I LF
iband_I = 'I' ;
# Bands for B LF
iband_B = 'B' ;
# Bands for V LF
iband_V = 'V' ;

## plot
#fig = plt.figure(constrained_layout=True,figsize=(15,12))
#gs = fig.add_gridspec(3, 3)
#ax00 = fig.add_subplot(gs[0,0])
#ax01 = fig.add_subplot(gs[0,1])
#ax02 = fig.add_subplot(gs[0,2])
#ax10 = fig.add_subplot(gs[1,0])
#ax11 = fig.add_subplot(gs[1,1])
#ax12 = fig.add_subplot(gs[1,2])
#ax20 = fig.add_subplot(gs[2,0])
#ax21 = fig.add_subplot(gs[2,1])
#ax22 = fig.add_subplot(gs[2,2])
#
#ax00.set_xlabel("$ M_{K}^{AB} - 5\\rm{log}(\it{h})$")
#ax00.set_ylabel("$\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
#ax00.set_xlim(-18,-24.5)
#ax00.set_ylim(-6.5,-0.5)
#ax01.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
#ax01.set_ylabel("$\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
#ax01.set_xlim(-18,-24.5)
#ax01.set_ylim(-7,-1)
#ax02.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
#ax02.set_ylabel('$\\rm{log}(\it{r_{50}}/ \it{h^{-1}}\\rm{kpc} )$')
#ax02.set_xlim(-15,-24)
#ax02.set_ylim(-1,1.5)
#ax10.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
#ax10.set_ylabel('$\\rm{log}(\it{r_{50}}/ \it{h^{-1}}\\rm{kpc} )$')
#ax10.set_xlim(-15,-24)
#ax10.set_ylim(-1,1.5)
#ax11.set_xlabel('$\\rm{log}(\it{M_{HI}}/ \it{h^{-2}}\\rm{M_{\odot}} )$')
#ax11.set_ylabel("$\\rm{log}(\\rm{dn/dlog\it{M_{HI}}/ \it{h^{3}} \\rm{Mpc^{-3}}})$")
#ax11.set_xlim(8,11)
#ax11.set_ylim(-6,0)
#ax12.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
#ax12.set_ylabel("f(early type)")
#ax12.set_xlim(-15,-24)
#ax12.set_ylim(0,1)
#ax20.set_xlabel('$\\rm{log}(\it{V_{c}}/\\rm{km s^{-1}})$')
#ax20.set_ylabel("$ M_{I}^{Vega} - 5\\rm{log}(\it{h})$")
#ax20.set_xlim(1.5,2.65)
#ax20.set_ylim(-24,-13)
#ax20.invert_yaxis()
#ax21.set_xlabel('$\\rm{log}(\it{M_{bulge}}/ \it{h^{-1}}\\rm{M_{\odot}} )$')
#ax21.set_ylabel('$\\rm{log}(\it{M_{BH}}/ \it{h^{-1}}\\rm{M_{\odot}} )$')
#ax21.set_xlim(8,13)
#ax21.set_ylim(5,10)
#ax22.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
#ax22.set_ylabel('$\\rm{log}(\it{Z_{star}}(\\rm{V-wt}))$')
#ax22.set_xlim(-17,-23)
#ax22.set_ylim(-2.8,-1.2)
# 
#fig1, ax1 = plt.subplots(figsize=(15,12))
#fig2, ax2 = plt.subplots(figsize=(15,12))
#ax1.set_xlabel("$ M_{K}^{AB} - 5\\rm{log}(\it{h})$")
#ax1.set_ylabel("$\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
#ax1.set_xlim(-18,-26)
#ax1.set_ylim(-6,-1)
#ax1.text(-24,-2,'z=1.1',fontsize=25)
#ax2.set_xlabel('$\\rm{log}(\it{M_{hhalo}}/ \it{M_{\odot}})$')
#ax2.set_ylabel('$M_{hot,gas}/M_{hhalo}$')
#ax2.set_xlim(13,15)
#ax2.set_ylim(0,0.16)
#
#
#plt.rcParams.update({'font.size': 12})
#matplotlib.rc('xtick', labelsize=18)
#matplotlib.rc('ytick', labelsize=18)
#
# defining arrays
plfe_K = np.zeros(shape = (nmodels,len(mbins_K)))
plfe_K_z1 = np.zeros(shape = (nmodels,len(mbins_K)))
plfe_r = np.zeros(shape = (nmodels,len(mbins_r)))
plfe_et = np.zeros(shape = (nmodels,len(xmf_r_lb)))
plfe_lt = np.zeros(shape = (nmodels,len(xmf_r_lb)))
nf = np.zeros(shape = (nmodels,len(mbins_r)))
ntot = np.zeros(shape = (nmodels,len(mbins_r)))
plfe_HI = np.zeros(shape = (nmodels,len(mbins_HI)))
plfe_TF = np.zeros(shape = (nmodels,len(xmf_vc)))
plfe_BH = np.zeros(shape = (nmodels,len(xmf_BH)))
plfe_z = np.zeros(shape = (nmodels,len(xmf_r_lb)))
plfe_g = np.zeros(shape = (nmodels,len(xmf_g)))

ndens_K = []
ndens_K_z1 = []
ndens_r = []
ndens_e = []
ndens_l = []
ndens_f = []
ndens_HI = []
ndens_TF = []
ndens_BH = []
ndens_z = []
ndens_g = []

# read input_free_file
file = open(input_file)
line = file.readline() # comment line
# loop over different runs
for run in range(nmodels):

    line = file.readline()
    # read free parameters
    f_stab = line.split(' ')[0]
    alpha_cool = line.split(' ')[1]
    alpha_ret = line.split(' ')[2]
    gamma_SN = line.split(' ')[3]
    V_SNd = line.split(' ')[4]
    V_SNb = line.split(' ')[5]
    f_burst = line.split(' ')[6]
    f_ellip = line.split(' ')[7]
    v_SF = line.split(' ')[8]
    f_SMBH = line.split(' ')[9]
    
    run = 'Fstab'+f_stab+'_Acool'+alpha_cool+'_Aret'+alpha_ret+'gSN'+gamma_SN+'vSNd'+V_SNd+'vSNb'+V_SNb+'Fburst'+f_burst+'Fellip'+f_ellip+'vSF'+v_SF+'fSMBH'+f_SMBH
    
    # obtain ngal
    ngal = 0
    for ivol in range(nvol):
        # z=0
        infile = path+models[run]+'/iz128/ivol'+str(ivol)+'/galaxies.hdf5'
        if (os.path.isfile(infile)):
            f = h5py.File(infile,'r')
            group = f['Output001']            
            dim = np.shape(group['xgal'][()])[0]
            ngal += dim
            f.close()
    
    r50_e_A = np.zeros(shape=(ngal))
    r50_e_A[:] = np.nan
    mag_r_e_A = np.zeros(shape=(ngal))
    mag_r_e_A[:] = np.nan
    r50_l_A = np.zeros(shape=(ngal))
    r50_l_A[:] = np.nan
    mag_r_l_A = np.zeros(shape=(ngal))
    mag_r_l_A[:] = np.nan
    vc_TF_A = np.zeros(shape=(ngal))
    vc_TF_A[:] = np.nan
    mag_I_TF_A = np.zeros(shape=(ngal))
    mag_I_TF_A[:] = np.nan
    mbh_BH_A = np.zeros(shape=(ngal))
    mbh_BH_A[:] = np.nan
    mbulge_BH_A = np.zeros(shape=(ngal))
    mbulge_BH_A[:] = np.nan
    mag_r_z_A = np.zeros(shape=(ngal))
    mag_r_z_A[:] = np.nan
    zstar_z_A = np.zeros(shape=(ngal))
    zstar_z_A[:] = np.nan
    mhot_A = np.zeros(shape=(ngal))
    mhot_A[:] = np.nan
    mhhalo_A = np.zeros(shape=(ngal))
    mhhalo_A[:] = np.nan
    
    print(models[run])
    volh = 0.
    first = True

    ni = 0
    for ivol in range(nvol):

        # z=0
        infile = path+models[run]+'/iz128/ivol'+str(ivol)+'/galaxies.hdf5'
        if (os.path.isfile(infile)):
            print(infile)
            f = h5py.File(infile,'r')
            volh = volh + f['Parameters/volume'][()]
            h0 =   f['Parameters/h0'][()]
            if index == 0:
                omega0 = f['Parameters/omega0'][()]
                omegab = f['Parameters/omegab'][()]
                lambda0 =f['Parameters/lambda0'][()]
                # rest frame offset AB to Vega system
                a0_ab_K = f['Bands/a0_ab'][0]
                a0_ab_B = f['Bands/a0_ab'][1]
                a0_ab_I = f['Bands/a0_ab'][2]
                #print(a0_ab_I)
                a0_ab_bJ = f['Bands/a0_ab'][3]
                a0_ab_r = f['Bands/a0_ab'][4]
                a0_ab_V = f['Bands/a0_ab'][5]
            group = f['Output001']
            dime = np.shape(group['xgal'][()])[0]
            
            # K LF
            mag_K = group['mag'+iband_K+'r_tot_ext'][()] # m-5logh (AB)
            mag_K_vega = mag_K + a0_ab_K # m-5logh (Vega)
            # r LF
            mag_r = group['mag'+iband_r+'r_tot_ext'][()] # m-5logh (AB)
            mag_r_vega = mag_r + a0_ab_r # m-5logh (Vega)
            # early and late type galaxy sizes, early fraction
            rbulge = group['rbulge'][()] # Mpc/h
            rdisk = group['rdisk'][()] # Mpc/h
            btr_r = group['BoverT'+iband_r+'_r_ext'][()]
            # HI MF, Tully-Fisher
            mcold = group['mcold'][()] # Msun/h
            mcold_mol = group['mcold_mol'][()] # Msun/h
            # Tully-Fisher
            mag_I = group['mag'+iband_I+'r_tot_ext0'][()] # m-5logh (AB)
            mag_I = mag_I + a0_ab_I # m-5logh (Vega)
            btr_B = group['BoverT'+iband_B+'_r_ext'][()]
            vdisk = group['vdisk'][()] # km/s
            vhalo = group['vhalo'][()] # km/s
            mstars_disk = group['mstars_disk'][()] # Msun/h
            isat = group['is_central'][()]
            # bulge-BH mass
            mbh = group['M_SMBH'][()] # Msun/h
            mstars_bulge = group['mstars_bulge'][()] # Msun/h
            # Zstars
            #zstar_bulge = group['star_metal_bulge'][()] # Msun/h
            #zstar_disk = group['star_metal_disk'][()] # Msun/h
            zstar = group['met'+iband_V+'_tot'][()] 
            mhalo = group['mhhalo'][()] # Msun/h
            # mgas fraction
            mhot = group['mhot'][()] # Msun/h
            mhhalo = group['mhhalo'][()] # Msun/h
            
            f.close()
               
                                                
            # K LF
            H_K, bins_edges_K = np.histogram(mag_K,bins=np.append(mbins_K,mupp_K)) # m-5logh (AB)
            plfe_K[index,:] = plfe_K[index,:] + H_K
            
            # r LF
            H_r, bins_edges_r = np.histogram(mag_r,bins=np.append(mbins_r,mupp_r)) # m-5logh (AB)
            plfe_r[index,:] = plfe_r[index,:] + H_r
            
            # galaxy sizes
            # convert radii to kpc/h
            rdisk = 1e3*rdisk
            rbulge = 1e3*rbulge
            # convert rbulge from 3D to 2D half-mass radius
            r50bulge = rbulge/1.35
            # assume measured r50 for disk same as face-on value
            r50disk = rdisk
            # estimate half-light radius in r-band is simple average of disk & bulge
            r50 = (1-btr_r)*r50disk + btr_r*r50bulge
            # early type
            ind_e = np.where(btr_r>0.5)
            mag_r_e = mag_r[ind_e]
            r50_e = r50[ind_e] # kpc/h
            r50_e_A[ni+ind_e[0]] = r50_e
            mag_r_e_A[ni+ind_e[0]] = mag_r_e
            # late type
            ind_l = np.where(btr_r<0.5)
            mag_r_l = mag_r[ind_l] 
            r50_l = r50[ind_l] # kpc/h
            r50_l_A[ni+ind_l[0]] = r50_l
            mag_r_l_A[ni+ind_l[0]] = mag_r_l
            
            # early type fraction
            H_tot, bins_edges_tot = np.histogram(mag_r,bins=np.append(mbins_r,mupp_r)) # m-5logh (AB)
            ntot[index,:] = ntot[index,:] + H_tot            
            H_f, bins_edges_f = np.histogram(mag_r_e,bins=np.append(mbins_r,mupp_r)) # m-5logh (AB)
            nf[index,:] = nf[index,:] + H_f
            
            # HI MF
            # extract HI mass - ignore bursts at z=0 for now
            # mcold is total cold gas mass (incluing He)
            # but mcold_mol is molecular gas mass (excluding He)
            # assuming hydrogen mass fraction XH = 0.74
            mHI = 0.74*mcold - mcold_mol
            # convert Mcold from h^-1 Msun to h^-2 Msun units
            mcold = mHI*h0
            H_HI, bins_edges_HI = np.histogram(np.log10(mcold+1e-10),bins=np.append(mbins_HI,mupp_HI)) # log10(M_HI*h^2/Msun)
            plfe_HI[index,:] = plfe_HI[index,:] + H_HI
            
            # Tully-Fisher
            # consider only disk galaxies
            #vc = np.log10(vhalo)
            vc = np.log10(vdisk+1e-10) # log10(v*s/km)
            # fullfilling some conditions (Sb-Sdm late-type galaxies)
            ind_TF = np.where((btr_B<0.2)&(mcold/(mcold+mstars_disk)>0.1))
            mag_I_TF = mag_I[ind_TF]
            vc_TF = vc[ind_TF]
            vc_TF_A[ni+ind_TF[0]] = vc_TF
            mag_I_TF_A[ni+ind_TF[0]] = mag_I_TF
            
            # bulge-BH mass
            ind_BH = np.where(btr_r>0.3)
            mbh_BH = mbh[ind_BH]
            mbulge_BH = np.log10(mstars_bulge[ind_BH]+1e-10)  # log10(M_bulge*h/Msun)
            mbh_BH_A[ni+ind_BH[0]] = mbh_BH
            mbulge_BH_A[ni+ind_BH[0]] = mbulge_BH
            
            # Zstars
            #zstar = (zstar_bulge+zstar_disk)/(mstars_bulge+mstars_disk)
            ind_z = np.where((btr_r>0.5)&(mhalo>1e14))
            mag_r_z = mag_r[ind_z]
            zstar_z = zstar[ind_z]
            mag_r_z_A[ni+ind_z[0]] = mag_r_z
            zstar_z_A[ni+ind_z[0]] = zstar_z
            
            # mgas fraction
            mhot = mhot/h0 # Msun
            mhhalo = mhhalo/h0 # Msun
            ind_type = np.where(isat==1)
            for i in range(len(ind_type[0])):
                if i>0:
                    ind0 = ind_type[0][i-1]
                    ind1 = ind_type[0][i]
                    mhot_A[ni+i] = sum(mhot[ind0:ind1])
                    mhhalo_A[ni+i] = mhhalo[ind0]

            ni += dime

        else:
            print('Not found: ',infile)                    
                        
        # z=1.1
        infile = path+models[run]+'/iz95/ivol'+str(ivol)+'/galaxies.hdf5'
        if (os.path.isfile(infile)):
            print(infile)
            f = h5py.File(infile,'r')
            group = f['Output001']

            # K LF z=1.1
            mag_K_z1 = group['mag'+iband_K+'r_tot_ext'][()] # m-5logh (AB)
            f.close()

                        
            # K LF z=1.1
            H_K_z1, bins_edges_K = np.histogram(mag_K_z1,bins=np.append(mbins_K,mupp_K)) # m-5logh (AB)
            plfe_K_z1[index,:] = plfe_K_z1[index,:] + H_K_z1
            
        else:
            print('Not found: ',infile)

    # Take logs and normalize
    print('Total volume considered = (', volh, ' (Mpc/h)^3)')
    print('Total volume considered = (', np.power(volh,1./3.), ' Mpc/h)^3')
    volh = 1000**3
    vol = volh/pow(h0,3.) # in Mpc^3

    

    # K LF 
    ind_K = np.where(plfe_K[index,:] > 0)
    plfe_K[index,ind_K] = np.log10(plfe_K[index,ind_K]/dm_K/volh)
    if index == 0:
        ndens_K.append(xmf_K)
    ndens_K.append(plfe_K[index,:])

    # K LF z=1.1 
    ind_K_z1 = np.where(plfe_K_z1[index,:] > 0)
    plfe_K_z1[index,ind_K_z1] = np.log10(plfe_K_z1[index,ind_K_z1]/dm_K/volh)
    if index == 0:
        ndens_K_z1.append(xmf_K)
    ndens_K_z1.append(plfe_K_z1[index,:])
    
    # r LF
    ind_r = np.where(plfe_r[index,:] > 0)
    plfe_r[index,ind_r] = np.log10(plfe_r[index,ind_r]/dm_r/volh)
    if index == 0:
        ndens_r.append(xmf_r)
    ndens_r.append(plfe_r[index,:])
    
    # early and late type
    # early
    ind_nan = np.where(mag_r_e_A[:]!=np.nan)
    mag_r_e_A_p = mag_r_e_A[ind_nan]
    r50_e_A_p = r50_e_A[ind_nan]
    ind_e_p = np.digitize(mag_r_e_A_p,mbins_r_lb)
    for i in range(1,np.amax(ind_e_p)+1):
        pos = np.where(np.array(ind_e_p)==i)
        if np.shape(pos)[1] >= 10:
            plfe_et[index,i-1] = percentiles(0.5,np.log10(r50_e_A_p[pos])) # log10(r*h/kpc)
        else:
            plfe_et[index,i-1] = 0
    ind_et = np.where(plfe_et[index,:]!=0)
    if index == 0:
        ndens_e.append(xmf_r_lb)
    ndens_e.append(plfe_et[index,:])
    # late
    ind_nan = np.where(mag_r_l_A[:]!=np.nan)
    mag_r_l_A_p = mag_r_l_A[ind_nan]
    r50_l_A_p = r50_l_A[ind_nan]
    ind_l_p = np.digitize(mag_r_l_A_p,mbins_r_lb)
    for i in range(1,np.amax(ind_l_p)+1):
        pos = np.where(np.array(ind_l_p)==i)
        if np.shape(pos)[1] >= 10:
            plfe_lt[index,i-1] = percentiles(0.5,np.log10(r50_l_A_p[pos])) # log10(r*h/kpc)
        else:
            plfe_lt[index,i-1] = 0
    ind_lt = np.where(plfe_lt[index,:]!=0)
    if index == 0:
        ndens_l.append(xmf_r_lb)
    ndens_l.append(plfe_lt[index,:])
    # early fraction
    for i in range(len(mbins_r)):
        if ntot[index,i]>0.:
            nf[index,i] = nf[index,i]/ntot[index,i]
        else:
            nf[index,i] = 0.
            py = nf[index,:] ; ind = np.where(py>0.)
            x = xmf_r[ind] ; y = py[ind]
    ind_f = np.where(nf[index,:] > 0)
    if index == 0:
        ndens_f.append(xmf_r)
    ndens_f.append(nf[index,:])
    
    # HI MF
    ind_HI = np.where(plfe_HI[index,:] > 0)
    plfe_HI[index,ind_HI] = np.log10(plfe_HI[index,ind_HI]/dm_HI/volh)
    if index == 0:
        ndens_HI.append(xmf_HI)
    ndens_HI.append(plfe_HI[index,:])
    
    # Tully-Fisher
    ind_nan = np.where(vc_TF_A[:]!=np.nan)
    vc_TF_A_p = vc_TF_A[ind_nan]
    mag_I_TF_A_p = mag_I_TF_A[ind_nan] 
    ind_TF_p = np.digitize(vc_TF_A_p,mbins_vc)
    for i in range(1,np.amax(ind_TF_p)+1):
        pos = np.where(np.array(ind_TF_p)==i)
        if np.shape(pos)[1] >= 10:
            plfe_TF[index,i-1] = percentiles(0.5,mag_I_TF_A_p[pos]) # m-5logh
        else:
            plfe_TF[index,i-1] = 0
    ind_TF = np.where(plfe_TF[index,:]!=0)
    if index == 0:
        ndens_TF.append(xmf_vc)
    ndens_TF.append(plfe_TF[index,:])
                
    # bulge-BH mass relation
    ind_nan = np.where(mbulge_BH_A[:]!=np.nan)
    mbulge_BH_A_p = mbulge_BH_A[ind_nan]
    mbh_BH_A_p = mbh_BH_A[ind_nan]
    ind_BH_p = np.digitize(mbulge_BH_A_p,mbins_BH)
    for i in range(1,np.amax(ind_BH_p)+1):
        pos = np.where(np.array(ind_BH_p)==i)
        if np.shape(pos)[1] >= 10:
            plfe_BH[index,i-1] = percentiles(0.5,np.log10(mbh_BH_A_p[pos]+1e-10)) # log10(M_BH*h/Msun)
        else:
            plfe_BH[index,i-1] = 0
    ind_BH = np.where(plfe_BH[index,:]!=0)
    if index == 0:
        ndens_BH.append(xmf_BH)
    ndens_BH.append(plfe_BH[index,:])

    # Zstars
    ind_nan = np.where(mag_r_z_A[:]!=np.nan)
    mag_r_z_A_p = mag_r_z_A[ind_nan]
    zstar_z_A_p = zstar_z_A[ind_nan]
    ind_z_p = np.digitize(mag_r_z_A_p,mbins_r_lb)
    for i in range(1,np.amax(ind_z_p)+1):
        pos = np.where(np.array(ind_z_p)==i)
        if np.shape(pos)[1] >= 10:
            plfe_z[index,i-1] = percentiles(0.5,np.log10(zstar_z_A_p[pos]+1e-10)) # log10(zV+1e-10)
        else:
            plfe_z[index,i-1] = 0
    ind_z = np.where(plfe_z[index,:]!=0)
    if index == 0:
        ndens_z.append(xmf_r_lb)
    ndens_z.append(plfe_z[index,:])
        
    # mgas fraction
    #print(max(np.log10(mhhalo_A)))
    #print(np.where((np.log10(mhhalo_A)>14)&(np.log10(mhhalo_A)<14.5)))
    ind_nan = np.where(mhhalo_A[:]!=np.nan)
    mhot_A_p = mhot_A[ind_nan]
    mhhalo_A_p = mhhalo_A[ind_nan]
    ind_g_p = np.digitize(np.log10(mhhalo_A_p),mbins_g) # log10(mhhalo)
    for i in range(1,np.amax(ind_g_p)+1):
        pos = np.where(np.array(ind_g_p)==i)
        if np.shape(pos)[1] >= 10:
            plfe_g[index,i-1] = percentiles(0.5,mhot_A_p[pos]/mhhalo_A_p[pos]) # ratio
        else:
            plfe_g[index,i-1] = 0
    ind_g = np.where(plfe_g[index,:]!=0)
    if index == 0:
        ndens_g.append(xmf_g)
    ndens_g.append(plfe_g[index,:])
            
#    ax00.plot(xmf_K[ind_K],np.transpose(plfe_K[index,ind_K]),c='blue',ls='-')
#    ax01.plot(xmf_r[ind_r],np.transpose(plfe_r[index,ind_r]),c='blue',ls='-')
#    ax02.plot(xmf_r_lb[ind_et],np.transpose(plfe_et[index,ind_et]),c='blue',ls='-')
#    ax10.plot(xmf_r_lb[ind_lt],np.transpose(plfe_lt[index,ind_lt]),c='blue',ls='-')
#    ax11.plot(xmf_HI[ind_HI],np.transpose(plfe_HI[index,ind_HI]),c='blue',ls='-')
#    ax12.plot(xmf_r[ind_f],np.transpose(nf[index,ind_f]),c='blue',ls='-')
#    ax20.plot(xmf_vc[ind_TF],np.transpose(plfe_TF[index,ind_TF]),c='blue',ls='-')
#    ax21.plot(xmf_BH[ind_BH],np.transpose(plfe_BH[index,ind_BH]),c='blue',ls='-')
#    ax22.plot(xmf_r_lb[ind_z],np.transpose(plfe_z[index,ind_z]),c='blue',ls='-')
#    
#    ax1.plot(xmf_K[ind_K_z1],np.transpose(plfe_K_z1[index,ind_K_z1]),c='blue',ls='-')
#    ax2.plot(xmf_g[ind_g],np.transpose(plfe_g[index,ind_g]),c='blue',ls='-')
#        
    index += 1

    
ndens_K = np.array(ndens_K)
ndens_K = np.transpose(ndens_K)
outfil1 = outpath+'KLF_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    np.savetxt(outf1,list(ndens_K))#,fmt=('%.5f'))
    outf1.closed

ndens_K_z1 = np.array(ndens_K_z1)
ndens_K_z1 = np.transpose(ndens_K_z1)
outfil1 = outpath+'KLF_z'+str(z1)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    np.savetxt(outf1,list(ndens_K_z1))#,fmt=('%.5f'))
    outf1.closed

ndens_r = np.array(ndens_r)
ndens_r = np.transpose(ndens_r)
outfil1 = outpath+'rLF_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    np.savetxt(outf1,list(ndens_r))#,fmt=('%.5f'))
    outf1.closed

ndens_e = np.array(ndens_e)
ndens_e = np.transpose(ndens_e)
outfil1 = outpath+'early-t_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    np.savetxt(outf1,list(ndens_e))#,fmt=('%.5f'))
    outf1.closed

ndens_l = np.array(ndens_l)
ndens_l = np.transpose(ndens_l)
outfil1 = outpath+'late-t_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    np.savetxt(outf1,list(ndens_l))#,fmt=('%.5f'))
    outf1.closed

ndens_HI = np.array(ndens_HI)
ndens_HI = np.transpose(ndens_HI)
outfil1 = outpath+'HIMF_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    np.savetxt(outf1,list(ndens_HI))#,fmt=('%.5f'))
    outf1.closed

ndens_f = np.array(ndens_f)
ndens_f = np.transpose(ndens_f)
outfil1 = outpath+'early-f_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    np.savetxt(outf1,list(ndens_f))#,fmt=('%.5f'))
    outf1.closed

ndens_TF = np.array(ndens_TF)
ndens_TF = np.transpose(ndens_TF)
outfil1 = outpath+'TF_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    np.savetxt(outf1,list(ndens_TF))#,fmt=('%.5f'))
    outf1.closed

ndens_BH = np.array(ndens_BH)
ndens_BH = np.transpose(ndens_BH)
outfil1 = outpath+'bulge-BH_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    np.savetxt(outf1,list(ndens_BH))#,fmt=('%.5f'))
    outf1.closed

ndens_z = np.array(ndens_z)
ndens_z = np.transpose(ndens_z)
outfil1 = outpath+'Zstars_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    np.savetxt(outf1,list(ndens_z))#,fmt=('%.5f'))
    outf1.closed

ndens_g = np.array(ndens_g)
ndens_g = np.transpose(ndens_g)
outfil1 = outpath+'mgasf_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    np.savetxt(outf1,list(ndens_g))#,fmt=('%.5f'))
    outf1.closed


#plt.show()
#fig.savefig('calibration_plots'+end+'.png',facecolor='white', transparent=False)
#fig1.savefig('KLF_z'+str(z1)+end+'.png',facecolor='white', transparent=False)
#fig2.savefig('mgasf_z'+str(z)+end+'.png',facecolor='white', transparent=False)

