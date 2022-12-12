#! /usr/bin/env python

import numpy as np
import os.path,sys
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
models = ['/neta_simplified/test/UNIT100/gp19.vimal/']
nvol = 64
end = '_UNIT100_test'
z = 0
z1 = 1.1
outpath = '/home/chandro/galform/elliott/'

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

# early and late type
centers_e = []
perc50_e = []
centers_l = []
perc50_l = []

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

# TF
centers_TF = []
perc50_TF = []

# histogram Mbulge (Msun/h)
mlow_BH = 7
mupp_BH = 14
dm_BH = 0.25
mbins_BH = np.arange(mlow_BH,mupp_BH,dm_BH)
xmf_BH = mbins_BH + dm_BH/2.0

# bulge-BH mass relation
centers_BH = []
perc50_BH = []
centers_cent_BH = []
perc50_cent_BH = []
centers_sat_BH = []
perc50_sat_BH = []

# Zstars
centers_z = []
perc50_z = []

# histogram mgas fraction (Msun)
mlow_g = 12
mupp_g = 16
dm_g = 0.2
mbins_g = np.arange(mlow_g,mupp_g,dm_g)
xmf_g = mbins_g + dm_g/2.0

# mgas fraction
centers_g = []
perc50_g = []

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

plfe_K = np.zeros(shape = (len(models),len(mbins_K)))
plfe_K_z1 = np.zeros(shape = (len(models),len(mbins_K)))
plfe_r = np.zeros(shape = (len(models),len(mbins_r)))
nf = np.zeros(shape = (len(models),len(mbins_r)))
ntot = np.zeros(shape = (len(models),len(mbins_r)))
plfe_HI = np.zeros(shape = (len(models),len(mbins_HI)))
print()
vols = np.zeros(shape = (len(models)))

for run in range(len(models)):

    r50_e_A = np.array([])
    mag_r_e_A = np.array([])
    r50_l_A = np.array([])
    mag_r_l_A = np.array([])
    vc_TF_A = np.array([])
    mag_I_TF_A = np.array([])
    mbh_BH_A = np.array([])
    mbulge_BH_A = np.array([])
    mbh_cent_BH_A = np.array([])
    mbulge_cent_BH_A = np.array([])
    mbh_sat_BH_A = np.array([])
    mbulge_sat_BH_A = np.array([])
    mag_r_z_A = np.array([])
    zstar_z_A = np.array([])    
    mhot_A = []
    mhhalo_A = []
            
    print(models[run])
    volh = 0.
    first = True
    for ivol in range(nvol):
        # z=0
        infile = path+models[run]+'/iz128/ivol'+str(ivol)+'/galaxies.hdf5'
        print(infile)
        if (os.path.isfile(infile)):
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
            r50_e_A = np.concatenate([r50_e_A,r50_e])
            mag_r_e_A = np.concatenate([mag_r_e_A,mag_r_e])
            # late type
            ind_l = np.where(btr_r<0.5)
            mag_r_l = mag_r[ind_l] 
            r50_l = r50[ind_l] # kpc/h
            r50_l_A = np.concatenate([r50_l_A,r50_l])
            mag_r_l_A = np.concatenate([mag_r_l_A,mag_r_l])

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
            vc_TF_A = np.concatenate([vc_TF_A,vc_TF])
            mag_I_TF_A = np.concatenate([mag_I_TF_A,mag_I_TF])
            
            # bulge-BH mass
            ind_BH = np.where(btr_r>0.3)
            mbh_BH = mbh[ind_BH]
            mbulge_BH = np.log10(mstars_bulge[ind_BH]+1e-10)  # log10(M_bulge*h/Msun)
            mbh_BH_A = np.concatenate([mbh_BH_A,mbh_BH])
            mbulge_BH_A = np.concatenate([mbulge_BH_A,mbulge_BH])
            cent = np.where((isat==1)&(btr_r>0.3)) # only centrals
            mbh_cent = mbh[cent]
            mstars_bulge_cent = mstars_bulge[cent]
            mbulge_cent_BH = np.log10(mstars_bulge_cent+1e-10)  # log10(M_bulge*h/Msun)
            mbh_cent_BH_A = np.concatenate([mbh_cent_BH_A,mbh_cent])
            mbulge_cent_BH_A = np.concatenate([mbulge_cent_BH_A,mbulge_cent_BH])
            sat = np.where((isat==0)&(btr_r>0.3)) # only satellites
            mbh_sat = mbh[sat]
            mstars_bulge_sat = mstars_bulge[sat]            
            mbulge_sat_BH = np.log10(mstars_bulge_sat+1e-10)  # log10(M_bulge*h/Msun)
            mbh_sat_BH_A = np.concatenate([mbh_sat_BH_A,mbh_sat])
            mbulge_sat_BH_A = np.concatenate([mbulge_sat_BH_A,mbulge_sat_BH])
            
            # Zstars
            #zstar = (zstar_bulge+zstar_disk)/(mstars_bulge+mstars_disk)
            ind_z = np.where((btr_r>0.5)&(mhalo>1e14))
            mag_r_z = mag_r_vega[ind_z]
            zstar_z = zstar[ind_z]
            mag_r_z_A = np.concatenate([mag_r_z_A,mag_r_z])
            zstar_z_A = np.concatenate([zstar_z_A,zstar_z])

            # mgas fraction
            mhot = mhot/h0 # Msun
            mhhalo = mhhalo/h0 # Msun
            ind_type = np.where(isat==1)
            for i in range(len(ind_type[0])):
                if i>0:
                    ind0 = ind_type[0][i-1]
                    ind1 = ind_type[0][i]
                    mhot_A.append(sum(mhot[ind0:ind1]))
                    mhhalo_A.append(mhhalo[ind0])
            
        else:
            print('Not found: ',infile)

        # z=1.1
        infile = path+models[run]+'/iz95/ivol'+str(ivol)+'/galaxies.hdf5'
        print(infile)
        if (os.path.isfile(infile)):
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
    vol = volh/pow(h0,3.) # in Mpc^3

    # K LF 
    ind = np.where(plfe_K[index,:] > 0)
    plfe_K[index,ind] = np.log10(plfe_K[index,ind]/dm_K/volh)
    if index == 0:
        ndens_K.append(xmf_K)
    ndens_K.append(plfe_K[index,:])

    # K LF z=1.1 
    ind = np.where(plfe_K_z1[index,:] > 0)
    plfe_K_z1[index,ind] = np.log10(plfe_K_z1[index,ind]/dm_K/volh)
    if index == 0:
        ndens_K_z1.append(xmf_K)
    ndens_K_z1.append(plfe_K_z1[index,:])

    # r LF
    ind = np.where(plfe_r[index,:] > 0)
    plfe_r[index,ind] = np.log10(plfe_r[index,ind]/dm_r/volh)
    if index == 0:
        ndens_r.append(xmf_r)
    ndens_r.append(plfe_r[index,:])
    
    # early and late type
    # early
    ind_e_p = np.digitize(mag_r_e_A,mbins_r_lb)
    for i in range(1,np.amax(ind_e_p)+1):
        pos = np.where(np.array(ind_e_p)==i)
        if np.shape(pos)[1] >= 10:
            centers_e.append(xmf_r_lb[i-1])
            perc50_e.append(percentiles(0.5,np.log10(r50_e_A[pos]))) # log10(r*h/kpc)
    # late
    ind_l_p = np.digitize(mag_r_l_A,mbins_r_lb)
    for i in range(1,np.amax(ind_l_p)+1):
        pos = np.where(np.array(ind_l_p)==i)
        if np.shape(pos)[1] >= 10:
            centers_l.append(xmf_r_lb[i-1])
            perc50_l.append(percentiles(0.5,np.log10(r50_l_A[pos]))) # log10(r*h/kpc)

    # early fraction
    for i in range(len(mbins_r)):
        if ntot[index,i]>0.:
            nf[index,i] = nf[index,i]/ntot[index,i]
        else:
            nf[index,i] = 0.
    py = nf[index,:] ; ind = np.where(py>0.)
    x = xmf_r[ind] ; y = py[ind]
    if index == 0:
        ndens_f.append(xmf_r)
    ndens_f.append(nf[index,:])

    # HI MF
    ind = np.where(plfe_HI[index,:] > 0)
    plfe_HI[index,ind] = np.log10(plfe_HI[index,ind]/dm_HI/volh)
    if index == 0:
        ndens_HI.append(xmf_HI)
    ndens_HI.append(plfe_HI[index,:])

    # Tully-Fisher
    ind_TF_p = np.digitize(vc_TF_A,mbins_vc)
    for i in range(1,np.amax(ind_TF_p)+1):
        pos = np.where(np.array(ind_TF_p)==i)
        if np.shape(pos)[1] >= 10:
            centers_TF.append(xmf_vc[i-1])
            perc50_TF.append(percentiles(0.5,mag_I_TF_A[pos])) # m-5logh

    # bulge-BH mass relation
    ind_BH_p = np.digitize(mbulge_BH_A,mbins_BH)
    for i in range(1,np.amax(ind_BH_p)+1):
        pos = np.where(np.array(ind_BH_p)==i)
        if np.shape(pos)[1] >= 10:
            centers_BH.append(xmf_BH[i-1])
            perc50_BH.append(percentiles(0.5,np.log10(mbh_BH_A[pos]+1e-10))) # log10(M_BH*h/Msun)
    ind_BH_p = np.digitize(mbulge_cent_BH_A,mbins_BH)
    for i in range(1,np.amax(ind_BH_p)+1):
        pos = np.where(np.array(ind_BH_p)==i)
        if np.shape(pos)[1] >= 10:
            centers_cent_BH.append(xmf_BH[i-1])
            perc50_cent_BH.append(percentiles(0.5,np.log10(mbh_cent_BH_A[pos]+1e-10))) # log10(M_BH*h/Msun)
    ind_BH_p = np.digitize(mbulge_sat_BH_A,mbins_BH)
    for i in range(1,np.amax(ind_BH_p)+1):
        pos = np.where(np.array(ind_BH_p)==i)
        if np.shape(pos)[1] >= 10:
            centers_sat_BH.append(xmf_BH[i-1])
            perc50_sat_BH.append(percentiles(0.5,np.log10(mbh_sat_BH_A[pos]+1e-10))) # log10(M_BH*h/Msun)

    # Zstars
    ind_z_p = np.digitize(mag_r_z_A,mbins_r_lb)
    for i in range(1,np.amax(ind_z_p)+1):
        pos = np.where(np.array(ind_z_p)==i)
        if np.shape(pos)[1] >= 10:
            centers_z.append(xmf_r_lb[i-1])
            perc50_z.append(percentiles(0.5,np.log10(zstar_z_A[pos]+1e-10))) # log10(zV+1e-10)

    # mgas fraction
    mhhalo_A = np.array(mhhalo_A)
    mhot_A = np.array(mhot_A)
    print(max(np.log10(mhhalo_A)))
    print(np.where((np.log10(mhhalo_A)>14)&(np.log10(mhhalo_A)<14.5)))
    ind_g_p = np.digitize(np.log10(mhhalo_A),mbins_g) # log10(mhhalo)
    for i in range(1,np.amax(ind_g_p)+1):
        pos = np.where(np.array(ind_g_p)==i)
        if np.shape(pos)[1] >= 10:
            centers_g.append(xmf_g[i-1])
            perc50_g.append(percentiles(0.5,mhot_A[pos]/mhhalo_A[pos])) # ratio
            
    index += 1

ndens_K = np.array(ndens_K)
ndens_K = np.transpose(ndens_K)
ndens_K_z1 = np.array(ndens_K_z1)
ndens_K_z1 = np.transpose(ndens_K_z1)
ndens_r = np.array(ndens_r)
ndens_r = np.transpose(ndens_r)
ndens_HI = np.array(ndens_HI)
ndens_HI = np.transpose(ndens_HI)
ndens_f = np.array(ndens_f)
ndens_f = np.transpose(ndens_f)

# save K LF
outfil1 = outpath+'KLF_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# m_K-5*log10(h)[AB]+_midpoint, log10(Phi*Mpc**3*mag/h**3)\n')
    np.savetxt(outf1,list(ndens_K))#,fmt=('%.5f'))
    outf1.closed
# save K LF z=1.1
outfil1 = outpath+'KLF_z'+str(z1)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# m_K-5*log10(h)[AB]+_midpoint, log10(Phi*Mpc**3*mag/h**3)\n')
    np.savetxt(outf1,list(ndens_K_z1))#,fmt=('%.5f'))
    outf1.closed

# save r LF
outfil1 = outpath+'rLF_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# m_r-5*log10(h)[AB]_midpoint, log10(Phi*Mpc**3*mag/h**3)\n')
    np.savetxt(outf1,list(ndens_r))#,fmt=('%.5f'))
    outf1.closed

# save early-type
outfil1 = outpath+'early-t_z'+str(z)+end+'.dat'
tofile = zip(centers_e,perc50_e)
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# m_r-5*log10(h)[AB]_midpoint, log10(R50*h/kpc)_perc50\n')
    np.savetxt(outf1,list(tofile))#,fmt=('%.5f'))
    outf1.closed
# save late-type
outfil1 = outpath+'late-t_z'+str(z)+end+'.dat'
tofile = zip(centers_l,perc50_l)
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# m_r-5*log10(h)[AB]_midpoint, log10(R50*h/kpc)_perc50\n')
    np.savetxt(outf1,list(tofile))#,fmt=('%.5f'))
    outf1.closed

# save early fraction
outfil1 = outpath+'early-f_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# m_r-5*log10(h)[AB]_midpoint, Early fraction\n')
    np.savetxt(outf1,list(ndens_f))#,fmt=('%.5f'))
    outf1.closed

# save HI MF
outfil1 = outpath+'HIMF_z'+str(z)+end+'.dat'
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# log10(M_HI*h^2/Msun)_midpoint, log10(dn*Mpc**3*M_HI/h**3)\n')
    np.savetxt(outf1,list(ndens_HI))#,fmt=('%.5f'))
    outf1.closed

# save Tully-Fisher
outfil1 = outpath+'TF_z'+str(z)+end+'.dat'
tofile = zip(centers_TF,perc50_TF)
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# log10(V_c*s/km)_midpoint, M_I-5*log10(h)[Vega]_perc50\n')
    np.savetxt(outf1,list(tofile))#,fmt=('%.5f'))
    outf1.closed

# save bulge-BH mass relation
outfil1 = outpath+'bulge-BH_z'+str(z)+end+'.dat'
tofile = zip(centers_BH,perc50_BH)
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# log10(M_bulge*h/Msun)_midpoint, log10(M_BH*h/Msun)_perc50\n')
    np.savetxt(outf1,list(tofile))#,fmt=('%.5f'))
    outf1.closed
outfil1 = outpath+'bulge-BH_cent_z'+str(z)+end+'.dat'
tofile = zip(centers_cent_BH,perc50_cent_BH)
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# log10(M_bulge*h/Msun)_midpoint, log10(M_BH*h/Msun)_perc50\n')
    np.savetxt(outf1,list(tofile))#,fmt=('%.5f'))
    outf1.closed
outfil1 = outpath+'bulge-BH_sat_z'+str(z)+end+'.dat'
tofile = zip(centers_sat_BH,perc50_sat_BH)
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# log10(M_bulge*h/Msun)_midpoint, log10(M_BH*h/Msun)_perc50\n')
    np.savetxt(outf1,list(tofile))#,fmt=('%.5f'))
    outf1.closed

# save Zstars
outfil1 = outpath+'Zstars_z'+str(z)+end+'.dat'
tofile = zip(centers_z,perc50_z)
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# m_r-5*log10(h)[AB]_midpoint, log10(Zstar(V-wt))_perc50\n')
    np.savetxt(outf1,list(tofile))#,fmt=('%.5f'))
    outf1.closed

# save mgas fraction
outfil1 = outpath+'mgasf_z'+str(z)+end+'.dat'
tofile = zip(centers_g,perc50_g)
with open(outfil1, 'w') as outf1: # written mode (not appended)
    outf1.write('# log10(Mhhalo/Msun)_midpoint, Mhotgas/Mhhalo_perc50\n')
    np.savetxt(outf1,list(tofile))#,fmt=('%.5f'))
    outf1.closed

