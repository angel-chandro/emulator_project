from statistics import median
import numpy as np 
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_style
plt.style.use(mpl_style.style1)
import matplotlib.gridspec as gridspec

# code that plots the different data from 1 single SAM run
# compared to observations
# plots considered: the calibration plots in Elliott

def percentiles(val,data,weights=None):
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
    


z = 0
z1 = 1.1
posf = '_UNIT_models'
posf_d = "_UNIT_250_dispersion_simp"
vol_t = 1000
subvol = 250
nbox_l = int(vol_t/subvol)

# plot 1
fig = plt.figure(constrained_layout=True,figsize=(15,12))
gs = fig.add_gridspec(3, 3)
ax00 = fig.add_subplot(gs[0,0])
ax01 = fig.add_subplot(gs[0,1])
ax02 = fig.add_subplot(gs[0,2])
ax10 = fig.add_subplot(gs[1,0])
ax11 = fig.add_subplot(gs[1,1])
ax12 = fig.add_subplot(gs[1,2])
ax20 = fig.add_subplot(gs[2,0])
ax21 = fig.add_subplot(gs[2,1])
ax22 = fig.add_subplot(gs[2,2])

ax00.set_xlabel("$ M_{K}^{AB} - 5\\rm{log}(\it{h})$")
ax00.set_ylabel("$\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
ax00.set_xlim(-18,-24.5)
ax00.set_ylim(-6.5,-0.5)
ax01.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax01.set_ylabel("$\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
ax01.set_xlim(-18,-24.5)
ax01.set_ylim(-7,-1)
ax02.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax02.set_ylabel('$\\rm{log}(\it{r_{50}}/ \it{h^{-1}}\\rm{kpc} )$')
ax02.set_xlim(-15,-24)
ax02.set_ylim(-1,1.5)
ax10.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax10.set_ylabel('$\\rm{log}(\it{r_{50}}/ \it{h^{-1}}\\rm{kpc} )$')
ax10.set_xlim(-15,-24)
ax10.set_ylim(-1,1.5)
ax11.set_xlabel('$\\rm{log}(\it{M_{HI}}/ \it{h^{-2}}\\rm{M_{\odot}} )$')
ax11.set_ylabel("$\\rm{log}(\\rm{dn/dlog\it{M_{HI}}/ \it{h^{3}} \\rm{Mpc^{-3}}})$")
ax11.set_xlim(8,11)
ax11.set_ylim(-6,0)
ax12.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax12.set_ylabel("f(early type)")
ax12.set_xlim(-15,-24)
ax12.set_ylim(0,1)
ax20.set_xlabel('$\\rm{log}(\it{V_{c}}/\\rm{km s^{-1}})$')
ax20.set_ylabel("$ M_{I}^{Vega} - 5\\rm{log}(\it{h})$")
ax20.set_xlim(1.5,2.65)
ax20.set_ylim(-24,-13)
ax20.invert_yaxis()
ax21.set_xlabel('$\\rm{log}(\it{M_{bulge}}/ \it{h^{-1}}\\rm{M_{\odot}} )$')
ax21.set_ylabel('$\\rm{log}(\it{M_{BH}}/ \it{h^{-1}}\\rm{M_{\odot}} )$')
ax21.set_xlim(8,13)
ax21.set_ylim(5,10)
ax22.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax22.set_ylabel('$\\rm{log}(\it{Z_{star}}(\\rm{V-wt}))$')
ax22.set_xlim(-17,-23)
ax22.set_ylim(-2.8,-1.2)


# plot 2
fig_r = plt.figure(constrained_layout=True,figsize=(15,12))
gs_r = fig_r.add_gridspec(3, 3)
ax00_r = fig_r.add_subplot(gs_r[0,0])
ax01_r = fig_r.add_subplot(gs_r[0,1])
ax02_r = fig_r.add_subplot(gs_r[0,2])
ax10_r = fig_r.add_subplot(gs_r[1,0])
ax11_r = fig_r.add_subplot(gs_r[1,1])
ax12_r = fig_r.add_subplot(gs_r[1,2])
ax20_r = fig_r.add_subplot(gs_r[2,0])
ax21_r = fig_r.add_subplot(gs_r[2,1])
ax22_r = fig_r.add_subplot(gs_r[2,2])

ax00_r.set_xlabel("$ M_{K}^{AB} - 5\\rm{log}(\it{h})$")
ax00_r.set_ylabel("Ratio $\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
ax00_r.set_xlim(-18,-24.5)
ax00_r.set_ylim(0.8,1.2)
ax01_r.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax01_r.set_ylabel("Ratio $\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
ax01_r.set_xlim(-18,-24.5)
ax01_r.set_ylim(0.8,1.2)
ax02_r.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax02_r.set_ylabel('Ratio $\\rm{log}(\it{r_{50}}/ \it{h^{-1}}\\rm{kpc} )$')
ax02_r.set_xlim(-15,-24)
ax02_r.set_ylim(0.8,1.2)
ax10_r.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax10_r.set_ylabel('Ratio $\\rm{log}(\it{r_{50}}/ \it{h^{-1}}\\rm{kpc} )$')
ax10_r.set_xlim(-15,-24)
ax10_r.set_ylim(0.8,1.2)
ax11_r.set_xlabel('$\\rm{log}(\it{M_{HI}}/ \it{h^{-2}}\\rm{M_{\odot}} )$')
ax11_r.set_ylabel("Ratio $\\rm{log}(\\rm{dn/dlog\it{M_{HI}}/ \it{h^{3}} \\rm{Mpc^{-3}}})$")
ax11_r.set_xlim(8,11)
ax11_r.set_ylim(0.8,1.2)
ax12_r.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax12_r.set_ylabel("Ratio f(early type)")
ax12_r.set_xlim(-15,-24)
ax12_r.set_ylim(0.8,1.2)
ax20_r.set_xlabel('$\\rm{log}(\it{V_{c}}/\\rm{km s^{-1}})$')
ax20_r.set_ylabel("Ratio $ M_{I}^{Vega} - 5\\rm{log}(\it{h})$")
ax20_r.set_xlim(1.5,2.65)
ax20_r.set_ylim(0.8,1.2)
ax20_r.invert_yaxis()
ax21_r.set_xlabel('$\\rm{log}(\it{M_{bulge}}/ \it{h^{-1}}\\rm{M_{\odot}} )$')
ax21_r.set_ylabel('Ratio $\\rm{log}(\it{M_{BH}}/ \it{h^{-1}}\\rm{M_{\odot}} )$')
ax21_r.set_xlim(8,13)
ax21_r.set_ylim(0.8,1.2)
ax22_r.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax22_r.set_ylabel('Ratio $\\rm{log}(\it{Z_{star}}(\\rm{V-wt}))$')
ax22_r.set_xlim(-17,-23)
ax22_r.set_ylim(0.8,1.2)


# other plots
fig1 = plt.figure(figsize=(7.8,10.8))
gs = gridspec.GridSpec(5,1)
gs.update(wspace=0., hspace=0.)
axb1 = plt.subplot(gs[-2:,:])
axb1.set_autoscale_on(False) ; axb1.minorticks_on()
axb1.set_xlim(-18,-26) ; axb1.set_ylim(0.8,1.2)
axb1.set_xlabel("$ M_{K}^{AB} - 5\\rm{log}(\it{h})$")
axb1.set_ylabel("Ratio $\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
ax1 = plt.subplot(gs[:-2,:],sharex=axb1)
ax1.set_ylabel("$\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
ax1.set_autoscale_on(False) ; ax1.minorticks_on()
ax1.set_ylim(-6,-1) ; start, end = ax1.get_xlim()
ax1.xaxis.set_ticks(np.arange(-19, -25, 1.))
axb1.yaxis.set_ticks(np.arange(0.8, 1.2, 0.1))
#ax.plot([],[],' ')
#ztext = 'z = '+str(redshift)
#ax.text(13.5,-9, ztext)
plt.setp(ax1.get_xticklabels(), visible=False)

fig2 = plt.figure(figsize=(7.8,10.8))
gs = gridspec.GridSpec(5,1)
gs.update(wspace=0., hspace=0.)
axb2 = plt.subplot(gs[-2:,:])
axb2.set_autoscale_on(False) ; axb2.minorticks_on()
axb2.set_xlim(13,15) ; axb2.set_ylim(0.8,1.2)
axb2.set_xlabel('$\\rm{log}(\it{M_{hhalo}}/ \it{M_{\odot}})$')
axb2.set_ylabel('Ratio $M_{hot,gas}/M_{hhalo}$')
ax2 = plt.subplot(gs[:-2,:],sharex=axb2)
ax2.set_ylabel("$M_{hot,gas}/M_{hhalo}$")
ax2.set_autoscale_on(False) ; ax2.minorticks_on()
ax2.set_ylim(0,0.16) ; start, end = ax2.get_xlim()
ax2.xaxis.set_ticks(np.arange(-13, -15, 1.))
axb2.yaxis.set_ticks(np.arange(0.8, 1.2, 0.1)) 
plt.setp(ax2.get_xticklabels(), visible=False)
 

# loading "true" value (whole volume)
m_K, lf_K = np.loadtxt('KLF_z'+str(z)+posf+'.dat', usecols=(0,1), unpack=True)
m_K_z1, lf_K_z1 = np.loadtxt('KLF_z'+str(z1)+posf+'.dat', usecols=(0,1), unpack=True)
m_r, lf_r = np.loadtxt('rLF_z'+str(z)+posf+'.dat', usecols=(0,1), unpack=True)
m_r_e, r_e = np.loadtxt('early-t_z'+str(z)+posf+'.dat', usecols=(0,1), unpack=True)
m_r_l, r_l = np.loadtxt('late-t_z'+str(z)+posf+'.dat', usecols=(0,1), unpack=True)
M_HI, HImf = np.loadtxt('HIMF_z'+str(z)+posf+'.dat', usecols=(0,1), unpack=True)
m_r_ef, ef = np.loadtxt('early-f_z'+str(z)+posf+'.dat', usecols=(0,1), unpack=True)
vc, m_I = np.loadtxt('TF_z'+str(z)+posf+'.dat', usecols=(0,1), unpack=True)
m_I = m_I
Mbulge, Mbh = np.loadtxt('bulge-BH_z'+str(z)+posf+'.dat', usecols=(0,1), unpack=True)
m_r_z, Zstar = np.loadtxt('Zstars_z'+str(z)+posf+'.dat', usecols=(0,1), unpack=True)
mhhalo_g, ratio_g = np.loadtxt('mgasf_z'+str(z)+posf+'.dat', usecols=(0,1), unpack=True)

# loading dispersion data (subvolumes)
A = np.loadtxt('KLF_z'+str(z)+posf_d+'.dat')
#m_K = A[:,0]
lf_K_m = A[:,1:]
B = np.loadtxt('KLF_z'+str(z1)+posf_d+'.dat')
#m_K_z1 = B[:,0]
lf_K_z1_m = B[:,1:]
C = np.loadtxt('rLF_z'+str(z)+posf_d+'.dat')
#m_r = C[:,0]
lf_r_m = C[:,1:]
D = np.loadtxt('early-t_z'+str(z)+posf_d+'.dat')
#m_r_e_m = D[:,0]
r_e_m = D[:,1:]
E = np.loadtxt('late-t_z'+str(z)+posf_d+'.dat')
#m_r_l_m = E[:,0]
r_l_m = E[:,1:]
F = np.loadtxt('HIMF_z'+str(z)+posf_d+'.dat')
#M_HI = F[:,0]
HImf_m = F[:,1:]
G = np.loadtxt('early-f_z'+str(z)+posf_d+'.dat')
#m_r_ef = G[:,0]
ef_m = G[:,1:]
H = np.loadtxt('TF_z'+str(z)+posf_d+'.dat')
#vc_m = H[:,0]
m_I_m = H[:,1:]
I = np.loadtxt('bulge-BH_z'+str(z)+posf_d+'.dat')
#Mbulge_m = I[:,0]
Mbh_m = I[:,1:]
J = np.loadtxt('Zstars_z'+str(z)+posf_d+'.dat')
#m_r_z_m = J[:,0]
Zstar_m = J[:,1:]
K = np.loadtxt('mgasf_z'+str(z)+posf_d+'.dat')
#mhhalo_g_m = K[:,0]
ratio_g_m = K[:,1:]

# plots
plt.rcParams.update({'font.size': 12})
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

ind_K_r = np.where(lf_K[:] != 0)
ind_K_z1_r = np.where(lf_K_z1[:] != 0)
ind_r_r = np.where(lf_r[:] != 0)
ind_et_r = np.where(r_e[:] > -99999)
ind_lt_r = np.where(r_l[:] > 0)
ind_HI_r = np.where(HImf[:] != 0)
ind_f_r = np.where(ef[:] > 0)
ind_TF_r = np.where(m_I[:] != 0)
ind_BH_r = np.where(Mbh[:] > 0)
ind_z_r = np.where(Zstar[:] != 0)
ind_g_r = np.where(ratio_g[:] != 0)

lf_T = 0

for i in range(nbox_l**3):

    lf_T += lf_K_m[:,i]*250**3  
    
    ind_K = np.where(lf_K_m[:,i] != 0)
    ind_K_i = np.intersect1d(ind_K,ind_K_r)
    ax00.plot(m_K[ind_K],np.transpose(lf_K_m[ind_K,i]),c='red',ls='-')                                                                                                           
    ax00_r.plot(m_K[ind_K_i],np.transpose(lf_K_m[ind_K_i,i])/np.transpose(lf_K[ind_K_i]),c='red',ls='-')
    ind_r = np.where(lf_r_m[:,i] != 0)
    ind_r_i = np.intersect1d(ind_r,ind_r_r)
    ax01.plot(m_r[ind_r],np.transpose(lf_r_m[ind_r,i]),c='red',ls='-')                                                                                                           
    ax01_r.plot(m_r[ind_r_i],np.transpose(lf_r_m[ind_r_i,i])/np.transpose(lf_r[ind_r_i]),c='red',ls='-')
    ind_et = np.where(r_e_m[:,i] > -99999)
    ind_et_i = np.intersect1d(ind_et,ind_et_r)
    ax02.plot(m_r_e[ind_et],np.transpose(r_e_m[ind_et,i]),c='red',ls='-')                                                                                                     
    ax02_r.plot(m_r_e[ind_et_i],np.transpose(r_e_m[ind_et_i,i])/np.transpose(r_e[ind_et_i]),c='red',ls='-')
    ind_lt = np.where(r_l_m[:,i] > 0)
    ind_lt_i = np.intersect1d(ind_lt,ind_lt_r)
    ax10.plot(m_r_l[ind_lt],np.transpose(r_l_m[ind_lt,i]),c='red',ls='-')                                                                                                     
    ax10_r.plot(m_r_l[ind_lt_i],np.transpose(r_l_m[ind_lt_i,i])/np.transpose(r_l[ind_lt_i]),c='red',ls='-')
    ind_HI = np.where(HImf_m[:,i] != 0)
    ind_HI_i = np.intersect1d(ind_HI,ind_HI_r)
    ax11.plot(M_HI[ind_HI],np.transpose(HImf_m[ind_HI,i]),c='red',ls='-')                                                                                                       
    ax11_r.plot(M_HI[ind_HI_i],np.transpose(HImf_m[ind_HI_i,i])/np.transpose(HImf[ind_HI_i]),c='red',ls='-')
    ind_f = np.where(ef_m[:,i] > 0)
    ind_f_i = np.intersect1d(ind_f,ind_f_r)
    ax12.plot(m_r_ef[ind_f],np.transpose(ef_m[ind_f,i]),c='red',ls='-')                                                                                                               
    ax12_r.plot(m_r_ef[ind_f_i],np.transpose(ef_m[ind_f_i,i])/np.transpose(ef[ind_f_i]),c='red',ls='-')
    ind_TF = np.where(m_I_m[:,i] != 0)
    ind_TF_i = np.intersect1d(ind_TF,ind_TF_r)
    ax20.plot(vc[ind_TF],np.transpose(m_I_m[ind_TF,i]),c='red',ls='-')                                                                                                       
    ax20_r.plot(vc[ind_TF_i],np.transpose(m_I_m[ind_TF_i,i])/np.transpose(m_I[ind_TF_i]),c='red',ls='-')
    ind_BH = np.where(Mbh_m[:,i] > 0)
    ind_BH_i = np.intersect1d(ind_BH,ind_BH_r)
    ax21.plot(Mbulge[ind_BH],np.transpose(Mbh_m[ind_BH,i]),c='red',ls='-')                                                                                                       
    ax21_r.plot(Mbulge[ind_BH_i],np.transpose(Mbh_m[ind_BH_i,i])/np.transpose(Mbh[ind_BH_i]),c='red',ls='-')
    ind_z = np.where(Zstar_m[:,i] != 0)
    ind_z_i = np.intersect1d(ind_z,ind_z_r)
    ax22.plot(m_r_z[ind_z],np.transpose(Zstar_m[ind_z,i]),c='red',ls='-')
    ax22_r.plot(m_r_z[ind_z_i],np.transpose(Zstar_m[ind_z_i,i])/np.transpose(Zstar[ind_z_i]),c='red',ls='-')

    ind_K_z1 = np.where(lf_K_z1_m[:,i] != 0)
    ind_K_z1_i = np.intersect1d(ind_K_z1,ind_K_z1_r)
    ax1.plot(m_K[ind_K_z1],np.transpose(lf_K_z1_m[ind_K_z1,i]),c='red',ls='-')                                                                                                         
    axb1.plot(m_K[ind_K_z1_i],np.transpose(lf_K_z1_m[ind_K_z1_i,i])/np.transpose(lf_K_z1[ind_K_z1_i]),c='red',ls='-') 
    ind_g = np.where(ratio_g_m[:,i] != 0)
    ind_g_i = np.intersect1d(ind_g,ind_g_r)
    ax2.plot(mhhalo_g[ind_g],np.transpose(ratio_g_m[ind_g,i]),c='red',ls='-')
    axb2.plot(mhhalo_g[ind_g_i],np.transpose(ratio_g_m[ind_g_i,i])/np.transpose(ratio_g[ind_g_i]),c='red',ls='-') 


print(lf_T)
print(lf_K*1000**3)
ax00.plot(m_K[ind_K_r],np.transpose(lf_K[ind_K_r]),c='black',ls='-')                                                                                                           
ax01.plot(m_r[ind_r_r],np.transpose(lf_r[ind_r_r]),c='black',ls='-')                                                                                                           
ax02.plot(m_r_e[ind_et_r],np.transpose(r_e[ind_et_r]),c='black',ls='-')                                                                                                     
ax10.plot(m_r_l[ind_lt_r],np.transpose(r_l[ind_lt_r]),c='black',ls='-')                                                                                                     
ax11.plot(M_HI[ind_HI_r],np.transpose(HImf[ind_HI_r]),c='black',ls='-')                                                                                                       
ax12.plot(m_r_ef[ind_f_r],np.transpose(ef[ind_f_r]),c='black',ls='-')                                                                                                               
ax20.plot(vc[ind_TF_r],np.transpose(m_I[ind_TF_r]),c='black',ls='-')                                                                                                       
ax21.plot(Mbulge[ind_BH_r],np.transpose(Mbh[ind_BH_r]),c='black',ls='-')                                                                                                       
ax22.plot(m_r_z[ind_z_r],np.transpose(Zstar[ind_z_r]),c='black',ls='-')

ax1.plot(m_K[ind_K_z1_r],np.transpose(lf_K_z1[ind_K_z1_r]),c='black',ls='-')
ax2.plot(mhhalo_g[ind_g_r],np.transpose(ratio_g[ind_g_r]),c='black',ls='-')

ax00_r.axhline(y=1,color='black',ls='-')                                                                                                           
ax01_r.axhline(y=1,color='black',ls='-')                                                                                                           
ax02_r.axhline(y=1,color='black',ls='-')                                                                                                           
ax10_r.axhline(y=1,color='black',ls='-')                                                                                                           
ax11_r.axhline(y=1,color='black',ls='-')                                                                                                           
ax12_r.axhline(y=1,color='black',ls='-')                                                                                                           
ax20_r.axhline(y=1,color='black',ls='-')                                                                                                           
ax21_r.axhline(y=1,color='black',ls='-')                                                                                                           
ax22_r.axhline(y=1,color='black',ls='-')                                                                                                           
axb1.axhline(y=1,color='black',ls='-')                                                                                                           
axb2.axhline(y=1,color='black',ls='-')                                                                                                           

#plt.show()
fig.savefig('calibration_plots_UNIT_250_dispersion_simp_ratio1.png',facecolor='white', transparent=False)
fig_r.savefig('calibration_plots_UNIT_250_dispersion_simp_ratio2.png',facecolor='white', transparent=False)
fig1.savefig('KLF_z1_UNIT_250_dispersion_simp_ratio.png',facecolor='white', transparent=False)
fig2.savefig('mgasf_z0_UNIT_250_dispersion_simp_ratio.png',facecolor='white', transparent=False)

# K LF
# Driver+2012
obs_K = '/home/chandro/galform/Obs_Data2/lfk_z0_driver12.data'
mag, den, err, num = np.loadtxt(obs_K,unpack=True)
# convert M_K from AB to Vega mags
#mag = mag - 1.87
# convert phi from no. per 0.5 mag to dn/dMag
den = den*2
err = err*2
ind = np.where(lf_K!=0)
m_K_f = m_K[ind]
lf_K_f = lf_K[ind]
ind = np.where(den > 0.)
error_K = err[ind]
x_K = mag[ind]
y_K = np.log10(den[ind])
eh_K = np.log10(den[ind]+err[ind]) - np.log10(den[ind])
el_K = np.log10(den[ind]) - np.log10(den[ind]-err[ind])
# Kochanek+2001
obs_K = '/home/chandro/galform/Obs_Data/2MASS.dat'
mag, den, err = np.loadtxt(obs_K,unpack=True)
mag = mag + 1.87
ind = np.where(den > -99990.)
e2_K = err[ind]
x2_K = mag[ind]
y2_K = den[ind]

ax00 = fig.add_subplot(gs[0,0])
ax00.plot(m_K_f,lf_K_f,c='blue',ls='-')    
ax00.errorbar(x_K, y_K, yerr=[el_K,eh_K], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Driver+2012")#label="GAMA, Driver+2012")
ax00.errorbar(x2_K, y2_K, yerr=e2_K, ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='s',zorder=200,label="Kochanek+2001")#label="2MASS, Kochanek+2001")
ax00.set_xlabel("$ M_{K}^{AB} - 5\\rm{log}(\it{h})$")
ax00.set_ylabel("$\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
ax00.set_xlim(-18,-24.5)
ax00.set_ylim(-6.5,-0.5)
# Put a legend to the right of the current axis
ax00.legend(loc='lower left')#, bbox_to_anchor=(1, 0.5))



# r LF

# Driver+2012
obs_r = '/home/chandro/galform/Obs_Data2/lfr_z0_driver12.data'
mag, den, err, num = np.loadtxt(obs_r,unpack=True)
# convert phi from no. per 0.5 mag to dn/dMag
den = den*2
err = err*2
ind = np.where(lf_r!=0)
m_r_f = m_r[ind]
lf_r_f = lf_r[ind]
ind = np.where(den > 0.)
error_r = err[ind]
x_r = mag[ind]
y_r = np.log10(den[ind])
eh_r = np.log10(den[ind]+err[ind]) - np.log10(den[ind])
el_r = np.log10(den[ind]) - np.log10(den[ind]-err[ind])

ax01 = fig.add_subplot(gs[0,1])
ax01.plot(m_r_f,lf_r_f,c='blue',ls='-')    
ax01.errorbar(x_r, y_r, yerr=[el_r,eh_r], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Driver+2012")#label="GAMA, Driver+2012")
ax01.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax01.set_ylabel("$\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
ax01.set_xlim(-18,-24.5)
ax01.set_ylim(-7,-1)
# Put a legend to the right of the current axis
ax01.legend(loc='lower left')#, bbox_to_anchor=(1, 0.5))


# Early type
# Shen+2003
obs_e = '/home/chandro/galform/Obs_Data2/sizes_sersic_shen03.data'
ml, mu, lnr50, sigma = np.loadtxt(obs_e,unpack=True,usecols=[0,1,2,3])
hobs = 0.7
# compute mean mag in each bin
mag = 0.5*(ml+mu)
# convert from H0=70 to H0=100
mag = mag - 5*np.log10(hobs)
# convert r50 to kpc/h
r50 = np.exp(lnr50)
r50 = r50*hobs
# for late-type galaxies, correct sizes up by factor 1.34
# to account for fact that Shen measurement is in circular apertures
# median correction factor for randomnly-inclined thin exp disks is 1.34
r50corr = 1.34*r50
# convert sigma from sigma(ln(R)) to sigma(log(R))
sigma_lgR = sigma/np.log(10)
r50_e = r50[:10]
sigma_lgR_e = sigma_lgR[:10]
mag_e = mag[:10]
lgR_e = np.log10(r50_e)
ind = np.where(mag_e > -99999.)
e_e = sigma_lgR_e[ind]
x_e = mag_e[ind]
y_e = lgR_e[ind]

ax02 = fig.add_subplot(gs[0,2])
ax02.plot(m_r_e,r_e,c='blue',ls='-')    
ax02.errorbar(x_e, y_e, yerr=e_e, ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Shen+2003")#label="SDSS, Shen+2003")
ax02.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax02.set_ylabel('$\\rm{log}(\it{r_{50}}/ \it{h^{-1}}\\rm{kpc} )$')
ax02.set_xlim(-15,-24)
ax02.set_ylim(-1,1.5)
# Put a legend to the right of the current axis
ax02.legend(loc='upper left')#, bbox_to_anchor=(1, 0.5))


# late type
r50_l = r50corr[10:]
sigma_lgR_l = sigma_lgR[10:]
mag_l = mag[10:]
lgR_l = np.log10(r50_l)
ind = np.where(mag_l > -99990.)
e_l = sigma_lgR_l[ind]
x_l = mag_l[ind]
y_l = lgR_l[ind]

ax10 = fig.add_subplot(gs[1,0])
ax10.plot(m_r_l,r_l,c='blue',ls='-')    
ax10.errorbar(x_l, y_l, yerr=e_l, ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Shen+2003")#label="SDSS, Shen+2003")
ax10.legend(loc='lower left')
ax10.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax10.set_ylabel('$\\rm{log}(\it{r_{50}}/ \it{h^{-1}}\\rm{kpc} )$')
ax10.set_xlim(-15,-24)
ax10.set_ylim(-1,1.5)


# HIMF
# Zwaan+2005
obs_HI = '/home/chandro/galform/Obs_Data2/HIMF_Zwaan2005.data'
lgMHI, lgdndlgMHI, errdn, errup = np.loadtxt(obs_HI,usecols=[0,1,2,3],unpack=True)
hobs = 0.75
# convert from HI masses to total gas masses, assuming constant H2/HI ratio
lgMcold = lgMHI
# convert MF from dn/dlog10(M) to dn/dln(M)
lgdndlnMcold = lgdndlgMHI - np.log10(np.log(10))
# finally, convert to H0=100 units
lgMcold = lgMcold + np.log10(hobs**2)
lgdndlnMcold = lgdndlnMcold - np.log10(hobs**3)
ind = np.where(lgdndlnMcold > -99990.)
x_HI = lgMcold[ind]
y_HI = lgdndlnMcold[ind]
eu_HI = errup[ind]
el_HI = errdn[ind]
# Martin+2010
obs_HI = '/home/chandro/galform/Obs_Data2/HIMF_Martin2010.data'
lgMHI, lgdndlgMHI, err = np.loadtxt(obs_HI,usecols=[0,1,2],unpack=True)
hobs = 0.75
lgMHI2 = lgMHI[24:]
lgdndlgMHI2 = lgdndlgMHI[24:]
err2 = err[24:]
# convert from HI masses to total gas masses, assuming constant H2/HI ratio
lgMcold = lgMHI2
# convert MF from dn/dlog10(M) to dn/dln(M)
lgdndlnMcold = lgdndlgMHI2 - np.log10(np.log(10))
# finally, convert to H0=100 units
lgMcold = lgMcold + np.log10(hobs**2)
lgdndlnMcold = lgdndlnMcold - np.log10(hobs**3)
ind = np.where(lgdndlnMcold > -99999.)
x2_HI = lgMcold[ind]
y2_HI = lgdndlnMcold[ind]
e2_HI = err2[ind]
ind = np.where(HImf!=0)
HImf_f = HImf[ind]
M_HI_f = M_HI[ind]

ax11 = fig.add_subplot(gs[1,1])
ax11.plot(M_HI_f,HImf_f,c='blue',ls='-')    
ax11.errorbar(x_HI, y_HI, yerr=[el_HI,eu_HI], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Zwann+2005") #label="HIPASS, Zwaan+2005")
ax11.errorbar(x2_HI, y2_HI, yerr=e2_HI, ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='s',zorder=200,label="Martin+2010")#label="ALFALFA, Martin+2010")
ax11.set_xlabel('$\\rm{log}(\it{M_{HI}}/ \it{h^{-2}}\\rm{M_{\odot}} )$')
ax11.set_ylabel("$\\rm{log}(\\rm{dn/dlog\it{M_{HI}}/ \it{h^{3}} \\rm{Mpc^{-3}}})$")
ax11.set_xlim(8,11)
ax11.set_ylim(-6,0)
# Put a legend to the right of the current axis
ax11.legend(loc='lower left')#, bbox_to_anchor=(1, 0.5))
 
# Early fraction
# Moffett+2016
obs_ef = '/home/chandro/Moffett+2016/ETfracvsmr.tab'
mr, frac_lo, frac, frac_up = np.loadtxt(obs_ef,unpack=True)
ind = np.where(frac > -999990.)
x_ef = mr[ind]
y_ef = frac[ind]
eh_ef = frac_up[ind] - frac[ind]
el_ef = frac[ind] - frac_lo[ind]
# Gonzalez+2009?
obs_ef = '/home/chandro/galform/Obs_Data2/morph_frac_conc_SDSS.data'
mag, frac_corr = np.loadtxt(obs_ef,unpack=True,usecols=(0,1))
ind = np.where(frac_corr > -999990.)
x2_ef = mag[ind]
y2_ef = frac_corr[ind]
ind = np.where(ef > 0)
ef_f = ef[ind]
m_r_ef_f = m_r_ef[ind]

ax12 = fig.add_subplot(gs[1,2])
ax12.plot(m_r_ef_f,ef_f,c='blue',ls='-')    
ax12.errorbar(x_ef, y_ef, yerr=[eh_ef,el_ef], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Moffett+2016")
ax12.errorbar(x2_ef, y2_ef, ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='s',zorder=200,label="Gonzalez+2009")
ax12.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax12.set_ylabel("f(early type)")
ax12.set_xlim(-15,-24)
ax12.set_ylim(0,1)
# Put a legend to the right of the current axis
ax12.legend(loc='upper left')#, bbox_to_anchor=(1, 0.5))


# Tully-Fisher
# Mathewson+1998
obs_TF = '/home/chandro/galform/Obs_Data2/math_disks_dhub.data'
mI, hD, Vrot, Vmax = np.loadtxt(obs_TF,unpack=True)
Vrot = np.log10(Vrot)
vbin = np.linspace(1.7,2.52,10)
centers = (vbin[0:-1]+vbin[1:])/2
print(centers)
ind = np.digitize(Vrot,vbin)
cent = []
p50 = []
p16 = []
p84 = []
for i in range(1,np.amax(ind)+1):
    pos = np.where(np.array(ind)==i)
    if np.shape(pos)[1] >= 10:
        cent.append(centers[i-1])
        p50.append(percentiles(0.5,mI[pos]))
        p16.append(percentiles(0.16,mI[pos]))
        p84.append(percentiles(0.84,mI[pos]))
cent = np.array(cent)
p50 = np.array(p50)
p16 = np.array(p16)
p84 = np.array(p84)
ind = np.where(p50 > -999990.)
x_TF = cent[ind]
y_TF = p50[ind]
el_TF = p16[ind]
eu_TF = p84[ind]

ax20 = fig.add_subplot(gs[2,0])
ax20.plot(vc,m_I,c='blue',ls='-')    
ax20.plot(Vrot,mI,'.',c='grey')    
ax20.errorbar(x_TF, y_TF, yerr=[abs(el_TF-y_TF),abs(eu_TF-y_TF)], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Mathewson+1998")
ax20.legend(loc='lower right')
ax20.set_xlabel('$\\rm{log}(\it{V_{c}}/\\rm{km s^{-1}})$')
ax20.set_ylabel("$ M_{I}^{Vega} - 5\\rm{log}(\it{h})$")
ax20.set_xlim(1.5,2.65)
ax20.set_ylim(-24,-13)
ax20.invert_yaxis()


# bulge-BH mass relation
# Haering+2004
obs_BH = '/home/chandro/galform/Obs_Data2/MBH_Mbulge_HR04.data'
mbh, err_up, err_dn, sigma, mbulge = np.loadtxt(obs_BH,unpack=True)
hobs = 0.7
err_lgmbulge = 0.18
lgmbh = np.log10(mbh)
lgerr_up = np.log10(1+err_up/mbh)
lgerr_dn = -np.log10(1-err_dn/mbh)
lgmbulge = np.log10(mbulge)
lgerr_across = 0*np.log10(mbulge) + err_lgmbulge
# convert from H0=70 to H0=100
lgmbh = lgmbh + np.log10(hobs)
lgmbulge = lgmbulge + np.log10(hobs)
# plot Haering & Rix best power-law fit
lgmbulge_fit = np.array([8,13])
lgmbh_fit = 8.20 + 1.12*(lgmbulge_fit-11)
# convert from H0=70 to H0=100
lgmbh_fit = lgmbh_fit + np.log10(hobs)
lgmbulge_fit = lgmbulge_fit + np.log10(hobs)
#ind = np.where(lgmbulge > -999990.)
#x_BH = lgmbulge[ind]
#y_BH = lgmbh[ind]
#el_BH = lgerr_dn[ind]
#eu_BH = lgerr_up[ind]
#ex_BH = lgerr_across[ind]
mbulgebin = np.linspace(9.55,11.9,6)
centers = (mbulgebin[0:-1]+mbulgebin[1:])/2
print(centers)
ind = np.digitize(lgmbulge,mbulgebin)
cent = []
p50 = []
p16 = []
p84 = []
for i in range(1,np.amax(ind)+1):
    pos = np.where(np.array(ind)==i)
    if np.shape(pos)[1] >= 3:
        cent.append(centers[i-1])
        p50.append(percentiles(0.5,lgmbh[pos]))
        p16.append(percentiles(0.16,lgmbh[pos]))
        p84.append(percentiles(0.84,lgmbh[pos]))
cent = np.array(cent)
p50 = np.array(p50)
p16 = np.array(p16)
p84 = np.array(p84)
ind = np.where(p50 > -999990.)
x_BH = cent[ind]
y_BH = p50[ind]
el_BH = p16[ind]
eu_BH = p84[ind]

ax21 = fig.add_subplot(gs[2,1])
ax21.plot(Mbulge,Mbh,c='blue',ls='-')    
#ax21.plot(Mbulge_cent,Mbh_cent,c='blue',ls='--')    
#ax21.plot(Mbulge_sat,Mbh_sat,c='blue',ls=':')    
ax21.errorbar(lgmbulge, lgmbh, xerr=lgerr_across, yerr=[lgerr_dn,lgerr_up], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='s',zorder=200)#,label="Haering+2004")
ax21.errorbar(x_BH, y_BH, yerr=[abs(el_BH-y_BH),abs(eu_BH-y_BH)], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Haering+2004")
ax21.legend(loc='lower right')
ax21.set_xlabel('$\\rm{log}(\it{M_{bulge}}/ \it{h^{-1}}\\rm{M_{\odot}} )$')
ax21.set_ylabel('$\\rm{log}(\it{M_{BH}}/ \it{h^{-1}}\\rm{M_{\odot}} )$')
ax21.set_xlim(8,13)
ax21.set_ylim(5,10)

#plt.rcParams.update({'font.size': 16})
#fig = plt.figure(9,figsize=(9.6,7.2))
#plt.plot(Mbulge,Mbh,c='blue',ls='-',label='All')    
#plt.plot(Mbulge_cent,Mbh_cent,c='blue',ls='--',label='Central')    
#plt.plot(Mbulge_sat,Mbh_sat,c='blue',ls=':',label='Satellite')    
#plt.errorbar(x_BH, y_BH, xerr=ex_BH, yerr=[el_BH,eu_BH], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='s',zorder=200,label="Haering+2004")
##ax21.errorbar(x_BH, y_BH, xerr=ex_BH, yerr=[el_BH,eu_BH], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Haering+2004")
#plt.legend(loc='lower right')
#plt.xlabel('$\\rm{log}(\it{M_{bulge}}/ \it{h^{-1}}\\rm{M_{\odot}} )$')
#plt.ylabel('$\\rm{log}(\it{M_{BH}}/ \it{h^{-1}}\\rm{M_{\odot}} )$')
#plt.xlim(8,13)
#plt.ylim(5,10)
#plt.show()

# Early metallicity
# Smith+2008
obs_z = '/home/chandro/galform/Obs_Data2/Zstar_Mr_Smith08.data'
lgsig, err_lgsig, lglumr, lgage, err_lgage, lgmet, err_lgmet, re_arcsec = np.loadtxt(obs_z,unpack=True,usecols=[1,2,3,4,5,6,7,10])
hobs = 0.7
# if no sigma measured, replace by 0
ind = np.where(lgsig<0)
lgsig[ind] = 0  
# convert from lumr = log(L_R/L_Rsun) to M_R, using assumed M_Rsun=4.42
magR_sun = 4.42
magR = magR_sun - 2.5*lglumr
# convert M_R from H0=70 to M_R-5logh
magR = magR - 5*np.log10(hobs)
# convert re from arcsec to kpc, using 1 arcmin = 57 kpc at adopted distance
ind = np.where(re_arcsec>0)
re = np.copy(re_arcsec)
re[ind] = (57/60)*re_arcsec[ind]
# convert to kpc/h
re = hobs*re
# convert from [Z/H] to Z
lgzstar = lgmet + np.log10(0.0173)
# correct metallicities to global values based on assumed constant gradient
# do not have measured re for all galaxies, so for missing values, use
# least-squares linear fit to log(re) vs M_R for same data
# log(re/ kpc/h) = -0.150*log(M_R-5logh+20) + 0.245
re_fit = 10**(-0.150*(magR+20) + 0.245)
# convert back to arcsec
re_fit_arcsec = re_fit/hobs * (60/57)
rap_arcsec = 1 # aperture size in arcec = fibre radius
rap_re = np.copy(rap_arcsec/re_fit_arcsec)
rap_re[ind] = rap_arcsec/re_arcsec[ind]
lgrap_re = np.log10(rap_re)
# calc median rap/re
# measured re
#ind = np.where(re_arcsec>0)
#rfac = rap_re[ind]
#xmed = median(rfac)
# re from fit
#ind = np.where(re_arcsec<0)
#rfac = rap_re[ind]
#xmed = median(rfac)
# read data on aperture correction factor from file
# this gives Z(<r)/Ze (lum-wtd) as fn of r/re
file = "/home/chandro/galform/SM/Zcorr/Zcorrection_alpha0.15.data"
r, zav = np.loadtxt(file,unpack=True,usecols=[0,1])
lgr = np.log10(r)
lgzav = np.log10(zav)
# interpolate to get Z(<rap)/Ze
fit = np.polyfit(lgr,lgzav,1)
lgzap = fit[0]*lgrap_re + fit[1]
zap = 10**lgzap
# also compute Z(<rmax)/Ze, where rmax is max radius tabulated
n = len(r)
zmax = zav[n-1]
# aperture correction factor Z(global)/Z(aperture) is ratio of these
zcorrfac = zmax/zap
# convert to dex
lgzcorrfac = np.log10(zcorrfac)
# calc median correction factor
# measured re
#lgzfac = lgzcorrfac[ind]
#xmed = median(lgzfac)
# re from fit
#ind = np.where(re_arcsec<0)
#lgzfac = lgzcorrfac[ind]
#xmed = median(lgzfac)
# combined sample
lgzfac = lgzcorrfac
#lgzcorrfac_med = xmed
# apply correction for metallicity gradients to get global metallicity
lgzstar_corr = lgzstar + lgzcorrfac
mr = np.linspace(-25.5,-15.5,11)
print(mr)
centers = (mr[0:-1]+mr[1:])/2
print(centers)
ind = np.digitize(magR,mr)
cent = []
p50 = []
p16 = []
p84 = []
for i in range(1,np.amax(ind)+1):
    pos = np.where(np.array(ind)==i)
    print(centers[i-1])
    print(lgzstar_corr[pos])
    print(len(lgzstar_corr[pos]))
    if np.shape(pos)[1] >= 10:
        cent.append(centers[i-1])
        p50.append(percentiles(0.5,lgzstar_corr[pos]))
        p16.append(percentiles(0.16,lgzstar_corr[pos]))
        p84.append(percentiles(0.84,lgzstar_corr[pos]))
cent = np.array(cent)
p50 = np.array(p50)
p16 = np.array(p16)
p84 = np.array(p84)
ind = np.where(p50 > -999990.)
x_z = cent[ind]
y_z = p50[ind]
el_z = p16[ind]
eu_z = p84[ind]

ax22 = fig.add_subplot(gs[2,2])
ax22.plot(m_r_z,Zstar,c='blue',ls='-')    
ax22.plot(magR,lgzstar_corr,'.',c='grey')    
ax22.errorbar(x_z, y_z, yerr=[abs(el_z-y_z),abs(eu_z-y_z)], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Smith+2008")
ax22.legend(loc='lower right')
ax22.set_xlabel("$ M_{r}^{AB} - 5\\rm{log}(\it{h})$")
ax22.set_ylabel('$\\rm{log}(\it{Z_{star}}(\\rm{V-wt}))$')
ax22.set_xlim(-17,-23)
ax22.set_ylim(-2.8,-1.2)

#plt.show()
plt.savefig('calibration_plots'+posf+'.png',facecolor='white', transparent=False)


plt.rcParams.update({'font.size': 16})


# K LF z=1.1
# Beare+2019
obs_K_z1 = '/home/chandro/galform/elliott/lfK_z1.1_beare2019.txt'
mag_l, mag_u, den, err = np.loadtxt(obs_K_z1,unpack=True)
hobs = 0.7
# compute mean mag in each bin
mag = 0.5*(mag_l+mag_u)
# convert from H0=70 to H0=100
mag = mag - 5*np.log10(hobs)
den = den*1e-3/hobs**3
err = err*1e-3/hobs**3
ind = np.where(den > 0.)
e_K_z1 = err[ind]
x_K_z1 = mag[ind]
y_K_z1 = np.log10(den[ind])
ind = np.where(lf_K_z1!=0)
m_K_z1_f = m_K_z1[ind]
lf_K_z1_f = lf_K_z1[ind]

fig = plt.figure(10,figsize=(9.6,7.2))
plt.plot(m_K_z1_f,lf_K_z1_f,c='blue',ls='-')    
plt.errorbar(x_K_z1, y_K_z1, yerr=e_K_z1, ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Beare+2019")
plt.xlabel("$ M_{K}^{AB} - 5\\rm{log}(\it{h})$")
plt.ylabel("$\\rm{log}(\Phi/ \it{h^{3}} \\rm{Mpc^{-3} mag^{-1}})$")
plt.xlim(-18,-26)
plt.ylim(-6,-1)
plt.text(-24,-2,'z=1.1',fontsize=25)
# Put a legend to the right of the current axis
plt.legend(loc='lower left')#, bbox_to_anchor=(1, 0.5))
#plt.show()
plt.savefig('KLF_z'+str(z1)+posf+'.png',facecolor='white', transparent=False)


# mgas fraction
# observations
obs_g = '/home/chandro/Obs_Data/calibration/all_fgas.txt'
m500, dm500l, dm500u, fgas500, dfgas500u, dfgas500l = np.loadtxt(obs_g,unpack=True,usecols=(0,1,2,3,4,5))
hobs = 0.7
m500 = m500*1e13#/hobs # Msun/h100
dm500u = dm500u*1e13#/hobs # Msun/h100
dm500l = dm500l*1e13#/hobs # Msun/h100
fgas500 = fgas500#/hobs**1.5 # h100
dfgas500u = dfgas500u#/hobs**1.5 # h100
dfgas500l = dfgas500l#/hobs**1.5 # h100
ind = np.where(fgas500 > -9999.)
x_g = np.log10(m500[ind])
#exh_g = np.log10(m500[ind]+dm500u[ind]) - np.log10(m500[ind])
#exl_g = np.log10(m500[ind]) - np.log10(m500[ind]-dm500l[ind])
y_g = fgas500[ind]
eh_g = dfgas500u[ind]
el_g = dfgas500l[ind]

fig = plt.figure(11,figsize=(9.6,7.2))
plt.plot(mhhalo_g,ratio_g,c='blue',ls='-')    
plt.errorbar(x_g[:23], y_g[:23], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Sun+2009")
plt.errorbar(x_g[23:33], y_g[23:33], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Vikhlinin+2006")
plt.errorbar(x_g[33:64], y_g[33:64], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Pratt+2009")
plt.errorbar(x_g[64:128], y_g[64:128], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Lin+2012")
#plt.errorbar(x_g[128:165], y_g[128:165], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Maughan+2008")
plt.errorbar(x_g[165:185], y_g[165:185], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Lovisari+2015")
#plt.errorbar(x_g[185:200], y_g[185:200], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Rassmussen+2009")
#plt.errorbar(x_g[200:208], y_g[200:208], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Pearson+2017")
#plt.errorbar(x_g[208:213], y_g[208:213], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Sanderson+2013")
#plt.errorbar(x_g[213:], y_g[213:], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="Gonzalez+2013")
#plt.errorbar(x_g, y_g, xerr=[exl_g,exh_g], yerr=[el_g,eh_g], ls='None', mfc='None', ecolor = 'black', mec='black',marker='s',zorder=200,label="observations")
plt.legend(loc='lower right')
plt.xlabel('$\\rm{log}(\it{M_{hhalo}}/ \it{M_{\odot}})$')
plt.ylabel('$M_{hot,gas}/M_{hhalo}$')
plt.xlim(13,15)
plt.ylim(0,0.16)
#plt.show()
plt.savefig('mgasf_z'+str(z)+posf+'.png',facecolor='white', transparent=False)
