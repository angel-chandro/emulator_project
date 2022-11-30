import numpy as np
import matplotlib.pylab as plt
import os
import sys
#np.set_printoptions(threshold=sys.maxsize)

# code that gather together the input and output of the models run for training
# and that shuffles the model order to produce a different emulator
# (in case there are regions not covered by the trained datasets)

outpath = '/home/chandro/galform/elliott/'
end = '_UNIT'

nparam = 10
f_input = '/home/chandro/emulator_project/hypercupe_10p_1000.dat'
f_K_z0 = outpath+'KLF_z0'+end+'.dat'
f_K_z1 = outpath+'KLF_z1'+end+'.dat'
f_r_z0 = outpath+'rLF_z0'+end+'.dat'
f_et_z0 = outpath+'early-t_z0'+end+'.dat'
f_lt_z0 = outpath+'late-t_z0'+end+'.dat'
f_ef_z0 = outpath+'early-f_z0'+end+'.dat'
f_HI_z0 = outpath+'HIMF_z0'+end+'.dat'
f_TF_z0 = outpath+'TF_z0'+end+'.dat'
f_BH_z0 = outpath+'bulge-BH_z0'+end+'.dat'
f_z_z0 = outpath+'Zstars_z0'+end+'.dat'
f_gf_z0 = outpath+'mgasf_z0'+end+'.dat'

# load data
A = np.loadtxt(f_K_z0)
lA = np.shape(A)[0]
B = np.loadtxt(f_K_z1)
lB = np.shape(B)[0]
C = np.loadtxt(f_r_z0)
lC = np.shape(C)[0]
D = np.loadtxt(f_et_z0)
lD = np.shape(D)[0]
E = np.loadtxt(f_lt_z0)
lE = np.shape(E)[0]
F = np.loadtxt(f_ef_z1)
lF = np.shape(F)[0]
G = np.loadtxt(f_HI_z0)
lG = np.shape(G)[0]
H = np.loadtxt(f_TF_z0)
lH = np.shape(H)[0]
I = np.loadtxt(f_BH_z0)
lI = np.shape(I)[0]
J = np.loadtxt(f_z_z1)
lJ = np.shape(J)[0]
K = np.loadtxt(f_gf_z0)
lK = np.shape(K)[0]

input = np.loadtxt(f_INPUT)
input = np.transpose(input)

# gather together data
# not considering the bins ([:,1:] index)
output = np.concatenate((input,A[:,1:],B[:,1:],C[:,1:],D[:,1:],E[:,1:],F[:,1:],G[:,1:],H[:,1:],I[:,1:],J[:,1:],K[:,1:]),axis=0)
output = np.array(output)

# shuffle data
output = np.transpose(output)
np.random.shuffle(output)
output = np.transpose(output)

# save data
# save all data
outfil = '/home/chandro/emulator_project/hypercupe_10p_1000_shuffle.dat'
#with open(f_input, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# f_stab, alpha_cool, alpha_ret, gamma_SN, V_SNdisk, V_SNburst, f_burst, f_ellip, v_SF, f_SMBH\n')
    np.savetxt(outf,list(output))#,fmt=('%.5f'))
    outf.closed
# save only K z0 data
K_shuffle = output[nmodels:nmodels+lA,:]
K_shuffle = np.transpose(K_shuffle)
tofile = zip(A[:,0],K_shuffle) 
outfil = outpath+'KLF_z0'+end+'_shuffle.dat'
#with open(f_K_z0, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# m_K-5*log10(h)[AB]+_midpoint, log10(Phi*Mpc**3*mag/h**3)\n')
    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    outf.closed
# save only K z1 data
K_shuffle = output[nmodels+lA:nmodels+lA+lB,:]
K_shuffle = np.transpose(K_shuffle)
tofile = zip(B[:,0],K_shuffle) 
outfil = outpath+'KLF_z1'+end+'_shuffle.dat'
#with open(f_K_z1, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# m_K-5*log10(h)[AB]+_midpoint, log10(Phi*Mpc**3*mag/h**3)\n')
    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    outf.closed
# save only r z0 data
r_shuffle = output[nmodels+lA+lB:nmodels+lA+lB+lC,:]
r_shuffle = np.transpose(r_shuffle)
tofile = zip(C[:,0],r_shuffle) 
outfil = outpath+'rLF_z0'+end+'_shuffle.dat'
#with open(f_r_z0, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# m_r-5*log10(h)[AB]_midpoint, log10(Phi*Mpc**3*mag/h**3)\n')
    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    outf.closed
# save only early-type z0 data
et_shuffle = output[nmodels+lA+lB+lC:nmodels+lA+lB+lC+lD,:]
et_shuffle = np.transpose(et_shuffle)
tofile = zip(D[:,0],et_shuffle) 
outfil = outpath+'early-t_z0'+end+'_shuffle.dat'
#with open(f_et_z0, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# m_r-5*log10(h)[AB]_midpoint, log10(R50*h/kpc)_perc50\n')
    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    outf.closed
# save only late-type z0 data
lt_shuffle = output[nmodels+lA+lB+lC+lD:nmodels+lA+lB+lC+lD+lE,:]
lt_shuffle = np.transpose(lt_shuffle)
tofile = zip(E[:,0],lt_shuffle) 
outfil = outpath+'late-t_z0'+end+'_shuffle.dat'
#with open(f_lt_z0, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# m_r-5*log10(h)[AB]_midpoint, log10(R50*h/kpc)_perc50\n')
    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    outf.closed
# save only early-fraction z0 data
ef_shuffle = output[nmodels+lA+lB+lC+lD+lE:nmodels+lA+lB+lC+lD+lE+lF,:]
ef_shuffle = np.transpose(ef_shuffle)
tofile = zip(F[:,0],ef_shuffle) 
outfil = outpath+'early-f_z0'+end+'_shuffle.dat'
#with open(f_ef_z0, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# m_r-5*log10(h)[AB]_midpoint, Early fraction\n')
    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    outf.closed
# save only HIMF z0 data
HI_shuffle = output[nmodels+lA+lB+lC+lD+lE+lF:nmodels+lA+lB+lC+lD+lE+lF+lG,:]
HI_shuffle = np.transpose(HI_shuffle)
tofile = zip(G[:,0],HI_shuffle) 
outfil = outpath+'HIMF_z0'+end+'_shuffle.dat'
#with open(f_HI_z0, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# log10(M_HI*h^2/Msun)_midpoint, log10(dn*Mpc**3*M_HI/h**3)\n')
    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    outf.closed
# save only TF z0 data
TF_shuffle = output[nmodels+lA+lB+lC+lD+lE+lF+lG:nmodels+lA+lB+lC+lD+lE+lF+lG+lH,:]
TF_shuffle = np.transpose(TF_shuffle)
tofile = zip(H[:,0],TF_shuffle) 
outfil = outpath+'TF_z0'+end+'_shuffle.dat'
#with open(f_TF_z0, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# log10(V_c*s/km)_midpoint, M_I-5*log10(h)[Vega]_perc50\n')
    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    outf.closed
# save only bulge-BH z0 data
BH_shuffle = output[nmodels+lA+lB+lC+lD+lE+lF+lG+lH:nmodels+lA+lB+lC+lD+lE+lF+lG+lH+lI,:]
BH_shuffle = np.transpose(BH_shuffle)
tofile = zip(I[:,0],BH_shuffle) 
outfil = outpath+'bulge-BH_z0'+end+'_shuffle.dat'
#with open(f_BH_z0, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# log10(M_bulge*h/Msun)_midpoint, log10(M_BH*h/Msun)_perc50\n')
    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    outf.closed
# save only Zstars z0 data
z_shuffle = output[nmodels+lA+lB+lC+lD+lE+lF+lG+lH+lI:nmodels+lA+lB+lC+lD+lE+lF+lG+lH+lI+lJ,:]
z_shuffle = np.transpose(z_shuffle)
tofile = zip(J[:,0],z_shuffle) 
outfil = outpath+'Zstars_z0'+end+'_shuffle.dat'
#with open(f_z_z0, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# m_r-5*log10(h)[AB]_midpoint, log10(Zstar(V-wt))_perc50\n')
    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    outf.closed
# save only gas mass fraction z0 data
gf_shuffle = output[nmodels+lA+lB+lC+lD+lE+lF+lG+lH+lI+lJ:nmodels+lA+lB+lC+lD+lE+lF+lG+lH+lI+lJ+lH,:]
gf_shuffle = np.transpose(gf_shuffle)
tofile = zip(H[:,0],gf_shuffle) 
outfil = outpath+'mgas_z0'+end+'_shuffle.dat'
#with open(f_gf_z0, 'w') as outf: # written mode (not appended)
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# log10(Mhhalo/Msun)_midpoint, Mhotgas/Mhhalo_perc50\n')
    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    outf.closed
