import numpy as np
import matplotlib.pylab as plt
import os
import sys
#np.set_printoptions(threshold=sys.maxsize)

# code that gather together the input and output of the models run for training
# and that shuffles the model order to produce a different emulator
# (in case there are regions not covered by the trained datasets)

filename = 'input_pyt.dat'
filename_K = 'output_binsdata_K.dat'
filename_bJ = 'output_binsdata_bJ.dat'
filename_p = 'output_binsdata_p.dat'
filename_s = 'output_binsdata_s.dat'

# load data
A = np.loadtxt(filename_K)
B = np.loadtxt(filename_bJ)
C = np.loadtxt(filename_p)
D = np.loadtxt(filename_s)
input = np.loadtxt(filename)
input = np.transpose(input)

# gather together data
print(np.shape(input))
print(np.shape(D))
output = np.concatenate((input,A[:,3:],B[:,3:],C[:,3:],D[:,3:]),axis=0)
print(np.shape(np.array(output)))
output = np.array(output)

# shuffle data
print(output)
print(np.shape(output))
output = np.transpose(output)
np.random.shuffle(output)
output = np.transpose(output)
print(output)

# save data
# save all data
outfil = 'output_test.dat'
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# alpha_cool, v_SN, F_stab\n')
    np.savetxt(outf,list(output))#,fmt=('%.5f'))
    outf.closed
# save only input data
input_shuffle = output[:3,:]
input_shuffle = np.transpose(input_shuffle)
print(input_shuffle.shape)
outfil = 'input_shuffle.dat'
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# alpha_cool, v_SN, F_stab\n')
    np.savetxt(outf,list(input_shuffle))#,fmt=('%.5f'))
    outf.closed
# save only K data
K_shuffle = output[3:27+3,:]
K_shuffle = np.transpose(K_shuffle)
print(K_shuffle.shape)
outfil = 'output_onlydata_K_shuffle.dat'
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# log10(Phi*Mpc**3*m/h**3)\n')
    np.savetxt(outf,list(K_shuffle))#,fmt=('%.5f'))
    outf.closed
# save only bJ data
bJ_shuffle = output[27+3:27+3+34,:]
bJ_shuffle = np.transpose(bJ_shuffle)
print(bJ_shuffle.shape)
outfil = 'output_onlydata_bJ_shuffle.dat'
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# log10(Phi*Mpc**3*m/h**3)\n')
    np.savetxt(outf,list(bJ_shuffle))#,fmt=('%.5f'))
    outf.closed
# save only p data
p_shuffle = output[3+27+34:3+27+34+19,:]
p_shuffle = np.transpose(p_shuffle)
print(p_shuffle.shape)
outfil = 'output_onlydata_p_shuffle.dat'
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# Passive fraction\n')
    np.savetxt(outf,list(p_shuffle))#,fmt=('%.5f'))
    outf.closed
# save only s data
s_shuffle = output[3+27+34+19:,:]
s_shuffle = np.transpose(s_shuffle)
print(s_shuffle.shape)
outfil = 'output_onlydata_s_shuffle.dat'
with open(outfil, 'w') as outf: # written mode (not appended)
    outf.write('# Passive fraction\n')
    np.savetxt(outf,list(s_shuffle))#,fmt=('%.5f'))
    outf.closed
