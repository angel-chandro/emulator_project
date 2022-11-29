from pyDOE import *
import numpy as np
import matplotlib.pylab as plt

# code that generates a Maximin Latin Hypercube for 10000 points
# to sample the whole parameter space after the emulator is trained,
# so that it is possible to study the probability distribution

params = 10 # number of free parameters
runs = 10000 # number of parameter space points

# generate the data
lhd = lhs(params,samples=runs,criterion="maximin")
#lhd = lhs(100,criterion="maximin")
print(np.shape(lhd))
print(lhd)

# save the data
outfile = '/home/chandro/emulator_project/hypercube_'+str(params)+'p_'+str(runs)+'.dat'
#outfile = '/home/chandro/emulator/tfm/input.dat'
p1 = lhd[:,0]*(1.1-0.61)+0.61
p2 = lhd[:,1]+0.2
p3 = lhd[:,2]+0.2
p4 = lhd[:,3]*3+1
p5 = lhd[:,4]*(550-100)+100
p6 = lhd[:,5]*(550-100)+100
p7 = lhd[:,6]*(0.3-0.01)+0.01
p8 = lhd[:,7]*(0.5-0.2)+0.2
p9 = lhd[:,8]*(1.7-0.2)+0.2
p10 = lhd[:,9]*(0.05-0.001)+0.001
tofile = zip(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10)
with open(outfile,'w') as outf:
    outf.write('# f_stab, alpha_cool, alpha_ret, gamma_SN, V_SNdisk, V_SNburst, f_burst, f_ellip, v_SF, f_SMBH\n')
    np.savetxt(outf,list(tofile))
    outf.closed
