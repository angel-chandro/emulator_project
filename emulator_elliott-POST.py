import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

model = tf.keras.models.load_model('model_saved')


import numpy as np 
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_style
plt.style.use(mpl_style.style1)

def cut_plot(lbins,ubins,data):
    # carry out the removal of bins on the plots
    # lbins: nº of bins to remove on the left side
    # ubins: nº of bins to remove on the right side
    bins = data[lbins:-ubins,0]
    output = data[lbins:-ubins,1:]
    return bins,output

nparam = 10
nmodels = 1000
sim = 'UNIT100' # simulation used
# array with the name of the plots we want to use
plots = ['KLF_z0','rLF_z0','early-t_z0','late-t_z0','HIMF_z0','early-f_z0','TF_z0',
         'bulge-BH_z0','Zstars_z0','KLF_z1.1']
xlabel = []
ylabel = []
xlim = []
ylim = []
# weight for each plot in the emulator training
weight = [1,1,1,1,1,1,1,1,1,1]

def check_cut(ind,nmodels,bins,output,xlab,ylab):
    count = 0
    fig = plt.figure(ind)
    for i in range(nmodels):
        count += np.shape(np.where(output[:,i]==0))[1]%%!
        plt.plot(bins[:,0],output[:,i])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    return count
        
# loading data
# DATA MUST BE ALREADY SHUFFLE
len_b = np.array([])
bins = np.array([])
output = np.array([])
for i in range(len(plots)):
    
    file = plots[i]+sim+'.dat'
    data = np.loadtxt(file)
    bins0,output0 = cut_plot(lbins,ubins,data)
    count = check_cut(0,nmodels,bins0,output0,)
    print(count)
    len_b = np.concatenate([len_b,len(bins0)])
    bins = np.concatenate([bins,bins0])
    output = np.concatenate([output,output0])
            
nbins = len(bins)
        
# input free parameters (Latin Hypercube)
# DATA ALREADY SHUFFLE IN THE SAME WAY AS OUTPUT
input_p = np.loadtxt('input_shuffle.dat')

    
# divide training (80%), evaluation (20%), test (20%)
n_train = 0.8*nmodels
n_eval = 0.2*nmodels
n_test = 0.2*nmodels

output = np.transpose(output)
output_test = output[:n_test]
input_test = input_p[:n_test]
output_training = output[n_test:]
input_training = input_p[n_test:]
# shuffling data train and evaluation
#np.random.shuffle(data_train)
input_train = input_training[:n_train]
output_train = output_training[:n_train]
input_eval = input_training[n_train:]
output_eval = output_training[n_train:]

(x_train, y_train) = (input_train, output_train)
(x_eval, y_eval) = (input_eval, output_eval)
(x_test, y_test) = (input_test, output_test)


x_train_f = x_train.astype("float32")
x_eval_f = x_eval.astype("float32")
x_test_f = x_test.astype("float32")
y_train_f = y_train.astype("float32")
y_eval_f = y_eval.astype("float32")
y_test_f = y_test.astype("float32")


# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test_f, y_test_f, batch_size=1)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for samples")
predictions = model.predict(x_test_f)#print(predictions)
print("predictions shape:", predictions.shape)

mae_a = []
maeObject = keras.losses.MeanAbsoluteError()
for i in range(n_test):
    maeTensor = maeObject(y_test_f[i,:], predictions[i])
    mae = maeTensor.numpy()
    print(mae)
    mae_a.append(mae)
mae_a = np.array(mae_a)    

print(model.metrics_names)


from matplotlib.pyplot import cm

def check_test(ind,n_train,galf,pred,xlab,ylab,xlim,ylim,file,comm):
    fig = plt.figure(ind,figsize=(9.6,7.2))
    for i,c in zip(range(n_test),color):
        plt.plot(galf[i,:],pred[i,:],'.',c=c,markersize=10)
    plt.plot(np.linspace(xlim[0],xlim[1],100),np.linspace(ylim[0],ylim[1],100),'-k')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    
    #galf = galf.flatten()
    #pred = pred.flatten()
    # ratio
    #tofile = zip(galf,pred/galf)
    #with open(file, 'w') as outf: # written mode (not appended)
    #    outf.write(comm)
    #    np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
    #    outf.closed 
    return

acum = 0
for i in range(len(plots)):
    
    if plots[i]=='KLF_z0':
        check_test(0,n_train,output_test[acum:acum+len_b[0]],predictions[acum:acum+len_b[0]])
        acum += len_b[0]
    elif plots[i]=='rLF_z0':
        check_test(1,n_train,output_test[acum:acum+len_b[1]],predictions[acum:acum+len_b[1]])
        acum += len_b[1]
    elif plots[i]=='early-t_z0':
        check_test(2,n_train,output_test[acum:acum+len_b[2]],predictions[acum:acum+len_b[2]])
        acum += len_b[2]
    elif plots[i]=='late-t_z0':
        check_test(3,n_train,output_test[acum:acum+len_b[3]],predictions[acum:acum+len_b[3]])
        acum += len_b[3]
    elif plots[i]=='HIMF_z0':
        check_test(4,n_train,output_test[acum:acum+len_b[4]],predictions[acum:acum+len_b[4]])
        acum += len_b[4]
    elif plots[i]=='early-f_z0':
        check_test(5,n_train,output_test[acum:acum+len_b[5]],predictions[acum:acum+len_b[5]])
        acum += len_b[5]
    elif plots[i]=='TF_z0':
        check_test(6,n_train,output_test[acum:acum+len_b[6]],predictions[acum:acum+len_b[6]])
        acum += len_b[6]
    elif plots[i]=='bulge-BH_z0':
        check_test(7,n_train,output_test[acum:acum+len_b[7]],predictions[acum:acum+len_b[7]])
        acum += len_b[7]
    elif plots[i]=='Zstars_z0':
        check_test(8,n_train,output_test[acum:acum+len_b[8]],predictions[acum:acum+len_b[8]])
        acum += len_b[8]
    elif plots[i]=='KLF_z1.1':
        check_test(9,n_train,output_test[acum:acum+len_b[9]],predictions[acum:acum+len_b[9]])
        acum += len_b[9]
    elif plots[i]=='mgasf_z0':
        check_test(10,n_train,output_test[acum:acum+len_b[10]],predictions[acum:acum+len_b[10]])
        acum += len_b[10]



plt.rcParams.update({'font.size': 22})

def check_test2(ind,n_train,n_test,bins,output,galf,pred,xlab,ylab,xlim,ylim):
    fig = plt.figure(ind,figsize=(9.6,7.2))
    for i in range(n_train):
        if i==0:
            plt.plot(bins[:,0],output[:,i],c='lightgrey',ls='-',label='Galform training')  
        else:
            plt.plot(bins[:,0],output[:,i],c='lightgrey',ls='-')  
    for i, c in zip(range(n_test), color):
        if i==0:
            plt.plot(bins[:,0],galf[i,:],'-',c=c,linewidth=2.5,label='Galform test')
            plt.plot(bins[:,0],pred[i,:],':',c=c,linewidth=3,label='Emulator',zorder=200)
        else:
            plt.plot(bins[:,0],galf[i,:],'-',c=c,linewidth=2.5)
            plt.plot(bins[:,0],pred[i,:],':',c=c,linewidth=3)    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    # Put a legend to the right of the current axis
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    return

acum = 0
for i in range(len(plots)):
    
    if plots[i]=='KLF_z0':
        check_test2(0,n_train,n_test,bins[acum:acum+len_b[0]],output_train[acum:acum+len_b[0]],output_test[acum:acum+len_b[0]],predictions[acum:acum+len_b[0]])
        acum += len_b[0]
    elif plots[i]=='rLF_z0':
        check_test2(1,n_train,n_test,bins[acum:acum+len_b[1]],output_train[acum:acum+len_b[1]],output_test[acum:acum+len_b[1]],predictions[acum:acum+len_b[1]])
        acum += len_b[1]
    elif plots[i]=='early-t_z0':
        check_test2(2,n_train,n_test,bins[acum:acum+len_b[2]],output_train[acum:acum+len_b[2]],output_test[acum:acum+len_b[2]],predictions[acum:acum+len_b[2]])
        acum += len_b[2]
    elif plots[i]=='late-t_z0':
        check_test2(3,n_train,n_test,bins[acum:acum+len_b[3]],output_train[acum:acum+len_b[3]],output_test[acum:acum+len_b[3]],predictions[acum:acum+len_b[3]])
        acum += len_b[3]
    elif plots[i]=='HIMF_z0':
        check_test2(4,n_train,n_test,bins[acum:acum+len_b[4]],output_train[acum:acum+len_b[4]],output_test[acum:acum+len_b[4]],predictions[acum:acum+len_b[4]])
        acum += len_b[4]
    elif plots[i]=='early-f_z0':
        check_test2(5,n_train,n_test,bins[acum:acum+len_b[5]],output_train[acum:acum+len_b[5]],,output_test[acum:acum+len_b[5]],predictions[acum:acum+len_b[5]])
        acum += len_b[5]
    elif plots[i]=='TF_z0':
        check_test2(6,n_train,n_test,bins[acum:acum+len_b[6]],output_train[acum:acum+len_b[6]],output_test[acum:acum+len_b[6]],predictions[acum:acum+len_b[6]])
        acum += len_b[6]
    elif plots[i]=='bulge-BH_z0':
        check_test2(7,n_train,n_test,bins[acum:acum+len_b[7]],output_train[acum:acum+len_b[7]],output_test[acum:acum+len_b[7]],predictions[acum:acum+len_b[7]])
        acum += len_b[7]
    elif plots[i]=='Zstars_z0':
        check_test2(8,n_train,n_test,bins[acum:acum+len_b[8]],output_train[acum:acum+len_b[8]],output_test[acum:acum+len_b[8]],predictions[acum:acum+len_b[8]])
        acum += len_b[8]
    elif plots[i]=='KLF_z1.1':
        check_test2(9,n_train,n_test,bins[acum:acum+len_b[9]],output_train[acum:acum+len_b[9]],output_test[acum:acum+len_b[9]],predictions[acum:acum+len_b[9]])
        acum += len_b[9]
    elif plots[i]=='mgasf_z0':
        check_test2(10,n_train,n_test,bins[acum:acum+len_b[10]],output_train[acum:acum+len_b[10]],output_test[acum:acum+len_b[10]],predictions[acum:acum+len_b[10]])
        acum += len_b[10]



#10000 points over a latin hypercube to sample the parameter space
lhd = np.loadtxt('hypercube_10p_10000.dat')

def plot_LH(ind,i1,i2,xlab,ylab,xlim,ylim):
    fig = plt.figure(ind,figsize=(9.8,9.8))
    ax = plt.subplot(111)
    ax.plot(lhd[:.i1],lhd[:,i2],'.b')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_box_aspect(1)
    plt.show()
    return

for i in range(nparam):
    for j in range(nparam):
        plot_LH(i+j,i,j,)

# Generating the predictions for the 10000 points
x_post = [lhd[:,0],lhd[:,1],lhd[:,2],lhd[:,3],lhd[:,4],lhd[:,5],lhd[:,6],lhd[:,7],lhd[:,8],lhd[:,9]]
print(np.shape(x_post))
x_post = np.transpose(x_post)
z = x_post.astype("float32")
pre = model.predict(z)
