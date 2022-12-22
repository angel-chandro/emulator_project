import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

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
weight = [2,2,1,1,3,1,1,1,1,2,1]
# cuts
lcut = []
ucut = []

def check_cut(ind,nmodels,bins,output,xlab,ylab):
    # check the cuts that has just been made in a plot
    # to see if the ranges are correct
    count = 0
    fig = plt.figure(ind)
    for i in range(nmodels):
        count += np.shape(np.where(output[:,i]==0))[1]
        plt.plot(bins[:,0],output[:,i])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    return count

# loading data
# DATA MUST BE ALREADY SHUFFLE (no K-FOLD case)
len_b = np.array([])
bins = np.array([])
output = np.array([])
for i in range(len(plots)):
    # load data from file into bins and output
    # len_b to see the length of each plot
    file = plots[i]+sim+'.dat'
    data = np.loadtxt(file)
    bins0,output0 = cut_plot(lbins,ubins,data)
    count = check_cut(0,nmodels,bins0,output0,)
    print(count)
    len_b = np.concatenate([len_b,len(bins0)])
    bins = np.concatenate([bins,bins0])
    output = np.concatenate([output,output0])

plt.close('all')
nbins = len(bins)
        
# input free parameters (Latin Hypercube)
# DATA ALREADY SHUFFLE IN THE SAME WAY AS OUTPUT (no K-FOLD case)
input_p = np.loadtxt('input_shuffle.dat')

    
# divide training (80%), evaluation (10%), test (10%)
n_train = 0.8*nmodels
n_eval = 0.1*nmodels
n_test = 0.1*nmodels

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
(x_data, y_data) = (input_p, output)

# Latin Hypercube distribution of free parameters
# over parameter space
# 10 free parameters

plt.rcParams.update({'font.size': 22})

def plot_LH(ind,i1,i2,xlab,ylab,xlim,ylim):
    # plot the LH points
    fig = plt.figure(ind,figsize=(9.8,9.8))
    ax = plt.subplot(111)
    ax.plot(x_train[:,i1],x_train[:,i2],'.b',markersize=15,label='Training')
    ax.plot(x_eval[:,i1],x_eval[:,i2],'.r',markersize=15,label='Evaluation')
    ax.plot(x_test[:,i1],x_test[:,i2],'.g',markersize=15,label='Test')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_box_aspect(1)
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    return

for i in range(nparam):
    for j in range(nparam):
        plot_LH(i+j,i,j,)
        
plt.close('all')

def check_training(ind,n_train,bins,output,xlab,ylab):
    # check the behaviour of the training model
    # and if they span the parameter space properly
    fig = plt.figure(ind)
    for i in range(n_train):
        plt.plot(bins[:,0],output[:,i])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    return

acum = 0
for i in range(len(plots)):
    
    if plots[i]=='KLF_z0':
        check_training(0,n_train,bins[acum:acum+len_b[0]],output[acum:acum+len_b[0]])
        acum += len_b[0]
    elif plots[i]=='rLF_z0':
        check_training(0,n_train,bins[acum:acum+len_b[1]],output[acum:acum+len_b[1]])
        acum += len_b[1]
    elif plots[i]=='early-t_z0':
        check_training(0,n_train,bins[acum:acum+len_b[2]],output[acum:acum+len_b[2]])
        acum += len_b[2]
    elif plots[i]=='late-t_z0':
        check_training(0,n_train,bins[acum:acum+len_b[3]],output[acum:acum+len_b[3]])
        acum += len_b[3]
    elif plots[i]=='HIMF_z0':
        check_training(0,n_train,bins[acum:acum+len_b[4]],output[acum:acum+len_b[4]])
        acum += len_b[4]
    elif plots[i]=='early-f_z0':
        check_training(0,n_train,bins[acum:acum+len_b[5]],output[acum:acum+len_b[5]])
        acum += len_b[5]
    elif plots[i]=='TF_z0':
        check_training(0,n_train,bins[acum:acum+len_b[6]],output[acum:acum+len_b[6]])
        acum += len_b[6]
    elif plots[i]=='bulge-BH_z0':
        check_training(0,n_train,bins[acum:acum+len_b[7]],output[acum:acum+len_b[7]])
        acum += len_b[7]
    elif plots[i]=='Zstars_z0':
        check_training(0,n_train,bins[acum:acum+len_b[8]],output[acum:acum+len_b[8]])
        acum += len_b[8]
    elif plots[i]=='KLF_z1.1':
        check_training(0,n_train,bins[acum:acum+len_b[9]],output[acum:acum+len_b[9]])
        acum += len_b[9]
    elif plots[i]=='mgasf_z0':
        check_training(0,n_train,bins[acum:acum+len_b[10]],output[acum:acum+len_b[10]])
        acum += len_b[10]
        
plt.close('all')

# SOLVE PROBLEM IF KLF_z=1.1 IS NOT USED AND MASS GAS FRACTION IS 

Here's what the typical end-to-end workflow looks like, consisting of:

- Training
- Validation on a holdout set generated from the original training data
- Evaluation on the test data

We'll use MNIST data for this example.

# Preprocess the data (these are NumPy arrays)
x_train_f = x_train.astype("float32")
x_eval_f = x_eval.astype("float32")
x_test_f = x_test.astype("float32")
y_train_f = y_train.astype("float32")
y_eval_f = y_eval.astype("float32")
y_test_f = y_test.astype("float32")

y_data_f = y_data.astype("float32")
y_data_f = y_data.astype("float32")

We specify the training configuration (optimizer, loss, metrics):

(x_data_train_f, y_data_train_f) = (input_training, output_training)
(x_test_f, y_test_f) = (input_test, output_test)


x_data_train_f = x_data_train_f.astype("float32")
x_test_f = x_test_f.astype("float32")
y_data_train_f = y_data_train_f.astype("float32")
y_test_f = y_test_f.astype("float32")

# define KFOLD
# here the test data is the same, and we vary the training and the validation set
nsplit = 9 # split into 9 (100 validation of 900=800+100)
kf = KFold(n_splits=nsplit,shuffle=False)
save_dir = 'saved_models_kfold'
fold_var = 1
nepoch = 500 # to impose the condition when to stop the training

def get_model():
    # define the emulator configuration
    inputs = keras.Input(shape=(nparam,), name="digits")
    x = layers.Dense(512, activation=tf.keras.activations.sigmoid, name="dense_1")(inputs)
    x = layers.Dense(512, activation=tf.keras.activations.sigmoid, name="dense_2")(x)
    outputs = layers.Dense(nbins, activation="linear", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
    optimizer=tf.keras.optimizers.Adam(amsgrad=True,name='Adam_ams'),  # Optimizer
    # Loss function to minimize
    loss='MAE',
    # List of metrics to monitor
    metrics=['accuracy'],
    )
    return model
    
for train_idx, val_idx in list(kf.split(x_data_train_f,y_data_train_f)):
    
    x_train_f = x_data_train_f[train_idx]
    y_train_f = y_data_train_f[train_idx]
    x_eval_f = x_data_train_f[val_idx]
    y_eval_f = y_data_train_f[val_idx]
    
    model = get_model()
    
    loss = []
    val_loss = []

    # only update the emulator when it improves its performance over the validation data
    # stop if nepoch have passed without improvement
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=nepoch, verbose=1, restore_best_weights=True)
    
    # TRAINING
    print("Fit model on training data")
    history = model.fit(
        x_train_f,
        y_train_f,
        batch_size=1,
        epochs=10000,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_eval_f, y_eval_f),
        callbacks=[callback]
    )

    loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])
    loss_p = np.array(loss)
    val_loss_p = np.array(val_loss)
    loss_p = loss_p.flatten()
    val_loss_p = val_loss_p.flatten()
    
    # freezing
    # compile and retrain with a low learning rate
    low_lr = 1e-5
    model.compile(loss='MAE',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=low_lr), 
                  metrics=['accuracy']
    )
    history = model.fit(
        x_train_f,
        y_train_f,
        batch_size=1,
        epochs=1,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_eval_f, y_eval_f),
    )
    
    loss_f = np.array(history.history['loss'])
    val_loss_f = np.array(history.history['val_loss'])

    loss_t = np.concatenate((loss_p,loss_f))
    val_loss_t = np.concatenate((val_loss_p,val_loss_f))
    loss_t = loss_t.flatten()
    val_loss_t = val_loss_t.flatten()

    # plot training curves with freezing
    plt.figure(figsize=(9.8,7.2))
    plt.plot(np.linspace(1,len(loss_t),len(loss_t)),loss_t,'-b',label='Training data')
    plt.plot(np.linspace(1,len(loss_t),len(loss_t)),val_loss_t,'-r',label='Validation data')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.xlim(0,len(loss_t))
    #plt.ylim(0,0.2)
    plt.legend()
    plt.show()
    
    outfil = save_dir+'performance_em_'+str(fold_var)+'.dat'
    tofile = zip(loss_t,val_loss_t)
    with open(outfil, 'w') as outf: # written mode (not appended)
        outf.write('# MAE training, MAE validation \n')
        np.savetxt(outf,list(tofile))#,fmt=('%.5f'))
        outf.closed 

    # Evaluate the model on the test data using `evaluate`
    # the same test data all the time
    print(x_test_f.shape)
    print(y_test_f.shape)
    print("Evaluate on test data")
    results = model.evaluate(x_test_f, y_test_f, batch_size=1)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for samples")
    if fold_var == 1:
        predictions1 = model.predict(x_test_f)
        print("predictions shape:", predictions1.shape)
    elif fold_var == 2:
        predictions2 = model.predict(x_test_f)
        print("predictions shape:", predictions2.shape)
    elif fold_var == 3:
        predictions3 = model.predict(x_test_f)
        print("predictions shape:", predictions3.shape)
    elif fold_var == 4:
        predictions4 = model.predict(x_test_f)
        print("predictions shape:", predictions4.shape)
    elif fold_var == 5:
        predictions5 = model.predict(x_test_f)
        print("predictions shape:", predictions5.shape)
    elif fold_var == 6:
        predictions6 = model.predict(x_test_f)
        print("predictions shape:", predictions6.shape)
    elif fold_var == 7:
        predictions7 = model.predict(x_test_f)
        print("predictions shape:", predictions7.shape)
    elif fold_var == 8:
        predictions8 = model.predict(x_test_f)
        print("predictions shape:", predictions8.shape)
    elif fold_var == 9:
        predictions9 = model.predict(x_test_f)
        print("predictions shape:", predictions9.shape)
    #print(y_test)

    # check that the test MAE is simply the mean
    mae_a = []
    maeObject = keras.losses.MeanAbsoluteError()
    for i in range(n_test):
        maeTensor = maeObject(y_test_f[i,:], predictions[i])
        mae = maeTensor.numpy()
        print(mae)
        mae_a.append(mae)
    mae_a = np.array(mae_a)    

    print(model.metrics_names)

    # save each of the nsplit models
    model.save(save_dir+str(fold_var))
    
    fold_var +=1

# generate the final prediction of the test data
predictions = (predictions1+predictions2+predictions3+predictions4+predictions5+predictions6+predictions7+predictions8+predictions9)/9

def check_test(ind,n_train,galf,pred,xlab,ylab,xlim,ylim,file,comm):
    # compare the predictions of the test and the real values visibly
    # the closer to the y=x curve, the more accurate the emulator performance is
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

plt.close('all')

plt.rcParams.update({'font.size': 22})

def check_test2(ind,n_train,n_test,bins,output,galf,pred,xlab,ylab,xlim,ylim):
    # compare the training models (real values) with the test models (real values)
    # to check if both of them span correctly the parameter space
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

plt.close('all')
