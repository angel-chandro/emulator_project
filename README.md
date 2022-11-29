
1. Choose the input free parameters of the SAM whose parameter space we want to analyse and their range from Elliott et al. 2021 (10 free parameters).
Choose also the output observables functions and their specific bins from Elliott+2021. 
- calibration.py: given the Galform output, it generates the observable functions. It also save the bins and the values of the observable functions (data to be used by the emulator) we consider for calibration for the models.
- calibration_plots.py: make the calibration plots with the data obtained from calibration.py 

----------------------------------------------------------------------------------------------

2. MAXIMIN LATIN HYPERCUBE (MHL)

Number of models we use for training following Elliott+2021: 1000 models. The values of the parameters for these 1000 models are distributed over a Latin Hypercube to have the best sampling of the parameter space with the minimum number of points.
Separate 80% training set, 10% evaluation (used to impose the condition that ends the training), 10% test (to study and compare the emulator performance).
- hypercube.py: code to generate the Latin Hypercube of the 10 parameters over 1000 models.


-----------------------------------------------------------------------------------------------

2. RUN GALFORM

Run these 1000 models taking advantage of parallelization tools:
- run_galform_em.csh: there is a wide variety of different models and simulations defined, as well as the output properties you can choose. Flags: set only "galform" (to run galform) and "elliott" (to produce the desired output) to true, while "models_dir" to indicate the output path and "./delete_variable.csh $galform_inputs_file aquarius_particle_file" in case there are no particle files. It generates the same number of subvolumes as the input Dhalos merger trees are distributed in. The difference respect to "run_galform.csh" is that Galform uses the model "gp19.vimal.em.project" in which each Galform run has a different set of free parameters (those we are going to study their variation), so the input free parameters take the value of the corresponding Latin Hypercube position and each model itself is stored in a different directory whose name indicate the parameter values.
- qsub_galform_par_em.sh: it reads the parameters of the latin hypercube from a file (each line corresponds to the ten parameter values) for 1 model at a time (4 jobs, 16 subvolumes per job, 1 cpu per subvolume). (./qsub_galform_par_em.sh)
- qsub_galform_par_em_eff.sh: it reads the parameters of the latin hypercube from a file (each line corresponds to the ten parameter values) for 2 models at the same time (1 job, 128 subvolumes per job, 1 cpu per subvolume). (./qsub_galform_par_em_eff.sh)


The same steps would be done in case the SHARK semi-analytic model was employed.

----------------------------------------------------------------------------------------------------------------------------------------------


3. GENERATE CALIBRATION PLOTS DATA

Once the training models have been run, you have to generate the arrays (bins + observable functions values) for the 1000 runs. I have to extend the "calibration.py" code to be used over the 1000 models and save all the generated data.
- calibration_em.py: save the bins and the values of the observable functions (data to be used by the emulator) for the 1000 models.

---------------------------------------------------------------------------------------------------


4. EMULATOR

- hypercube_shuffle.py: once the SAM has been run over the 1000 models in case the sampling of the training, evaluation or test set doesn't seem to be adequate visually, shuffle the free parameters and the output from the observables.

TensorFlow codes in Jupiter notebook.

- emulator_elliott-SAVE.ipynb: it trains the emulator and it saves it. It separates the 1000 models: 80% training set, 10% evaluation (used to impose the condition that ends the training), 10% test (to study and compare the emulator performance).
- hypercube_10000models.py: once the emulator is trained, span the whole parameter space through 10000 points in a MHL again make use of the emulator.
- emulator_elliott-POST.ipynb: once the emulator trained, it generates the predictions of the observable functions to study the whole parameter space. Compute the chi square compared to the different observational data employed and it finds the set of free parameters most consistent to observations (best fit with the lowest chi square). Run the best fit with the SAM itself and compare both results. Some parts haven't been extrapolated to the current project yet.

- emulator_elliott-K-FOLD.ipynb: k-fold technique applied. Still not implemented.

