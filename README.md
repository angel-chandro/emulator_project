
For the fnl_sam simulations I have been running the Dhalos code in the path /home/chandro/dhalo-trees_rhalf+snap_nfids_mod (it is exactly the same as the one in /home/chandro/dhalo-trees_rhalf+snap_nfids_nftab, but I have been using it for different tests). The different parameter files are stored in the directory /home/chandro/dhalo-trees_rhalf+snap_nfids_mod/Parameters/fnl_sam. The Dhalos output data has been stored for the moment in the directory /home/chandro/fnl_sam. While the Galform output data has been placed in /home/chandro/Galform_Out/simulations/fnl_sam/test.

PROBLEMS:
- the dispersion code is given wrong results due to the fact that the type 2 galaxies are defined with positions=-1 and velocities=-2 even when the Vimal's merging scheme is employed.
- the simulation with the Dhalos merger trees hasn't been run over Galform since the number of merging galaxies is higher than the limit imposed (NMERGEMAX=5000). The error is found in "merging.find_merger_time.f90".


-------------------------------------------------------------------------------------------

1. Choose the input free parameters of the SAM whose parameter space we want to analyse and their range from Elliott et al. 2021 (10 free parameters).
Choose also the output observables functions and their specific bins from Elliott+2021. 
- calibration.py: given the Galform output, it generates the observable functions. It also save the bins and the values of the observable functions (data to be used by the emulator) we consider for calibration for the models.
- calibration_plots.py: make the calibration plots with the data obtained from calibration.py 
- calibration_models.py: compute and graphic the calibration plots for 1 or more than 1 models to compare different runs. 
- calibration_dispersion.py: compute and graphic the calibration plots for 250Mpc/h subboxes of the UNIT simulation to analyse the dispersion of these plots. 
- calibration_dispersion_ratio.py: given the output from calibration.dispersion.py it computes the ratio between the subboxes and the "real value" of the whole 1000Mpc/h volume of the UNIT simulation. This is a visible way to quantify the dispersion, which could be useful to define the range of these plots where we are going to train the emulator (where the dispersion is lower than 10% f.e.). 

----------------------------------------------------------------------------------------------

2. MAXIMIN LATIN HYPERCUBE (MLH)

Number of models we use for training following Elliott+2021: 1000 models. The values of the parameters for these 1000 models are distributed over a Latin Hypercube to have the best sampling of the parameter space with the minimum number of points.
Separate 80% training set, 10% evaluation (used to impose the condition that ends the training), 10% test (to study and compare the emulator performance).
- hypercube.py: code to generate the Latin Hypercube of the 10 parameters over 1000 models.


-----------------------------------------------------------------------------------------------

3. RUN GALFORM

Run these 1000 models taking advantage of parallelization tools:
- run_galform_em.csh: there is a wide variety of different models and simulations defined, as well as the output properties you can choose. Flags: set only "galform" (to run galform) and "elliott" (to produce the desired output) to true, while "models_dir" to indicate the output path and "./delete_variable.csh $galform_inputs_file aquarius_particle_file" in case there are no particle files. It generates the same number of subvolumes as the input Dhalos merger trees are distributed in. The difference respect to "run_galform.csh" is that Galform uses the model "gp19.vimal.em.project" in which each Galform run has a different set of free parameters (those we are going to study their variation), so the input free parameters take the value of the corresponding Latin Hypercube position and each model itself is stored in a different directory whose name indicate the parameter values.
- qsub_galform_par_em.sh: it reads the parameters of the latin hypercube from a file (each line corresponds to the ten parameter values) for 1 model at a time (4 jobs, 16 subvolumes per job, 1 cpu per subvolume). (./qsub_galform_par_em.sh)
- qsub_galform_par_em_eff.sh: it reads the parameters of the latin hypercube from a file (each line corresponds to the ten parameter values) for 2 models at the same time (1 job, 128 subvolumes per job, 1 cpu per subvolume). (./qsub_galform_par_em_eff.sh)


The same steps would be done in case the SHARK semi-analytic model was employed.

----------------------------------------------------------------------------------------------------------------------------------------------


4. GENERATE CALIBRATION PLOTS DATA

Once the training models have been run, you have to generate the arrays (bins + observable functions values) for the 1000 runs. I have to extend the "calibration.py" code to be used over the 1000 models and save all the generated data.
- calibration_em.py: save the bins and the values of the observable functions (data to be used by the emulator) for the 1000 models.

---------------------------------------------------------------------------------------------------


5. EMULATOR

I have finally installed Tensorflow in Taurus with the command: "pip3 install tensorflow --user" (with anaconda activated).
If you obtain some error messages related to non-updated libraries, run "pip3 install --user --upgrade matplotlib".

- #hypercube_shuffle.py (still need to be tested): once the SAM has been run over the 1000 models in case the sampling of the training, evaluation or test set doesn't seem to be adequate visually, shuffle the free parameters and the output from the observables. Although K-FOLD is implemented (the K-FOLD applied only varies the training and the validation sets with a fixed test set), the shuffle has to be taken into account.

TensorFlow codes in Jupiter notebook.

- #emulator_elliott-SAVE.ipynb/.py (still need to be tested): it trains the emulator and it saves it. It separates the 1000 models: 80% training set, 10% evaluation (used to impose the condition that ends the training), 10% test (to study and compare the emulator performance).
- hypercube_10000models.py: once the emulator is trained, span the whole parameter space through 10000 points in a MLH again make use of the emulator.
- #emulator_elliott-POST.ipynb/.py (still in progress): once the emulator trained, it generates the predictions of the observable functions to study the whole parameter space. Compute the chi square compared to the different observational data employed and it finds the set of free parameters most consistent to observations (best fit with the lowest chi square). Run the best fit with the SAM itself and compare both results. Some parts haven't been extrapolated to the current project yet.
- #emulator_elliott-KFOLD.ipynb/.py (still in progess): k-fold technique applied (only to the training and validation sets, not the test set that is fixed). Generating the emulators.
- #emulator_elliott-POST-KFOLD.ipynb/.py (still in progess): k-fold technique applied (only to the training and validation sets, not the test set that is fixed). Using the emulators to study the probability distribution

K-FOLD TECHNIQUE COULD BE IMPLEMENTED INCLUDING ALSO THE TEST DATA (THE SHUFFLING CODE WOULDN'T BE NECESSARY).