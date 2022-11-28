1. Run Dhalos: upload code to github. Needed to run SAMs over merger trees because it gives the appropriate format. Able to be run in parallel. 2 versions: the one used to run the whole UNIT simulation (2 new flags added) and the one that allows to run Gadget4 simulations (Gadget 4 merger trees or Dhalos merger trees, I need to add a flag referring to the number of files in Find_Descendants)
- Gadget4 merger trees or UNIT simulation (taking advantage of the Consistent Trees merger trees): only necessary to run Build_Trees executable.
- Create own Dhalos merger trees: run Find_Descendants executable.
- Trace particles for type 2 galaxies: run Trace_Particles (to use in later in Galform). Run Gadget4 with the SUBFIND_ORPHAN_TREATMENT activated to have catalogues with only the most bound particles of the orphan galaxies (way of saving memory). Anyways Gadget4 always produces as output the catalogues with all the simulated particles. Doubt: flag update_tree_files T or F modifies positions and velocities of interpolated halos, ask to John Helly? Following John's answer is better to set the flag to T.

2. Run SAMs: GALFORM or SHARK.
- GALFORM: different codes to send parameter files and output required.
- SHARK: no parallelization easy to run. I discovered how to solve the problem running in parallel. Different codes to run it.

3. Emulators: TensorFlow codes in Jupiter notebook.
- 1. Choose input parameters from the SAM whose parameter space we want to analyse and their range. Choose also the output observables functions and their specific bins. Codes that generate the observable functions. 
- 2. Number of models you want to use for training. Distribute them over a Latin Hypercube. Separate 80% training set, 10% evaluation (used to impose the condition that ends the training), 10% test (to study and compare the emulator performance).
- 3. Run these Latin Hypercube models taking advantage of parallelization tools (codes I worked with).
- 4. Follow what it is done in the Jupyter notebooks, but extrapolated to the current project.
- 5. Once the emulator trained, cover the parameter space with a large number of points and predicts the values of the observable functions. Compute the chi square compared to the different observational data employed.
- 6. Find the best fit: parameter space values with the lowest chi square. RUn the best fit with the SAM itself and compare both results.

CODES
- How to run DHalos.
- How to run Galform (parallelization for emulator training models).
- How to run SHARK (parallelization).
- Generate observable functions.
- Generate Latin Hypercube.
- Emulator Jupyter notebooks.


----------------------------------------------------------------------------------------------

DHalos:

New parameters added:
- rhalf_consistenttrees_aprox (Build_Trees): whether or not an approximation for the half mass
is used. Specific to UNIT simulations and consistent trees merger trees.
- snap0 (Build_Trees): whether or not the snap=0 is considered. Specific to UNIT simulations
and consistent trees merger trees.
- n_files_ids (Find_Descendants): number of files where IDs are stored when constructing the
merger tree directly with DHalos from Subfind data. Specific to the different formats that
work with Subfind ("LGADGET2","LGADGET3","PGADGET3","COCO","GADGET4_HDF5","EAGLE")

New parameter files:
/Parameters/UNITsim+CT: whole 1Gpc/h UNIT simulation (ConsistentTrees merger trees). Build_Trees. Data generated in /data8/vgonzalez/SAMs/trees/
/Parameters/UNITsim+SUBFIND_GHDF5+S_multiple_HBT : Gadget4 with Gadget4 own merger trees.
Build_Trees+Trace_Particles(if we want to use most bound particles to define satellites)
/Parameters/UNITsim+SUBFIND_GHDF5+D_multiple_HBT : Gadget4 with Dhalos own merger trees.
Find_Descendants+Build_Trees+Trace_Particles(if we want to use most bound particles to define satellites)

Code to run Dhalos using Slurm queues
/Parameters/UNITsim+CT/submit_mpi.sh: code to run UNIT simulation parallelized with 3 nodes communication.
/Parameters/UNITsim+SUBFIND_GHDF5+S_multiple_HBT and /Parameters/UNITsim+SUBFIND_GHDF5+D_multiple_HBT :
mpirun -np 8(number of Gadget4 output files) ./path/to/build/find_descendants parameter_file 
mpirun -np 8(number of Gadget4 output files) ./path/to/build/build_trees parameter_file 
mpirun -np 8(number of Gadget4 output files) ./path/to/build/trace_particles parameter_file 
These commands can be run through Slurm queueing system.

---------------------------------------------------------------------------------------------------

GALFORM

To run the code we need a parameter files (.ref file) where the different parameter values are defined.
- UNIT.ref: example.

Later, these params can be modified using the codes to run Galform and to send it to slurm queues.
Run only 1 simple model or more (not emulator): you need to provide a model (I have usually used gp19.vimal as reference and then I have changed some values)
and a simulation that has to be predefined.
- run_galform_vio_simplified.csh: you have the different models and simulations defined, as well as the output properties you can choose.
- qsub_galform_vio_simplified.csh: you choose the model and simulation and it is sent to a Slurm queue. (each subvolume is a job)
- test_par.sh: 1 model parallelized (1 job, 64 subvolumes per job, 1 subvolume per cpu)

Run 1 or more models (specific for the training models of the emulator):
- run_galform_vio_simplified_em_tfm_eff.csh: it saves each model run in a directory whose name indicates the values of the parameters varied.
- test_par_mult_em.sh: it reads the parameters of the latin hypercube file for 1 model at a time (4 jobs, 16 subvolumes per job, 1 subvolume per cpu)
- test_par_mult_em_eff.sh: it reads the parameters of the latin hypercube file for 2 models at the same time (1 job, 128 subvolumes per job, 1 subvolume per cpu)
./qsub_galform_vio_simplified.csh

Once the training models run:
- calibration.py: save the bins and the values of the observable functions (data to be used by the emulator) we consider for calibration for the different models.
- calibration_plots.py: make the calibration plots with the data obtained from calibration.py 

---------------------------------------------------------------------------------------------------

SHARK

You can find more info in the website https://shark-sam.readthedocs.io/en/latest/

To run it we need to provide the free parameter values, the path to the Dhalos output and a file indicating
the redshift-snapshot correspondence.

Not parallelized:
1 subvolume: ./shark parameter_file "simulation_batches = nº of subvolume"
All subvolumes: ./shark parameter_file -t nº of threads "simulation_batches = 0-maximum_subvolume"
(this way doesn't produce the output distributed in subvolumes)

Parallelized:
I suppose you can apply the same strategy as the one in Galform (sending different subolumes to different Slurm queues)
./shark-submit parameter_file "simulation_batches=0-maximum_subvolume" doesn't work. sbatch: error: Unable to open file hpc/shark-run
I have implemented a new "shark-run" file that send 1 subvolume per job. What it remains is to implement the same parallelization
carried out in Galform (send more than 1 model for emulator training and make the code the more efficient).

For example I haven't worked varying the different parameters, but it is described how to do it in
https://shark-sam.readthedocs.io/en/latest/optim.html

---------------------------------------------------------------------------------------------------

EMULATOR

Maximin Latin Hypercube (MHL):
- hypercube.py: code to generate the Latin Hypercube of the 10 parameters over 1000 models.
- hypercube_shuffle.py: once the SAM run over the 1000 models, shuffle the free parameters and the output from the observables.
- hypercube_10000models.py: once the emulator is trained, span the whole parameter space through 10000 points in a MHL again.

Jupyter notebook:
- : it trains the emulator and it saves it.
- : once the emulator trained, it generates the predictions to study the whole parameter space and it finds the best fit.
- : k-fold technique applied.