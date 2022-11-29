#!/bin/bash

# run models for emulator training
# 1 model each time: 16 subvolumes each job x 4 simultaneous jobs
# 64 cores queue

name='test'
logpath=/home/chandro/junk
max_jobs=64

Nbody_sim=UNIT200 # indicate the simulation

# choose the Nbody simulation
if [ $Nbody_sim == UNIT100 ]
then
    iz_list=(128) # z = 0
    
elif [ $Nbody_sim == UNIT200 ]
then
    iz_list=(128) # z = 0

elif [ $Nbody_sim == UNIT_GADGET ]
then
    iz_list=(128 95) # z = 0, 1.1

fi

# choose the model: only base model
for model in {"gp19.vimal.em.project",}
do
    echo 'Model: ' $model
    # file where input parameters are stored (each line is a Latin Hypercube position)
    input='/home/chandro/emulator_project/hypercube_10p_10000.dat'
    echo $input
    # loop over each set of input parameters
    while IFS= read -r line
    do
	echo $line
	a=($line) # transform to an array
	var1=${a[0]} # parameter 1
	echo Variable1 $var1
	var2=${a[1]} # parameter 2 
	echo Variable2 $var2
	var3=${a[2]} # parameter 3
	echo Variable3 $var3
	var4=${a[3]} # parameter 4
	echo Variable4 $var4
	var5=${a[4]} # parameter 5 
	echo Variable5 $var5
	var6=${a[5]} # parameter 6
	echo Variable6 $var6
	var7=${a[6]} # parameter 7
	echo Variable7 $var7
	var8=${a[7]} # parameter 8 
	echo Variable8 $var8
	var9=${a[8]} # parameter 9
	echo Variable9 $var9
	var10=${a[9]} # parameter 10
	echo Variable10 $var10

	# skip first comment line
	if [ $var1 == '#' ]
	then
	    continue
	fi

	# loop over redshifts
	for iz in "${iz_list[@]}"
	do
	    
	    script=run_galform_em.csh
	    jobname=$Nbody_sim.$model
    
	    for i in {0..3}
	    do
		logpath2=${logpath}/elliott/em/${Nbody_sim}
		\mkdir -p ${logpath2:h}
		logname=${logpath2}/${model}.${i}.%A.%a.log
		#\mkdir -p ${logname:h}
		job_file=${logpath2}/${model}.${i}.job

		# sends 4 jobs, each of them with 16 tasks = subvolumes
		# 64 subvolumes = 1 model at the same time
		tasks=16 
		j_i=$( expr $i \* $tasks) # values  0,16,32,48
		int1=$( expr $i + 1)
		int2=$( expr $int1 \* $tasks)
		j_f=$( expr $int2 - 1) # values 15,31,47,63
		
		cat > $job_file <<EOF
#!/bin/bash 
# 
#SBATCH --ntasks=${tasks}
#SBATCH --cpus-per-task=1
#SBATCH -J ${jobname}
#SBATCH -o ${logname}
#SBATCH --nodelist=miclap
#SBATCH -A 16cores
#SBATCH -t 4:00:00   
#
#
for ivol in {$j_i..$j_f}
do  
    echo Ivol \$ivol
    srun -n1 -c1 -N1 --exclusive ./${script} ${model} ${Nbody_sim} ${iz} \$ivol ${var1} ${var2} ${var3} ${var4} ${var5} ${var6} ${var7} ${var8} ${var9} ${var10} &
done
wait

EOF
		sbatch $job_file
		rm $job_file
	    done
	    
	done
	
    done <"$input"
    
done
echo 'Finished'
