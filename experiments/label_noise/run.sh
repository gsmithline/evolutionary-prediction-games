#! /bin/bash

#SBATCH --cpus-per-task 16
#SBATCH --gres gpu:1

ind=${SLURM_ARRAY_TASK_ID}
total=${SLURM_ARRAY_TASK_MAX}
jobid=${SLURM_ARRAY_JOB_ID}
id=${jobid}_${ind}_${total}
p=$(python -c "print($ind/$total)")
echo hostname=`hostname`
echo id=${id}
echo p=${p}

script_dir=../../scripts
python $script_dir/write_mnist.py --p ${p} --runid ${id}
python $script_dir/train_mnist.py --runid ${id}

# example: 
# for i in {1..20}; do sbatch -a 0-10 --nodelist node_name ../../scripts/run.sh; done
# watch squeue -u $USER
# SLURM_ARRAY_JOB_ID=0 SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_MAX=1 bash ../scripts/run.sh