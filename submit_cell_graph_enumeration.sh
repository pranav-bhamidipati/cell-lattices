#!/bin/bash

#Submit this script with: sbatch submit_cell_graph_enumeration.sh

#SBATCH --time=3:00:00               # walltime
#SBATCH --ntasks=64                  # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=4G             # memory per CPU core
#SBATCH -J "CellGraphClustering"     # job name
#SBATCH --mail-user=pbhamidi@usc.edu # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH -o /home/pbhamidi/scratch/cell-lattices/slurm_out/slurm.%N.%j.out # STDOUT
#SBATCH -e /home/pbhamidi/scratch/cell-lattices/slurm_out/slurm.%N.%j.err # STDERR

 
#======START===============================

source ~/.bashrc
echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST"
echo "A total of $SLURM_NTASKS tasks is used"
echo "Environment Variables"
env
echo ""

echo "Activating conda environment"
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ~/git/cell-lattices/env
echo ""

CMD="python3 cell_graph_enumeration.py"
echo $CMD
$CMD

#======END================================= 


