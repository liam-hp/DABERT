#!/usr/bin/env bash
# slurm template for serial jobs
# Set SLURM options
#SBATCH --job-name=hello_world # Job name
#SBATCH --output=hello_world-%j.out
# Standard output and error log
#SBATCH --mail-user=aballo@middlebury.edu
# Where to send mail
#SBATCH --mail-type=ALL
# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=100mb
# Job memory request 
#SBATCH --partition=standard
# Partition (queue) 
#SBATCH --time=00:01:00
# Time limit hrs:min:sec 
#SBATCH --partition=gpu-standard 

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" echo "Starting: "`date +"%D %T"` 
echo "$USER"
# Running your code here 
python BERT_run.py
# End of job info 
echo "Ending: "`date +"%D %T"`

