#!/bin/bash
#SBATCH --partition=standard --time=04:00:00
#SBATCH -c 24

# Runs non-temporal classification using various classification methods and 5 different seeds on the 
# default, changes-only, listening-only, and speaking-only cluster index sequences infolders. All frames 
# are sampled (15 fps) since the cluster distributions are used as X in the classification. Both 5 and
# 10 folds of KFold cross-validation are tested. 

# The results are written to results.csv for later analysis, designed for use on BlueHive cluster.

# Usage: 'sbatch batch.sh' to dispatch to compute node or './batch.sh' to run on interactive node.

module load python3
module load anaconda

# Base directory of the cluster index data
basefolder=/home/mlevin6/Desktop/cluster/cluster_sequences

# Infolders to use
infolders=( default/every_frame changes listening/every_frame speaking/every_frame )
# Seeds to use
seeds=( 101 2002 30003 400004 5000005 )
# Folds to use in cross-validation
foldsoptions=( 5 10 )
# Cluster definitions to use
clusterdefs=( KM_AU06_r_AU12_r_5 )

for clusterdef in ${clusterdefs[@]}; do
	for infolder in ${infolders[@]}; do
		for folds in ${foldsoptions[@]}; do
			for seed in ${seeds[@]}; do
				echo "python3 clf.py --all_methods -i $basefolder/$clusterdef/$infolder -k $folds -s $seed"
				python3 clf.py --all_methods -i "$basefolder/$clusterdef/$infolder" -k "$folds" -s "$seed"
			done
		done
	done
done