#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --output=out_tfrec_indvid.out

#SBATCH --mail-user=somnathsharmaji05@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN

echo Setting up environment
module load python/3.8
module load httpproxy

pwd
cd $SLURM_TMPDIR
virtualenv ./env
source ./env/bin/activate

pip install --no-index tensorflow

scp -r /home/s0mnaths/projects/def-skrishna/s0mnaths/datasets/Users/03253 .
echo untar done
ls 
cd $SLURM_TMPDIR/03253
ls
scp -r /home/s0mnaths/projects/def-skrishna/s0mnaths/model/create_tfrec_individuals.py .

echo copy done

echo begin tfrec conversion

python create_tfrec_individuals.py --filename 03253

echo DONE!

