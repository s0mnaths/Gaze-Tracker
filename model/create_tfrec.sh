#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --output=test.out

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

echo begin untar
tar -xf  /home/s0mnaths/projects/def-skrishna/s0mnaths/datasets/gt_fin_gs.tar.gz -C .

echo untar done
ls 
cd $SLURM_TMPDIR/gt_fin_gs
ls
scp -r /home/s0mnaths/projects/def-skrishna/s0mnaths/model/create_tfrec.py .

echo copy done
pwd
ls

echo begin tfrec conversion

python create_tfrec.py

echo DONE!