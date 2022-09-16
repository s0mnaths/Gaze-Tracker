#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=test1.out

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

echo Installing TF!
pip install --no-index comet-ml tensorflow numpy matplotlib pillow tqdm pandas

ls
scp -r /home/s0mnaths/projects/def-skrishna/s0mnaths .
echo copy done

echo begin training
cd $SLURM_TMPDIR/s0mnaths/model
python main.py --gpus 1 --epochs 1

echo DONE