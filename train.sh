#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --output=test1.out

#SBATCH --mail-user=somnathsharmaji05@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN

echo Setting up environment
module load python/3.8
module load httpproxy

pwd

virtualenv ./env
source ./env/bin/activate

echo Installing TF!
pip install --no-index comet-ml tensorflow numpy matplotlib pillow tqdm pandas

echo begin training
cd model
python main.py --gpus 1 --epochs 2

echo DONE