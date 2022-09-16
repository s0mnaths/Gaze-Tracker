#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=test1.out

#SBATCH --mail-user=somnathsharmaji05@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=ALL


echo Setting up environment
module load python/3.8
module load httpproxy

pwd
virtualenv ./env
source ./env/bin/activate

echo Installing TF!
pip install --no-index tensorflow numpy matplotlib pillow comet_ml tqdm pandas

echo installed!
python comet-test.py

echo DONE!
