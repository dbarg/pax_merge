#!/bin/bash

#SBATCH --job-name=merge
#SBATCH --account=pi-lgrandi
#SBATCH --partition=xenon1t,dali
#SBATCH --qos=xenon1t

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00

#SBATCH --output=log_pax.txt
#SBATCH --error=log_pax.txt

cd ${HOME}/dali/xe1t-processing/pax_merge
source ~/.bash/.setup_pax_head.sh

#python ./merge_pax.py -dir_out ./temp_pax -dir_in ../../data-xe1t/pax2019-11-14/zip/ -dir_fmt instructions_[0-9]* -zip_fmt *.zip -n_intr 1 -isStrict True
python ./merge_pax.py -dir_out ./temp_pax -dir_in ../../data-xe1t/pax2019-11-14/zip/ -dir_fmt instructions_00000[0-2] -zip_fmt *.zip -n_intr 1 -isStrict True
