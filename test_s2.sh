#!/bin/bash

#SBATCH --job-name=merge
#SBATCH --account=pi-lgrandi
#SBATCH --partition=xenon1t,dali
#SBATCH --qos=xenon1t

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00

#SBATCH --output=log_s2.txt
#SBATCH --error=log_s2.txt

#cd ${HOME}/dali/xe1t-processing/pax_merge
cd /dali/lgrandi/dbarge/xe1t-processing/pax_merge
source ~/.bash/.setup_pax_head.sh

python ./merge_pax.py -dir_out ./temp_s2 -dir_in ../../data-xe1t/s2only/fax/ -dir_fmt [0-9]* -zip_fmt sim_s2s/*.zip -n_intr 0 -isStrict False -n_zip 10

