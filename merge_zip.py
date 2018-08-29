
####################################################################################################
####################################################################################################

import datetime
import os
import pandas as pd
import pickle
import sys
import time


print()

####################################################################################################
# Merge zip files
####################################################################################################

#dir_base      = '/home/dbarge/scratch/simulations/wimp/merged/may07/'
dir_base      = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.8.3/merged/'
dir_in_zip    = dir_base + 'zip/'
lst_files_zip = os.listdir(dir_in_zip)
lst_files_zip.sort()
#lst_files_zip = lst_files_zip[0:80]

df_merged_all = pd.DataFrame()


####################################################################################################
####################################################################################################

for file_i, filename in enumerate(lst_files_zip):

    t1            = time.time()
    file_zip_in   = dir_in_zip + filename
    df_zip        = pd.read_pickle(file_zip_in)
    df_merged_all = df_merged_all.append(df_zip)
    t2            = time.time()
    dt21          = round(t2 - t1, 2)
    

    print(file_zip_in)

    ###############################################################################################
    ###############################################################################################
    
    continue

    
####################################################################################################
####################################################################################################
    
nRows = len(df_merged_all.index)

file_out_all  = dir_base + 'merged_zip_' + str(nRows) + '.pkl'

df_merged_all.to_pickle(file_out_all)


####################################################################################################
####################################################################################################
    
print()
print(df_merged_all.shape)
#display(df_merged_all[0:5][:])
print()


