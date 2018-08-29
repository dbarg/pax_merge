
####################################################################################################
####################################################################################################

import os
import pickle
import sys

import numpy as np
import pandas as pd

#sys.path.append("../")
#from pax_utils import pax_utils


####################################################################################################
####################################################################################################

#dir_input = '/home/dbarge/scratch/simulations/wimp/merged/may07/'
file_inst = '/home/dbarge/pax_instructions/merged_instructions_files.pkl'

dir_input = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.8.3/merged/'
#file_inst = '/project/lgrandi/dbarge/pax_instructions/merged_instructions_files.pkl'

file_pax  = dir_input + 'merged_zip_200000.pkl'
file_all  = dir_input + 'merged_all_200000.pkl'


####################################################################################################
####################################################################################################

nInst = 200000
#nInst = 2000

df_inst = pd.read_pickle(file_inst)[:][0:nInst].reset_index(drop=True).rename(columns={'instruction': 'event_number'}).copy()
df_pax  = pd.read_pickle(file_pax).reset_index(drop=True)

df_inst['event_number'] = df_inst.index
df_pax ['event_number'] = df_pax.index

rows = 1

print(df_inst.shape)
print(df_pax.shape)

#display(df_inst[0:rows][:])
#display(df_pax[0:rows][:])

nInst = df_inst.shape[0]
nPax  = df_pax.shape[0]

if (nInst != nPax):
    
    print("Error! number of pax events (" +str(nPax) + ") not equal to number of instructions (" + str(nInst) + ")") 
    
####################################################################################################
####################################################################################################

df_all = pd.merge(df_inst, df_pax, on='event_number')

print(file_all)

df_all.to_pickle(file_all)
 

 
####################################################################################################
####################################################################################################

df_test = pd.read_pickle(file_all)
nEvents = df_test.shape[0
                       ]
print()
print("MergedEvents: " + str(nEvents))
#display(df_test[:][0:5])
print()

 
