#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import numpy as np
import os
import pandas as pd
import pickle
import sys


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

file_inst = '/home/dbarge/pax_instructions/merged_instructions_files.pkl'
dir_input = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.8.3/merged/'
file_pax  = dir_input + 'merged_zip_200000.pkl'
file_all  = dir_input + 'merged_all_200000.pkl'


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

nInst = 2000

df_inst = pd.read_pickle(file_inst)[:][0:nInst].reset_index(drop=True).rename(columns={'instruction': 'event_number'}).copy()
df_pax  = pd.read_pickle(file_pax).reset_index(drop=True)

df_inst['event_number'] = df_inst.index
df_pax ['event_number'] = df_pax.index

print(df_inst.shape)
print(df_pax.shape)

nInst = df_inst.shape[0]
nPax  = df_pax.shape[0]

if (nInst != nPax):
    print("Error! number of pax events (" +str(nPax) + ") not equal to number of instructions (" + str(nInst) + ")")

    
#------------------------------------------------------------------------------
# Save
#------------------------------------------------------------------------------

df_all = pd.merge(df_inst, df_pax, on='event_number')
df_all.to_pickle(file_all)
df_test = pd.read_pickle(file_all)
nEvents = df_test.shape[0]

print(file_all)
print()
print("MergedEvents: " + str(nEvents))
#display(df_test[:][0:5])
print()

 
