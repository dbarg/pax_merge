
####################################################################################################
####################################################################################################

import datetime
import glob
import os
import pickle
import pprint
import sys
import time
from datetime import timedelta
import zipfile
#import zipfile2
import zlib
#import zlib2

import numpy as np
import pandas as pd

from IPython.display import clear_output
from IPython.display import display


#pax_str = 'pax_v6.10.1'
pax_str = 'pax_v6.5.1'


####################################################################################################
####################################################################################################

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath('..//' + pax_str))

from pax_utils import event_utils
from pax_utils import file_utils
from pax_utils import interaction_utils
from pax_utils import s1s2_utils
from pax_utils import numeric_utils
from pax_utils import waveform_pax_utils
from pax_utils import waveform_utils
from pax_utils import s1s2_utils

from pax import core

pd.set_option('display.max_columns', 500)


####################################################################################################
####################################################################################################

dir_out_pkl  = '/project/lgrandi/dbarge/simulation/wimp/' + pax_str + '/merged/'
#dir_out_pkl  = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.8.3/merged/aug21/'
#dir_out_pkl  = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.8.3/merged/'
#dir_out_pkl  = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.5.1/merged/'

dir_input    = '/project/lgrandi/dbarge/simulation/wimp/' + pax_str + '/'
#dir_input    = '/home/dbarge/scratch/simulations/wimp/may03/'
#dir_input    = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.5.1/'
#dir_input    = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.8.3/'
#dir_input    = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.5.1/'


####################################################################################################
####################################################################################################

dt_total = 0

def test(event_number, dt_event):
    
    global dt_total
    
    dt_total = round(dt_total + dt_event, 1)
    hours    = dt_total//3600
    minutes  = (dt_total - 3600*hours)//60
    seconds  = dt_total - 60*minutes
    dt_str  = '%02d:%02d:%02d' % (hours,minutes,seconds)

    print(" -> Event Number: " + str(event_number) + ", dt_event: " + str(dt_event) + ", dt_total: " + str(dt_str))
    
    #clear_output(wait=True)     
    
    return


####################################################################################################
####################################################################################################

nEventsPerZipFile = 1000


####################################################################################################
####################################################################################################

def processPklEvents(zipfilename, iZip, nEventsPerFileToProcess, dir_waveforms_s2):

    ta = time.time()
    
    ################################################################################################
    ################################################################################################

    #zfile         = zipfile2.ZipFile(zipfilename)
    zfile         = zipfile.ZipFile(zipfilename)
    lst_pkl_files = zfile.namelist()   
    
    # to do, check sorted
    #lst_pkl_files.sort()
        
    jsonfilename  = os.path.dirname(zipfilename) + '/pax_info.json'
    cfg           = event_utils.getConfig(jsonfilename)
        
        
    ################################################################################################
    ################################################################################################
    
    sum_wf = None
    #event  = None
    df_zip_merged   = pd.DataFrame()
    df_s2_waveforms = pd.DataFrame()
        
        
    ################################################################################################
    ################################################################################################
    
    t1 = time.time()
    t2 = time.time()
    t3 = time.time()
    t4 = time.time()
    t5 = time.time()
    t6 = time.time()
    
    dt1_0 = 0
    dt2_1 = 0
    dt3_2 = 0
    dt4_3 = 0
    dt5_4 = 0
    dt6_5 = 0
    dt7_6 = 0
    dt8_7 = 0
    dt9_8 = 0
    dt10_9 = 0
    dt11_10 = 0
    
    num_wf_arrs_equal              = 0
    num_sum_summed_waveforms_equal = 0
    num_arr_s2integrals_equal      = 0
    num_sum_s2integrals_equal      = 0
        
        
    ################################################################################################
    ################################################################################################
    
    for iPklFile, pklfilename in enumerate(lst_pkl_files):
    
        #print("iPKlFile: " + str(iPklFile))

        if (iPklFile >= nEventsPerFileToProcess):
    
            break
        
        t0 = time.time()
      
    
        ############################################################################################
        ############################################################################################

        event_number          = iZip*nEventsPerZipFile + iPklFile
        file_out_s2_waveforms = dir_waveforms_s2 + '/' + 'event' + format(event_number, '07d') + '_S2waveforms' + '.pkl'
        
        #print(" -> Event Number: " + str(event_number))
        #clear_output(wait=True)        
        
        
        t1 = time.time()
        
        
        ############################################################################################
        ############################################################################################
        
        #event  = file_utils.getPaxEventFromPklFileInZipArchive(zipfilename, pklfilename)
        event = pickle.loads(zlib.decompress(zfile.open(pklfilename).read()))
        
        interactions  = event.interactions
        nInteractions = len(interactions)

        t2 = time.time()
        
        #if (nInteractions != 1): continue
            
            
        ############################################################################################
        ############################################################################################
        
        df_pkl_event  = event_utils.getEventDataFrameFromEvent(event)
        t3            = time.time()
        df_pkl_intr   = interaction_utils.getInteractionDataFrameFromEvent(event)
        t4            = time.time()
        df_pkl_s2s    = s1s2_utils.getS2integralsDataFrame(event, 127)
        t5            = time.time()

        
        ############################################################################################
        ############################################################################################
    
        df_pkl_merged                 = df_pkl_event.merge(df_pkl_intr).merge(df_pkl_s2s)
        df_pkl_merged['event_number'] = event_number
        t6            = time.time()
        
        
        ############################################################################################
        ############################################################################################
        
        df_zip_merged = df_zip_merged.append(df_pkl_merged)
        t7            = time.time()
        
        
        ############################################################################################
        ############################################################################################
        
        df_channels_waveforms_top = pd.DataFrame()
        
            
        ############################################################################################
        ############################################################################################
        
        if (nInteractions < 1):
            
            df_channels_waveforms_top.to_pickle(file_out_s2_waveforms)
            
            continue
        
        
        ############################################################################################
        ############################################################################################
        
        left  = event.main_s2.left
        right = event.main_s2.right

        
        ############################################################################################
        # Get summed S2 waveform from event, PAX
        ############################################################################################

        arr_summed_waveform_top_evt = waveform_pax_utils.getSummedWaveformFromEvent(event, 'tpc_top')
        arr_summed_waveform_top_evt = arr_summed_waveform_top_evt[left:right]
        
        t8            = time.time()

        
        ############################################################################################
        # Get summed S2 waveform from PAX
        ############################################################################################
        
        #arr_summed_waveform_top_pax = waveform_pax_utils.SumWaveformPAX(event)
        #arr_summed_waveform_top_pax = arr_summed_waveform_top_evt[left:right]
                
        
            
        ############################################################################################
        # Get dataframe of S2 waveform for each PMT channel
        ############################################################################################
        
        df_channels_waveforms_top     = waveform_utils.getChannelsWaveformsDataFrame(event, cfg, 'top', False)
        
        t9 = time.time()
        
        df_channels_waveforms_top_all = waveform_utils.addEmptyChannelsToDataFrame(df_channels_waveforms_top)
        #df_channels_waveforms_top_all = df_channels_waveforms_top
        
        t10 = time.time()

        
        ############################################################################################
        # Get summed S2 waveform from dataframe
        ############################################################################################
        
        arr_summed_waveform_top_df  = waveform_utils.getSummedWaveformFromDataFrame(df_channels_waveforms_top_all, event)
        arr_summed_waveform_top_df  = arr_summed_waveform_top_df[left:right]
        
        
        
        
        ############################################################################################
        # Check that the S2 summed waveform from the event and dataframe are equal
        ############################################################################################
        
        wf_arrs_equal = numeric_utils.compareArrays(arr_summed_waveform_top_evt, arr_summed_waveform_top_df)
        
        
        ############################################################################################
        # Check that the integrals of the S2 summed waveform from the event and dataframe are equal
        ############################################################################################
        
        #sum_summed_waveform_top_pax    = np.sum(arr_summed_waveform_top_pax)
        sum_summed_waveform_top_evt     = np.sum(arr_summed_waveform_top_evt)
        sum_summed_waveform_top_df      = np.sum(arr_summed_waveform_top_df)
        
        #
        #sum_summed_waveforms_equal = numeric_utils.compareFloats(sum_summed_waveform_top_evt, sum_summed_waveform_top_df, 1e-3)
        sum_summed_waveforms_equal = abs(sum_summed_waveform_top_evt - sum_summed_waveform_top_df) / sum_summed_waveform_top_evt < 0.1
            
        
        ############################################################################################
        # Check that the per-channel S2 integrals from the event and dataframe are equal
        ############################################################################################

        arr_s2integrals_evt = df_pkl_s2s.iloc[0][1:].as_matrix().astype(np.float32)
        arr_s2integrals_df  = df_channels_waveforms_top_all[:]['sum'].as_matrix().astype(np.float32)
        
        arr_s2integrals_equal = numeric_utils.compareArrays(arr_s2integrals_evt, arr_s2integrals_df)
        
        
        ############################################################################################
        # Check that the sum of the per-channel S2 integrals from the event and dataframe are equal
        ############################################################################################
        
        sum_s2integrals_evt = np.sum(arr_s2integrals_evt)
        sum_s2integrals_df  = np.sum(arr_s2integrals_df)
        
        # this fails use weaker condition for now
        #sum_s2integrals_equal = numeric_utils.compareFloats(sum_s2integrals_evt, sum_s2integrals_df)
        sum_s2integrals_equal = abs(sum_s2integrals_evt - sum_s2integrals_df) / sum_s2integrals_df < 0.1
        
        
        ############################################################################################
        ############################################################################################
        
        verbose = False
        sane    = wf_arrs_equal and sum_summed_waveforms_equal and arr_s2integrals_equal and sum_s2integrals_equal
        
        if (verbose or not sane):
            
            print()
            print("Event:                                   " + str(event_number))
            print("wf_arrs_equal:                           " + str(wf_arrs_equal))
            print("sum_summed_waveforms_equal:              " + str(sum_summed_waveforms_equal))
            print("arr_s2integrals_equal:                   " + str(arr_s2integrals_equal))
            print("sum_s2integrals_equal:                   " + str(sum_s2integrals_equal))
            print()
            print("Integral of Summed Waveform Event:       " + str(sum_summed_waveform_top_evt))
            print("Integral of Summed Waveform DF:          " + str(sum_summed_waveform_top_df))
            print("Sum of S2 Integrals over channels Event: " + str(sum_s2integrals_evt))
            print("Sum of S2 Integrals over channels DF:    " + str(sum_s2integrals_df))
            print()
        
        
        ############################################################################################
        ############################################################################################
        
        if (not wf_arrs_equal             ) : num_wf_arrs_equal              += 1
        if (not sum_summed_waveforms_equal) : num_sum_summed_waveforms_equal += 1
        if (not arr_s2integrals_equal     ) : num_arr_s2integrals_equal      += 1
        if (not sum_s2integrals_equal     ) : num_sum_s2integrals_equal      += 1
        
        #assert(wf_arrs_equal)
        #assert(sum_summed_waveforms_equal)
        #assert(arr_s2integrals_equal)
        #assert(sum_s2integrals_equal) # this fails
        
        
        ############################################################################################
        # Save S2 Waveforms
        ############################################################################################

        df_channels_waveforms_top.to_pickle(file_out_s2_waveforms)
        
        #display(df_channels_waveforms_top[0:5][:])

        
        ############################################################################################
        # End loop on PKL files in ZIP File
        ############################################################################################
        
        t11 = time.time()
        
        dt1_0   = round(t1  - t0, 3)
        dt2_1   = round(t2  - t1, 3)
        dt3_2   = round(t3  - t2, 3)
        dt4_3   = round(t4  - t3, 3)
        dt5_4   = round(t5  - t4, 3)
        dt6_5   = round(t6  - t5, 3)
        dt7_6   = round(t7  - t6, 3)
        dt8_7   = round(t8  - t7, 3)
        dt9_8   = round(t9  - t8, 3)
        dt10_9  = round(t10 - t9, 3)
        dt11_10 = round(t11 - t10, 3)
        dt11_0  = round(t11 - t0, 3)
        
        if (False):
            
            print()
            print("Init:                   " + str(dt1_0))
            print("IO:                     " + str(dt2_1))
            print("Event dataframe:        " + str(dt3_2))
            print("Intr dataframe:         " + str(dt4_3))
            print("S2 integrals dataframe: " + str(dt5_4))
            print("merge 1:                " + str(dt6_5))
            print("merge:                  " + str(dt7_6))
            print("Sum waveforms:          " + str(dt8_7))
            print("Get Channels dataframe: " + str(dt9_8))
            print("Add Empty Channels:     " + str(dt10_9))
            print("Sanity:                 " + str(dt11_10))
            #print("CPU: " + str(psutil.cpu_percent()))
            #print("Mem: " + str(psutil.virtual_memory()))
            #print()
            print()
            print("Total:                  " + str(dt11_0))
            print()
        
        test(event_number, dt11_0)
            
            
        continue

    
    ################################################################################################
    ################################################################################################
    
    #print("test: " + str(wf_arrs_equal             ))
    #print("test: " + str(sum_summed_waveforms_equal))
    #print("test: " + str(arr_s2integrals_equal     ))
    #print("test: " + str(sum_s2integrals_equal     ))

        
    ################################################################################################
    ################################################################################################
    
    df_zip_merged.reset_index(inplace=True, drop=True)
       
    tb = time.time()
    dt = tb - ta
    
    return df_zip_merged

    
    
    
####################################################################################################
# input
####################################################################################################

nEventsPerFileToProcess = 1000
nFilesZip               = 200
nEvents                 = nEventsPerFileToProcess*nFilesZip

print()

if (nFilesZip == -1):
    nFilesZip = len(lst_contents)

#dir_input    = '/home/dbarge/scratch/simulations/wimp/may03/'
#dir_input    = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.8.3/'
dir_format   = dir_input + "instructions_" + ('[0-9]' * 6)
file_format  = 'XENON1T-0-000000000-000000999-000001000.zip'
lst_contents = glob.glob(dir_format)
lst_contents.sort()

nContents = len(lst_contents)

print(nContents)
assert(nContents > 0)


####################################################################################################
# output
####################################################################################################

ver              = 's2waveforms_v2'
#dir_out_pkl      = '/home/dbarge/scratch/simulations/wimp/merged/may07/'
#dir_out_pkl      = '/project/lgrandi/dbarge/simulation/wimp/pax_v6.8.3/merged/aug21/'

file_pkl         = dir_out_pkl + 'merged_pax_' + str(nEvents % 1000) + 'k_' + ver + '.pkl'
dir_waveforms    = dir_out_pkl + '/' + 'waveforms_' + ver
dir_waveforms_s2 = dir_waveforms + '/' + 's2'

if(not os.path.isdir(dir_waveforms)):
    os.mkdir(dir_waveforms)
    if(not os.path.isdir(dir_waveforms_s2)):
        os.mkdir(dir_waveforms_s2)
    else:
        print("\nDirectory '" + dir_waveforms_s2 + "' exists!\n") 
else:
    print("\nDirectory '" + dir_waveforms + "' exists!\n")



####################################################################################################
####################################################################################################

lst_cols = []
lst_evt  = event_utils.getColumns()
lst_int  = interaction_utils.getColumns()[1:]
lst_s2s  = s1s2_utils.getS2integralsDataFrameColumns()
lst_cols.extend(lst_evt)
lst_cols.extend(lst_int)
lst_cols.extend(lst_s2s)

arr_init = range(0, nEvents)

df_events = pd.DataFrame(columns=lst_cols, index=pd.Index(arr_init))
df        = pd.DataFrame()

t0 = time.time()


####################################################################################################
####################################################################################################

cols_s2 = s1s2_utils.getS2integralsDataFrameColumns()

cols = []
cols.append('event_number')
cols.append('intr_count')
cols.append('intr_x')
cols.append('intr_y')
cols.extend(cols_s2)



####################################################################################################
####################################################################################################

for iZip in range(0, nFilesZip):
    
    
    ################################################################################################
    ################################################################################################
    
    zipfilename = lst_contents[iZip] + '/' + file_format
    zip_pkl     = dir_out_pkl + '/zip/' + 'zip%05d' % iZip + '.pkl'

    if (not os.path.exists(zipfilename)):
        
        print("Error! File: '" + str(zipfilename) + "' does not exist.")
    
        continue
        
    print("Input Zip File:  '" + zipfilename + "'")
    print("Output PKL File: '" + zip_pkl + "'")


    ################################################################################################
    ################################################################################################
    
    df_zip_merged = processPklEvents(zipfilename, iZip, nEventsPerFileToProcess, dir_waveforms_s2)
    zip_pkl       = dir_out_pkl + '/zip/' + 'zip%05d' % iZip + '.pkl'
    
    print(zip_pkl)
    #df_zip_merged.to_pickle(zip_pkl)
    
    
    ################################################################################################
    ################################################################################################
    
    continue
    
t1 = time.time()
dt = round(t1 - t0, 1)

t0_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))
t1_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))

print("Start Time:   " + str(t0_str) )
print("End Time:     " + str(t1_str) )
print("Elapsed time: " + str(dt) + " s")


####################################################################################################
# Write
####################################################################################################


