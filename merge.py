
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import glob
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import sys
import time
import zipfile
import zlib

from pax_utils import utils_event as event_utils
from pax_utils import utils_s2integrals
from pax_utils import utils_interaction as interaction_utils
from pax_utils import utils_waveform as waveform_utils
from pax_utils import utils_waveform2 as waveform_utils2


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_dir(dir_in):

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    dir_out_pkl = './'
    
    nEventsPerFileToProcess = 1000
    dir_format              = "instructions_" + ('[0-9]' * 6)
    file_format             = 'XENON1T-0-000000000-000000999-000001000.zip'
    lst_contents            = glob.glob(dir_in + '/' + dir_format)
    lst_contents            = sorted(lst_contents)
    n_dir                   = len(lst_contents)
    nEvents                 = nEventsPerFileToProcess*n_dir
    
    print("\n{0} directories files found in:".format(n_dir, dir_in))
    print("   {0}\n".format(dir_in))
    
    
    #------------------------------------------------------------------------------
    # output
    #------------------------------------------------------------------------------
    
    file_pkl         = dir_out_pkl   + 'merged_pax_' + str(nEvents % 1000) + '.pkl'
    dir_waveforms_s2 = dir_out_pkl   + '/waveforms_s2'
    
    ex = os.path.exists(dir_waveforms_s2)
    
    assert(not ex)
        
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    for i_dir in range(0, n_dir):
        
        zipfilename = lst_contents[i_dir] + '/' + file_format
        zip_pkl     = dir_out_pkl + '/zip/' + 'zip%05d' % i_dir + '.pkl'
    
        if (not os.path.exists(zipfilename)):
            
            print("Error! File: '" + str(zipfilename) + "' does not exist.")
        
            continue
            
        print("Input Zip File:  '" + zipfilename + "'")
        #print("Output PKL File: '" + zip_pkl + "'")
        
        process_zip(zipfilename)
        
        #df_zip_merged = processPklEvents(zipfilename, iZip, nEventsPerFileToProcess, dir_waveforms_s2)
        #zip_pkl       = dir_out_pkl + '/zip/' + 'zip%05d' % iZip + '.pkl'
        #df_zip_merged.to_pickle(zip_pkl)

    return



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_zip(zipfilename):
    
    n_pkl_per_zip   = 1000
    
    df_zip_merged   = pd.DataFrame()
    df_s2_waveforms = pd.DataFrame()
    
    for iPkl in range(0, n_pkl_per_zip):
    
        df_pkl_merged = process_pkl(zipfilename, iPkl)
        df_zip_merged = df_zip_merged.append(df_pkl_merged)

        continue
        
    df_zip_merged.reset_index(inplace=True, drop=True)
    df_zip_merged.to_pickle('df_merge.pkl')
   
    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_pkl(zipfilename, ipklfile):
    
    if (ipklfile % 100 == 0):
        print("   Processing PKL file: {0}".format(ipklfile) )

    jsonfilename  = os.path.dirname(zipfilename) + '/pax_info.json'
    cfg           = event_utils.getConfig(jsonfilename)
    zfile         = zipfile.ZipFile(zipfilename)
    event         = pickle.loads(zlib.decompress(zfile.open(str(ipklfile)).read()))

    process_evt(event, cfg)
    
    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_evt(event, cfg, verbose=True):
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    interactions  = event.interactions
    nInteractions = len(interactions)
        
    #print("{0} interactions".format(nInteractions))
        
    if (nInteractions < 1):
        #print("   No interactions!")
        return
        
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------

    df_pkl_event                  = event_utils.getEventDataFrameFromEvent(event)
    df_pkl_intr                   = interaction_utils.getInteractionDataFrameFromEvent(event)
    df_pkl_s2s                    = utils_s2integrals.getS2integralsDataFrame(event, 127)
    df_pkl_merged                 = df_pkl_event.merge(df_pkl_intr).merge(df_pkl_s2s)
    df_pkl_merged['event_number'] = event.event_number
    df_channels_waveforms_top     = pd.DataFrame()
    
    #if (nInteractions < 1):
    #    df_channels_waveforms_top.to_pickle(file_out_s2_waveforms)
        
    
    #----------------------------------------------------------------------
    # S2 Window
    #----------------------------------------------------------------------
    
    left  = event.main_s2.left
    right = event.main_s2.right

    
    #----------------------------------------------------------------------
    # Get summed S2 waveform PAX from event
    #----------------------------------------------------------------------
    
    arr_summed_waveform_top_evt = waveform_utils.GetSummedWaveformFromEvent(event)
    arr_summed_waveform_top_evt = arr_summed_waveform_top_evt[left:right]
    

    #----------------------------------------------------------------------
    # Get dataframe of S2 waveform for each PMT channel
    #----------------------------------------------------------------------
    
    df_channels_waveforms_top     = waveform_utils.getChannelsWaveformsDataFrame2(event, cfg, False)
    #df_channels_waveforms_top     = waveform_utils2.getChannelsWaveformsDataFrame(event, cfg, False)
    df_channels_waveforms_top_all = waveform_utils.addEmptyChannelsToDataFrame(df_channels_waveforms_top)
    
    
    #----------------------------------------------------------------------
    # Get summed S2 waveform from dataframe
    #----------------------------------------------------------------------
    
    arr_summed_waveform_top_df = waveform_utils.getSummedWaveformFromDataFrame(df_channels_waveforms_top_all, event)
    arr_summed_waveform_top_df = arr_summed_waveform_top_df[left:right]
    
     
    #----------------------------------------------------------------------
    # Check that the S2 summed waveform from the event and dataframe are equal
    #----------------------------------------------------------------------

    wf_arrs_equal = np.allclose(arr_summed_waveform_top_evt, arr_summed_waveform_top_df, atol=1e-1, rtol=1e-1)

    assert(wf_arrs_equal)
              
    
    #----------------------------------------------------------------------
    # Check that the per-channel S2 integrals from the event and dataframe are equal
    #----------------------------------------------------------------------
    
    #print(df_pkl_s2s.shape)
    
    arr_s2integrals_evt   = df_pkl_s2s.iloc[0][1:].as_matrix().astype(np.float32)
    arr_s2integrals_df    = df_channels_waveforms_top_all[:]['sum'].as_matrix().astype(np.float32)
    arr_s2integrals_diff  = arr_s2integrals_evt - arr_s2integrals_df
    arr_s2integrals_equal = np.allclose(arr_s2integrals_diff, np.zeros(arr_s2integrals_diff.size))
    
    
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    
    wf_sum_df  = np.sum(arr_summed_waveform_top_df)
    wf_sum_evt = np.sum(arr_summed_waveform_top_evt)
    s2_sum_df  = np.sum(arr_s2integrals_df)
    s2_sum_evt = np.sum(arr_s2integrals_evt)

    marg = 1e-1
    eq_wf_df_wf_ev = np.isclose(wf_sum_df , wf_sum_evt)
    eq_s2_ev_wf_ev = np.isclose(s2_sum_evt, wf_sum_evt, atol=marg, rtol=marg)
    eq_s2_ev_wf_df = np.isclose(s2_sum_evt, wf_sum_df , atol=marg, rtol=marg)
    eq_s2_df_wf_df = np.isclose(s2_sum_df , wf_sum_df , atol=marg, rtol=marg)
    eq_s2_df_wf_ev = np.isclose(s2_sum_df , wf_sum_evt , atol=marg, rtol=marg)
    
    if (not arr_s2integrals_equal and True):
        
        xmin = np.abs(np.amin(arr_s2integrals_diff))
        xmax = np.abs(np.amax(arr_s2integrals_diff))
        rmax = max(xmin, xmax)

        print("s2 int sum evt: {0}".format(s2_sum_evt))
        print("s2 int sum df:  {0}".format(s2_sum_df))
        print()
        print("event {0} Error! S2 Integrals not equal".format(event.event_number))
        print("   max diff: {0:.1f}".format(event.event_number, rmax))
        print()
        #print(arr_s2integrals_df)
        #print(arr_s2integrals_evt)

    
    assert(eq_wf_df_wf_ev)
    assert(eq_s2_ev_wf_ev)
    assert(eq_s2_ev_wf_df)
    assert(eq_s2_df_wf_df)
    assert(eq_s2_df_wf_ev)
    
    
    #----------------------------------------------------------------------
    # Check that the integrals of the S2 summed waveform from the event and dataframe are equal
    #----------------------------------------------------------------------
                                              
    verbose = False
    sane    = wf_arrs_equal and arr_s2integrals_equal
    
    if (verbose and not sane):
        
        print()
        #print("Event:                                   " + str(event_number))
        print("wf_arrs_equal:                           " + str(wf_arrs_equal))
        print("arr_s2integrals_equal:                   " + str(arr_s2integrals_equal))
        print()
    
                                              
    #----------------------------------------------------------------------
    # Save S2 Waveforms
    #----------------------------------------------------------------------
    
    file_out_s2_waveforms = 's2s/event{0:07d}_S2waveforms.pkl'.format(event.event_number)

    df_channels_waveforms_top.to_pickle(file_out_s2_waveforms)

    
    #----------------------------------------------------------------------
    # End loop on PKL files in ZIP File
    #----------------------------------------------------------------------

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
        print()
        print("Total:                  " + str(dt11_0))
        print()
    


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    return df_pkl_merged
