#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import numpy as np
import os
import pandas as pd
import pprint
import sys
import time

#import utils_fax.helpers_fax_truth as utils_fax

from pax_utils import utils_event
from pax_utils import utils_s2integrals
from pax_utils import utils_interaction as interaction_utils
from pax_utils import utils_waveform_channels
from pax_utils import utils_waveform_summed


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_evt(event, cfg, left, right, i_zip, ipklfile, n_intr, strArr, isStrict=True, verbose=True):
   
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    interactions  = event.interactions
    nInteractions = len(interactions)
        

    if (nInteractions < n_intr):
        print("   No interactions! Skipping...")
        #return df_pkl, df_channels_waveforms_top        
        return

    
    #----------------------------------------------------------------------
    # Load Data
    #----------------------------------------------------------------------

    df_pkl_event                  = utils_event.getEventDataFrameFromEvent(event)
    df_pkl_intr                   = interaction_utils.getInteractionDataFrameFromEvent(event)
    df_pkl_s2s                    = utils_s2integrals.getS2integralsDataFrame(event, 127)
    df_pkl                        = df_pkl_event.merge(df_pkl_intr).merge(df_pkl_s2s)
    df_pkl['event_number']        = event.event_number
    df_channels_waveforms_top     = pd.DataFrame()
        
    
    #----------------------------------------------------------------------
    # Get dataframe of S2 waveform for top PMT channels
    #----------------------------------------------------------------------
    
    df_channels_waveforms_top     = utils_waveform_channels.getChannelsWaveformsDataFrame(event, cfg, isStrict, False)
    df_channels_waveforms_top_all = utils_waveform_channels.addEmptyChannelsToDataFrame(df_channels_waveforms_top)
    
    
    #----------------------------------------------------------------------
    # Get summed S2 waveform from dataframe
    #----------------------------------------------------------------------
    
    arr_summed_waveform_top_df = utils_waveform_summed.getSummedWaveformFromDataFrame(
        df_channels_waveforms_top_all, event.length())
    arr_summed_waveform_top_df = arr_summed_waveform_top_df[left:right]
    wf_sum_df                  = np.sum(arr_summed_waveform_top_df)
    
    
    #----------------------------------------------------------------------
    # Get summed S2 waveform PAX from event
    #----------------------------------------------------------------------

    wf_sum_evt    = 0
    wf_arrs_equal = False
    wf_sum_diff   = None
    sum_sum_diff  = None
    s2_sum_diff   = None
    summedInfo    = True
    
    if (summedInfo):
        
        arr_summed_waveform_top_evt = utils_waveform_summed.GetSummedWaveformFromEvent(event)
        arr_summed_waveform_top_evt = arr_summed_waveform_top_evt[left:right]

     
        #------------------------------------------------------------------
        # Check that the S2 summed waveform from the event and dataframe are equal
        #------------------------------------------------------------------

        wf_sum_diff   = arr_summed_waveform_top_evt - arr_summed_waveform_top_df
        wf_arrs_equal = np.allclose(wf_sum_diff, np.zeros(arr_summed_waveform_top_evt.size), atol=1e-1, rtol=1e-1)
        wf_sum_evt    = np.sum(arr_summed_waveform_top_evt)
        sum_sum_diff  = abs(wf_sum_df - wf_sum_evt)

        
    #----------------------------------------------------------------------
    # Check that the per-channel S2 integrals from the event and dataframe are equal
    #----------------------------------------------------------------------
    
    arr_s2integrals_evt = df_pkl_s2s.iloc[0][1:].as_matrix().astype(np.float32)
    arr_s2integrals_df  = None
    
    if (event.main_s2):
        arr_s2integrals_df = df_channels_waveforms_top_all[:]['sum'].as_matrix().astype(np.float32)
    else:
        arr_s2integrals_df = utils_waveform_summed.getS2IntegralsFromDataFrame(df_channels_waveforms_top)

    assert(np.sum(arr_s2integrals_df) > 0)
    
    arr_s2integrals_diff  = arr_s2integrals_evt - arr_s2integrals_df
    arr_s2integrals_equal = np.allclose(arr_s2integrals_diff, np.zeros(arr_s2integrals_diff.size))
    
    
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    
    s2_sum_df  = np.sum(arr_s2integrals_df)
    s2_sum_evt = np.sum(arr_s2integrals_evt)

    marg = 1e-1
    
    eq_wf_df_wf_ev = np.isclose(sum_sum_diff,            0, atol=marg, rtol=marg)
    eq_s2_ev_wf_ev = np.isclose(s2_sum_evt  , wf_sum_evt  , atol=marg, rtol=marg)
    eq_s2_ev_wf_df = np.isclose(s2_sum_evt  , wf_sum_df   , atol=marg, rtol=marg)
    eq_s2_df_wf_df = np.isclose(s2_sum_df   , wf_sum_df   , atol=marg, rtol=marg)
    eq_s2_df_wf_ev = np.isclose(s2_sum_df   , wf_sum_evt  , atol=marg, rtol=marg)
    
    if (not arr_s2integrals_equal and False):
        
        xmin = np.abs(np.amin(arr_s2integrals_diff))
        xmax = np.abs(np.amax(arr_s2integrals_diff))
        rmax = max(xmin, xmax)

        print("s2 int sum evt: {0}".format(s2_sum_evt))
        print("s2 int sum df:  {0}".format(s2_sum_df))
        print()
        print("event {0} Error! S2 Integrals not equal".format(event.event_number))
        print("   max diff: {0:.1f}".format(event.event_number, rmax))
        print()

    
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    
    evt_s2_integrals_exist = False
    
    if (not wf_arrs_equal and summedInfo):
        print("   Error! Summed waveform from dataframe & event not equal")
        print(np.amax(np.abs(wf_sum_diff)))

    if (not eq_wf_df_wf_ev and summedInfo):
        print("   Error! Sum of summed waveform from dataframe & event not equal.")
        print("      max difference: {0:.1f}".format(sum_sum_diff))
        
    if (not eq_s2_df_wf_ev and summedInfo):
        print("   Error! Sum of summed waveform from event & summed S2s from dataframe not equal")
        print(abs(wf_sum_evt - s2_sum_df))
        
    if (not eq_s2_ev_wf_ev and evt_s2_integrals_exist):
        print("   Error! Sum of summed waveform & summed S2s from event not equal")

    if (not eq_s2_ev_wf_df):
        print("   Error! Sum of summed waveform from dataframe & summed S2s from event not equal")
        
    #if (not eq_s2_df_wf_df):
    #    print("   Error! Sum of summed waveform from dataframe & summed S2s from dataframe not equal")

    #assert(wf_arrs_equal)
    #assert(eq_wf_df_wf_ev)
    #assert(eq_s2_ev_wf_ev)
    #assert(eq_s2_ev_wf_df)
    #assert(eq_s2_df_wf_df)
    #assert(eq_s2_df_wf_ev)
    
    
    #----------------------------------------------------------------------
    # Check that the integrals of the S2 summed waveform from the event and dataframe are equal
    #----------------------------------------------------------------------
                                              
    verbose = False
    sane    = wf_arrs_equal and arr_s2integrals_equal
    
    if (verbose and not sane):
        
        print()
        #print("Event:                 " + str(event_number))
        print("wf_arrs_equal:         " + str(wf_arrs_equal))
        print("arr_s2integrals_equal: " + str(arr_s2integrals_equal))
        print()
       
    
    #----------------------------------------------------------------------
    # End loop on PKL files in ZIP File
    #----------------------------------------------------------------------

    return df_pkl, df_channels_waveforms_top
