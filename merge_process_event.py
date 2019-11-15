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

def process_evt(event, cfg, left, right, i_zip, ipklfile, n_intr, isStrict, verbose=True):
   
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

    df_pkl_event              = utils_event.getEventDataFrameFromEvent(event)
    df_pkl_intr               = interaction_utils.getInteractionDataFrameFromEvent(event)
    df_pkl_s2s                = utils_s2integrals.getS2integralsDataFrame(event, 127)
    df_pkl                    = df_pkl_event.merge(df_pkl_intr).merge(df_pkl_s2s)
    df_pkl['event_number']    = event.event_number
    df_channels_waveforms_top = pd.DataFrame()
        
    
    #----------------------------------------------------------------------
    # Get dataframe of S2 waveform for top PMT channels
    #----------------------------------------------------------------------
    
    df_channels_waveforms_top     = utils_waveform_channels.getChannelsWaveformsDataFrame(event, cfg, isStrict, False)
    df_channels_waveforms_top_all = utils_waveform_channels.addEmptyChannelsToDataFrame(df_channels_waveforms_top)
    #df_channels_waveforms_top     = df_channels_waveforms_top[left:right]
    #df_channels_waveforms_top_all = df_channels_waveforms_top_all[left:right]
    
    
    #----------------------------------------------------------------------
    # Get summed S2 waveform from dataframe
    #----------------------------------------------------------------------
    
    length                     = event.length()
    arr_summed_waveform_top_df = utils_waveform_summed.getSummedWaveformFromDataFrame(df_channels_waveforms_top_all, length)
    arr_summed_waveform_top_df = arr_summed_waveform_top_df[left:right]
    wf_sum_df                  = np.sum(arr_summed_waveform_top_df)
    
    
    #----------------------------------------------------------------------
    # Get summed S2 waveform PAX from event
    #----------------------------------------------------------------------
    
    arr_summed_waveform_top_evt = utils_waveform_summed.GetSummedWaveformFromEvent(event)
    arr_summed_waveform_top_evt = arr_summed_waveform_top_evt[left:right]
    wf_sum_evt                  = np.sum(arr_summed_waveform_top_evt)

    
    
    
    #------------------------------------------------------------------
    # Sanity - Check Summed Waveform from the event & dataframe are equal
    #------------------------------------------------------------------
    
    marg = 1e-3
    
    arr_diff_wf_evt_df = arr_summed_waveform_top_evt - arr_summed_waveform_top_df
    eq_wf_evt_df       = np.allclose(arr_diff_wf_evt_df, np.zeros(arr_diff_wf_evt_df.size), rtol=marg, atol=marg)
    
    max_diff_wf_evt_df = np.amax(arr_diff_wf_evt_df)
    
    if not (eq_wf_evt_df):
        print("   Summed Waveforms max diff: {0:.1f}".format(max_diff_wf_evt_df))
        
    diff_wfsum_evt_df = wf_sum_evt - wf_sum_df
    
    if (not np.isclose(diff_wfsum_evt_df, 0, rtol=marg, atol=marg)):
        print("   Sum of Summed Waveforms diff: {0:.1f}".format(diff_wfsum_evt_df))    
        print("   Sum of Waveform, evt:         {0:.1f}".format(wf_sum_evt))    
        pct = diff_wfsum_evt_df/wf_sum_evt
        print(pct)
        

    #----------------------------------------------------------------------
    # Check that the per-channel S2 integrals from the event and dataframe are equal
    #----------------------------------------------------------------------
    
    arr_s2integrals_evt = df_pkl_s2s.iloc[0][1:].as_matrix().astype(np.float32)
    arr_s2integrals_df  = None
    
    if (event.main_s2):
        arr_s2integrals_df = df_channels_waveforms_top_all[:]['sum'].as_matrix().astype(np.float32)
    else:
        arr_s2integrals_df = utils_waveform_summed.getS2IntegralsFromDataFrame(df_channels_waveforms_top)

    s2_sum_evt = np.sum(arr_s2integrals_evt)
    s2_sum_df  = np.sum(arr_s2integrals_df)

    assert(s2_sum_df > 0)
    #assert(s2_sum_evt > 0)

    if (s2_sum_evt > 0):
        
        print("Sum S2 integrnals evt: {0}".format(s2_sum_evt))
        print("Sum S2 integrnals df:  {0}".format(s2_sum_df))

        assert(np.allclose(arr_s2integrals_evt, arr_s2integrals_df, atol=marg, rtol=marg))
        assert(np.isclose(s2_sum_evt, s2_sum_df, atol=marg, rtol=marg))

        
        #------------------------------------------------------------------
        #------------------------------------------------------------------
        
        assert(np.isclose( abs(s2_sum_evt - wf_sum_evt), 0, atol=marg, rtol=marg))
        assert(np.isclose( abs(s2_sum_evt - wf_sum_df) , 0, atol=marg, rtol=marg))

        
        
    #----------------------------------------------------------------------
    # End loop on PKL files in ZIP File
    #----------------------------------------------------------------------

    return df_pkl, df_channels_waveforms_top
