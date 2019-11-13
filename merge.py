
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import argparse
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

import utils_fax.helpers_fax_truth as utils_fax

from pax_utils import utils_event as event_utils
from pax_utils import utils_s2integrals
from pax_utils import utils_interaction as interaction_utils
from pax_utils import utils_waveform_channels
from pax_utils import utils_waveform_summed


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_sarr(nEvents, n_samples_max):
    
    sArr = np.zeros(
        nEvents,
        dtype=[
            ('x_ins'       , np.float32), # Truth
            ('y_ins'       , np.float32),
            ('true_left'   , np.int32),
            ('true_right'  , np.int32),
            ('true_x'      , np.float32),
            ('true_y'      , np.float32),
            ('true_n_els'  , np.int32),
            ('true_n_phs'  , np.int32),
            ('left_index'  , np.int32),   # Reco
            ('left'        , np.int32),
            ('right'       , np.int32),
            ('x'           , np.float32),
            ('y'           , np.float32),
            ('z'           , np.float32),
            ('dt'          , np.float32),
            #('s2_truncated', np.int32),
            ('s2_area'     , np.float32),
            ('s2_areas'    , np.float16, 127),
            ('image'       , np.float16, (127, n_samples_max) ),
        ]
    )

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def fill_sarr(strArr2, df, idx_arr):

    assert(df is not None)
    
    idx_df = idx_arr 
    
    try:
        strArr2[idx_arr]['true_left']    = df.at[idx_df, 't_first_electron_true']
        strArr2[idx_arr]['true_right']   = df.at[idx_df, 't_last_photon_true']
        strArr2[idx_arr]['true_nels']    = df.at[idx_df, 'n_electrons_true']
        strArr2[idx_arr]['true_nphs']    = df.at[idx_df, 'n_photons_true']
        strArr2[idx_arr]['x_ins']        = df.at[idx_df, 'x_ins']
        strArr2[idx_arr]['y_ins']        = df.at[idx_df, 'y_ins']            
        strArr2[idx_arr]['x_true']       = df.at[idx_df, 'x_true']
        strArr2[idx_arr]['y_true']       = df.at[idx_df, 'y_true']
        strArr2[idx_arr]['x_s2']         = df.at[idx_df, 's2_x']
        strArr2[idx_arr]['y_s2']         = df.at[idx_df, 's2_y']
        strArr2[idx_arr]['s2_left']      = df.at[idx_df, 's2_left']
        strArr2[idx_arr]['s2_right']     = 2*(df.at[idx_df, 's2_center_time' ] - df.at[idx_df, 's2_left' ])
        strArr2[idx_arr]['index_left']   = min(strArr2[idx_arr]['true_left'] , strArr2[idx_arr]['s2_left'] )
        ##strArr2[idx_arr]['s2_truncated'] = num_s2_samples_truncated
        strArr2[idx_arr]['s2_area']      = df.at[idx_df, 's2_area']
        #strArr2[idx_arr]['image']        = arr2d_s2
        #strArr2[idx_arr]['s2_areas']     = arr_s2_areas
    
    except Exception as e:
        print("Exception!")
        print(e)
        
    return

            
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_dir(dir_in, dir_fmt, dir_out, isStrict):

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    dir_out_pkl = './'
    
    nEventsPerFileToProcess = 1000
    file_format             = 'XENON1T-0-000000000-000000999-000001000.zip'
    lst_contents            = glob.glob(dir_in + '/' + dir_fmt)
    lst_contents            = sorted(lst_contents)
    n_dir                   = len(lst_contents)
    nEvents                 = nEventsPerFileToProcess*n_dir
    
    print("\n{0} directories (matching '{1}') files found in:".format(n_dir, dir_fmt))
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
        

        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------

        n_pkl         = 1000
        n_zip         = 10
        
        strArr = init_sarr(n_pkl*n_zip, 1000)
        
        process_zip(zipfilename, isStrict, strArr)
        
        f_out = dir_out + '/strArr_dir{0}'.format(i_dir)
        
        print("Out: {0}".format(f_out))
        
        np.save(f_out, strArr)
        
        
        #df_zip_merged = processPklEvents(zipfilename, iZip, nEventsPerFileToProcess, dir_waveforms_s2)
        #zip_pkl       = dir_out_pkl + '/zip/' + 'zip%05d' % iZip + '.pkl'
        #df_zip_merged.to_pickle(zip_pkl)

    print("Done")
    
    return



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_zip(zipfilename, isStrict, strArr):
    
    n_pkl_per_zip   = 1000
    n_zip           = 10
    
    df_zip_merged   = pd.DataFrame()
    df_s2_waveforms = pd.DataFrame()
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    for iPkl in range(0, n_pkl_per_zip):
    
        df_pkl_merged = process_pkl(zipfilename, iPkl, isStrict)
        df_zip_merged = df_zip_merged.append(df_pkl_merged)

        assert(df_pkl_merged is not None)
        
        print(df_pkl_merged.columns)
        
        fill_sarr(strArr, df_pkl_merged, iPkl)
        
        continue
    
    df_zip_merged.reset_index(inplace=True, drop=True)
    df_zip_merged.to_pickle('df_merge.pkl')
   
    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_pkl(zipfilename, ipklfile, isStrict):
    
    if (ipklfile % 100 == 0):
        print("   Processing PKL file: {0}".format(ipklfile) )

    jsonfilename  = os.path.dirname(zipfilename) + '/pax_info.json'
    cfg           = event_utils.getConfig(jsonfilename)
    zfile         = zipfile.ZipFile(zipfilename)
    event         = pickle.loads(zlib.decompress(zfile.open(str(ipklfile)).read()))

    
    left  = 0
    right = 0
    
    if (event.main_s2):
        left  = event.main_s2.left
        right = event.main_s2.right
    else:
        f   = '/dali/lgrandi/dbarge/data-xe1t/s2only/fax/0/sim_s2s_minitrees.hdf5'
        dft = pd.read_hdf(f)
        
        s2_left   = dft.at[ipklfile, 's2_left']
        s2_center = dft.at[ipklfile, 's2_center_time']
        s2_width  = 2*(s2_center - s2_left)
        s2_right = s2_left + s2_width    
        
        #left  = min(true_left , s2_left )
        #right = min(max(true_right, s2_right)+1, window_left + n_samples_max)
        
        left  = s2_left
        right = s2_right
    
    
    df = process_evt(event, cfg, left, right, isStrict)
    
    return df


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_evt(event, cfg, left, right, isStrict=True, verbose=True):
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    interactions  = event.interactions
    nInteractions = len(interactions)
        
    #print("{0} interactions".format(nInteractions))
        
    #if (nInteractions < 1):
        #print("   No interactions!")
        #return
        
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
    # Get dataframe of S2 waveform for each PMT channel
    #----------------------------------------------------------------------
    
    df_channels_waveforms_top     = utils_waveform_channels.getChannelsWaveformsDataFrame(event, cfg, isStrict, False)
    df_channels_waveforms_top_all = utils_waveform_channels.addEmptyChannelsToDataFrame(df_channels_waveforms_top)
    
    
    #----------------------------------------------------------------------
    # Get summed S2 waveform from dataframe
    #----------------------------------------------------------------------
    
    length = event.length()
    arr_summed_waveform_top_df = utils_waveform_summed.getSummedWaveformFromDataFrame(df_channels_waveforms_top_all, length)
    arr_summed_waveform_top_df = arr_summed_waveform_top_df[left:right]
    
    
    #----------------------------------------------------------------------
    # Get summed S2 waveform PAX from event
    #----------------------------------------------------------------------

    wf_arrs_equal = False
    summedInfo    = True
    
    if (summedInfo):
        
        arr_summed_waveform_top_evt = utils_waveform_summed.GetSummedWaveformFromEvent(event)
        arr_summed_waveform_top_evt = arr_summed_waveform_top_evt[left:right]

     
        #------------------------------------------------------------------
        # Check that the S2 summed waveform from the event and dataframe are equal
        #------------------------------------------------------------------
    
        wf_arrs_equal = np.allclose(arr_summed_waveform_top_evt, arr_summed_waveform_top_df, atol=1e-1, rtol=1e-1)


    #----------------------------------------------------------------------
    # Check that the per-channel S2 integrals from the event and dataframe are equal
    #----------------------------------------------------------------------
    
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
    eq_s2_df_wf_ev = np.isclose(s2_sum_df , wf_sum_evt, atol=marg, rtol=marg)
    
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

    
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
        
    if (not wf_arrs_equal and summedInfo):
        print("Error! Summed waveform from dataframe & event not equal")

    if (not eq_wf_df_wf_ev):
        print("Error! Sum of summed waveform from dataframe & event not equal")

    if (not eq_s2_ev_wf_ev):
        print("Error! Sum of summed waveform & summed S2s from event not equal")

    if (not eq_s2_ev_wf_df):
        print("Error! Sum of summed waveform from dataframe & summed S2s from event not equal")
        
    if (not eq_s2_df_wf_df):
        print("Error! Sum of summed waveform from dataframe & summed S2s from dataframe not equal")

    if (not eq_s2_df_wf_ev):
        print("Error! Sum of summed waveform from event & summed S2s from dataframe not equal")
        
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


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def main():

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    args    = parse_arguments()
    dir_in  = args.dir_in
    dir_out = args.dir_out
    
    path, dirs, files = next(os.walk(dir_out))

    ex_out = len(files) > 0
    
    if (ex_out):
        print("\n*** Output directory: '" + dir_in + "' already exists and has contents. First delete directory. ***\n")
        assert(not ex_out)
    
    assert(os.path.exists(dir_in))

        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    lst_dir = glob.glob(dir_in + "[0-9]*")
    lst_dir.sort()
    lst_dir.sort(key = len) 
    f_hdf    = dir_in + 'data_new.hdf5' # 'data.hdf5'
    df       = pd.read_hdf(f_hdf)

    #import utils_fax.helpers_fax_truth

    df = utils_fax.nsToSamples(df, 's2_center_time')
    
    try:
        df = utils_fax.nsToSamples(df, 't_first_electron_true')
        df = utils_fax.nsToSamples(df, 't_last_electron_true')
        df = utils_fax.nsToSamples(df, 't_first_photon_true')
        df = utils_fax.nsToSamples(df, 't_last_photon_true')
    except Exception as e:
        print("Exception!")
        print(e)
        
    #dir_format = "instructions_" + ('[0-9]' * 6)
    dir_format = "sim_s2s"
        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    lst_dir = lst_dir[0:1]
    
    print("Input directories:\n")
    for x in lst_dir:
        print("   " + x)
        process_dir(x, dir_format, dir_out, False)
        
    print()
    
        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    return

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-dir_in'  , required=True)
    parser.add_argument('-dir_out' , required=True)
    parser.add_argument('-isStrict', default=True, type=lambda x: (str(x).lower() == 'true'))
    #parser.add_argument('-events_per_batch', required=True, type=int)

    return parser.parse_args()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):

    print("\nStarting...\n")
    
    t1 = time.time()
    
    main()
    
    t2 = time.time()
    dt = (t2 - t1)/60
    
    print("\nDone in {0:.1f} min".format(dt))
    
