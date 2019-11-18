
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

from pax_utils import utils_event
from pax_utils import utils_s2integrals
from pax_utils import utils_interaction as interaction_utils
from pax_utils import utils_waveform_channels
from pax_utils import utils_waveform_summed
from utils_fax import helpers_fax_truth

import merge_looper as looper


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class mergePax():

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __init__(self):
  
        #----------------------------------------------------------------------
        # Parse Arguments
        #----------------------------------------------------------------------
    
        args               = parse_arguments()
        self.dir_out       = args.dir_out
        self.dir_in        = args.dir_in
        self.dir_fmt       = args.dir_fmt
        self.zip_fmt       = args.zip_fmt
        self.n_intr        = args.n_intr
        self.isStrict      = args.isStrict
        self.n_samples_max = 1000
        
        self.x            = np.nan
        self.y            = np.nan
        self.true_x       = np.nan
        self.true_y       = np.nan
        self.true_left    = np.nan
        self.true_right   = np.nan
        self.left         = np.nan
        self.right        = np.nan
        self.width        = np.nan
        self.window_left  = np.nan
        self.window_right = np.nan
        self.window_width = np.nan
            
        
        #----------------------------------------------------------------------
        # Check output directory
        #----------------------------------------------------------------------
        
        p, d, files  = next(os.walk(self.dir_out))
        n_in_out_dir = len(files)
        
        dir_in_exists  = os.path.isdir(self.dir_out)
        dir_out_exists = os.path.isdir(self.dir_out)
        dir_out_empty  = (n_in_out_dir == 0)
      
        if (not dir_in_exists):
            print("Input directory '{0}' does not exist!".format(self.dir_in))
            
        if (not dir_out_exists):
            print("Output directory '{0}' does not exist!".format(self.dir_out))
            
        if (not dir_out_empty):
            print("Output directory '{0}' is not empty!".format(self.dir_out))
            print(n_in_out_dir)
            print(files)
            
        assert(dir_in_exists)
        assert(dir_out_exists)
        assert(dir_out_empty)
    

        #----------------------------------------------------------------------
        # Kludge
        #----------------------------------------------------------------------
    
        f_df_all   = self.dir_in + '/data_new.hdf5'
        self.df_all = pd.DataFrame()

        try:
            self.df_all = pd.read_hdf(f_df_all)
            self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 's2_center_time')
            self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_first_electron_true')
            self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_last_electron_true')
            self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_first_photon_true')
            self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_last_photon_true')
            #self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 's2_left') # This is already in sample units
        except Exception as ex:
            print(ex)
            
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        return
        

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def zip_callback_pax(self, zipname, i_dir, i_zip, verbose=False):
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        n_events_modulus = 10000
        jsonfilename     = os.path.dirname(zipname) + '/pax_info.json'
        cfg              = utils_event.getConfig(jsonfilename)
        zfile            = zipfile.ZipFile(zipname, 'r')
        zip_namelist     = zfile.namelist()
        n_pkl_per_zip    = len(zip_namelist)
        df_merged        = pd.DataFrame()
            
        assert(n_pkl_per_zip == 1000)
       
        strArr = np.zeros(
            n_pkl_per_zip,
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
                ('s2_area'     , np.float32),
                ('s2_areas'    , np.float16, 127),
                ('image'       , np.float16, (127, 1000) ),
                ('s2_n_truncated', np.int32),
                ('s2_area_truncated', np.float32),
                
            ]
        )
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        for i_pkl, pklfilename in enumerate(zip_namelist):
               
            #------------------------------------------------------------------
            #------------------------------------------------------------------

            n_zip_per_dir = 10
            i_glb         = i_dir*n_zip_per_dir*n_pkl_per_zip + i_zip*n_pkl_per_zip + i_pkl
            #i_arr         = i_zip*n_pkl_per_zip + i_pkl
            i_arr         = i_pkl

            if (i_pkl % 10 == 0):
                print("      PKL File: {0}".format(i_pkl))
                
            if(i_glb % n_events_modulus == 0 and i_glb != 0):
                print("   Save")
        

            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            pklfile = zfile.open(pklfilename)
            event   = pickle.loads(zlib.decompress(pklfile.read()))
            intrs   = event.interactions
            nIntr   = len(intrs)
            
            if (nIntr < self.n_intr):
                print("      -> Global event number:{0}, {1} interactions. Skipping...".format(i_glb, nIntr))
                continue
            #else:
            #    print("-> Global event number:{0}, interactions: {1}.".format(i_glb, nIntr))
            
            
            #------------------------------------------------------------------
            # Determine S2 Window
            #------------------------------------------------------------------
            
            if (event.main_s2):
                
                self.left         = event.main_s2.left
                self.right        = event.main_s2.right
                self.width        = self.right - self.left
                self.window_left  = self.left
                self.window_right = self.right
                self.window_width = self.width
                
            else:

                print("Using data from: '{0}'...".format(f_df_all))
                
                #event      = utils_event.getVerifiedEvent(event, verbose=False)
                #x          = self.df_all.at[i_glb, 's2_x']
                #y          = self.df_all.at[i_glb, 's2_y']
                #true_x     = self.df_all.at[i_glb, 'x_true']
                #true_y     = self.df_all.at[i_glb, 'y_true']
                #true_left  = self.df_all.at[i_glb, 't_first_electron_true']
                #true_right = self.df_all.at[i_glb, 't_last_photon_true']
                #true_nels  = self.df_all.at[i_glb, 'n_electrons_true']
                #true_nphs  = self.df_all.at[i_glb, 'n_photons_true']
                #true_width = true_right + 1 - true_left
                #s2_left    = self.df_all.at[i_glb, 's2_left']
                #s2_center  = self.df_all.at[i_glb, 's2_center_time']
                #s2_width   = 2*(s2_center - s2_left)
                #s2_right   = s2_left + s2_width   
                #window_left = min(true_left , s2_left )
                #window_right = min(max(true_right, s2_right)+1, window_left + self.n_samples_max)
                #window_width = window_right - window_left
                #assert(event.duration() == self.df_all.at[i_glb, 'event_duration'])
                #left  = window_left
                #right = window_right
                

            assert(self.left >= 0 and self.right >= self.left)


          
            
            
            #------------------------------------------------------------------
            # Load Data
            #------------------------------------------------------------------
        
            df_pkl_event              = utils_event.getEventDataFrameFromEvent(event)
            df_pkl_intr               = interaction_utils.getInteractionDataFrameFromEvent(event)
            df_pkl_s2s                = utils_s2integrals.getS2integralsDataFrame(event, 127)
            df_pkl                    = df_pkl_event.merge(df_pkl_intr).merge(df_pkl_s2s)
            df_pkl['event_number']    = event.event_number

                        
            #----------------------------------------------------------------------
            # Get summed S2 waveform PAX from event
            #----------------------------------------------------------------------
            
            arr_sum_wf_top_evt = utils_waveform_summed.GetSummedWaveformFromEvent(event)
            arr_sum_wf_top_evt = arr_sum_wf_top_evt[self.left:self.right]
            wf_sum_evt         = np.sum(arr_sum_wf_top_evt)
            arr_s2areas_evt    = df_pkl_s2s.iloc[0][1:].as_matrix().astype(np.float32)
            sum_s2_areas_evt   = np.sum(arr_s2areas_evt)
            
            
            #------------------------------------------------------------------
            # Get channels & summed waveform as dataframe
            #------------------------------------------------------------------
            
            df_chs_wfs_top     = utils_waveform_channels.getChannelsWaveformsDataFrame(event, cfg)
            arr_sum_wf_top_df  = utils_waveform_summed.getSummedWaveformFromDataFrame(
                df_chs_wfs_top,
                event.length()
            )
            arr_sum_wf_top_df = arr_sum_wf_top_df[self.left:self.right]
            wf_sum_df         = np.sum(arr_sum_wf_top_df)
  

            #------------------------------------------------------------------
            #------------------------------------------------------------------

            if (False):
                print()
                print("Total Sum, Evt: {0:.1f}".format(wf_sum_evt))
                print("Total Sum, S2s: {0:.1f}".format(sum_s2_areas_evt))
                print("Total Sum, Df:  {0:.1f}".format(wf_sum_df))

            
            #------------------------------------------------------------------
            # Sanity - Check Summed Waveform from the event & dataframe are equal
            # To Do: Check s2 areas from df
            #------------------------------------------------------------------
            
            marg = 1e-3
            
            assert(np.isclose(wf_sum_evt, sum_s2_areas_evt, atol=marg, rtol=marg))
            assert(np.isclose(wf_sum_evt, wf_sum_df       , atol=marg, rtol=marg))
            assert(np.allclose(arr_sum_wf_top_evt, arr_sum_wf_top_df, atol=marg, rtol=marg))
            

            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            df_merged = df_merged.append(df_pkl_event)
                

            
            
                
            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            arr2d = utils_waveform_channels.covertChannelWaveformsDataFrametoArray(df_chs_wfs_top, 0, event.length())

            arr2d_s2                         = np.zeros(shape=(127, self.n_samples_max))
            arr2d_s2[:, 0:self.window_width] = arr2d[:, self.window_left:self.window_right]
            #arr1d_s2                    = np.sum(arr2d_s2, axis=0)
            #arr1d_sum                   = np.sum(arr1d_s2) 

            
            #strArr[i_arr]['true_nels']    = true_nels
            #strArr[i_arr]['true_nphs']    = true_nphs
            #strArr[i_arr]['x_ins']        = df.at[idx_df, 'x_ins']
            #strArr[i_arr]['y_ins']        = df.at[idx_df, 'y_ins']    
            #strArr[i_arr]['s2_n_truncated'] = num_s2_samples_truncated
            #strArr[i_arr]['s2_area']      = s2_area
            strArr[i_arr]['left_index']   = self.window_left
            #strArr[i_arr]['true_left']    = self.true_left
            #strArr[i_arr]['true_right']   = self.true_right
            #strArr[i_arr]['true_x']       = self.true_x
            #strArr[i_arr]['true_y']       = self.true_y
            strArr[i_arr]['x']            = self.x
            strArr[i_arr]['y']            = self.y
            strArr[i_arr]['left']         = self.left
            strArr[i_arr]['right']        = self.right
            strArr[i_arr]['image']        = arr2d_s2
            #strArr[i_arr]['s2_areas']  = arr_s2_areas_arr_inwindow
            
            
            #------------------------------------------------------------------
            # Save Waveform Dataframes
            #------------------------------------------------------------------
            
            if (False):
                file_out_s2_waveforms = 's2s/event{0:07d}_S2waveforms.pkl'.format(event.event_number)
                df_chans.to_pickle(file_out_s2_waveforms)

            continue
            
            
        #----------------------------------------------------------------------
        # Save
        #----------------------------------------------------------------------
            
        df_merged.reset_index(inplace=True, drop=True)
            
        f_out_strArr = self.dir_out + '/strArr_dir{0}'.format(i_zip)
        f_out_df     = self.dir_out + '/df_merge_dir{0}'.format(i_zip)
        
        print()
        print("\nSaving dataframe (shape: {0}) to file: {1}".format(df_merged.shape, f_out_df))
        print()
        print("\nSaving structured array (shape: {0}) to file: {1}".format(strArr.shape, f_out_strArr))
        print()
        
        np.save(f_out_strArr, strArr)
        df_merged.to_pickle(f_out_df)
        

        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        return

   
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def main(self):
            
        looper.looper(
            self.dir_in,
            self.dir_fmt,
            self.zip_fmt,
            self.zip_callback_pax
        )
        
        return

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dir_out' , required=True)
    parser.add_argument('-dir_in'  , required=True)
    parser.add_argument('-dir_fmt' , required=True)
    parser.add_argument('-zip_fmt' , required=True)
    parser.add_argument('-n_intr'  , required=True, type=int)
    parser.add_argument('-isStrict', required=True, default=True, type=lambda x: (str(x).lower() == 'true'))

    return parser.parse_args()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):

    t1 = time.time()
    
    print("Starting...")
    mrg = mergePax()
    mrg.main()
    
    t2 = time.time()
    
    print("Done in {0:.1f}".format(t2-t1))
    
    
    