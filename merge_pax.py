
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
from utils_fax import helpers_fax_truth

import merge_looper as looper
import merge_process_event as process_event


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class mergePax():

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __init__(self):
  
 
        #--------------------------------------------------------------------------
        # Parse Arguments
        #--------------------------------------------------------------------------
    
        args          = parse_arguments()
        self.dir_out  = args.dir_out
        self.dir_in   = args.dir_in
        self.dir_fmt  = args.dir_fmt
        self.zip_fmt  = args.zip_fmt
        self.n_intr   = args.n_intr
        self.isStrict = args.isStrict

        
        #--------------------------------------------------------------------------
        # Check output directory
        #--------------------------------------------------------------------------
        
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
    

        #--------------------------------------------------------------------------
        # Kludge
        #--------------------------------------------------------------------------
    
        f_df_all = self.dir_in + '/data_new.hdf5'

        self.df_all = pd.read_hdf(f_df_all)
        self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 's2_left')
        self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 's2_center_time')
        self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_first_electron_true')
        self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_last_electron_true')
        self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_first_photon_true')
        self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_last_photon_true')

        print(self.df_all.at[0, 's2_left'])

               
            
        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------

        return
        

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def zip_callback_pax(self, zipname, i_dir, i_zip, verbose=False):
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        jsonfilename  = os.path.dirname(zipname) + '/pax_info.json'
        cfg           = utils_event.getConfig(jsonfilename)
        zfile         = zipfile.ZipFile(zipname, 'r')
        zip_namelist  = zfile.namelist()
        n_pkl_per_zip = len(zip_namelist)
        df_merged     = pd.DataFrame()
            
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
                #('s2_truncated', np.int32),
            ]
        )
        
        
        n_events_modulus = 10000
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        for i_pkl, pklfilename in enumerate(zip_namelist):
               
            #------------------------------------------------------------------
            #------------------------------------------------------------------

            n_zip_per_dir = 10
            i_glb         = i_dir*n_zip_per_dir*n_pkl_per_zip + i_zip*n_pkl_per_zip + i_pkl

            if (i_glb % 100 == 0):
                print("   i_glb: {0}".format(i_glb))
          
            if(i_glb % n_events_modulus == 0 and i_glb != 0):
                print("   Save")
        
            if (i_pkl % 100 == 0):
                print("   PKL File: {0}".format(i_pkl))

        
            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            pklfile = zfile.open(pklfilename)
            event   = pickle.loads(zlib.decompress(pklfile.read()))
            intrs   = event.interactions
            nIntr   = len(intrs)
            
            
            #------------------------------------------------------------------
            #------------------------------------------------------------------

            if (nIntr < self.n_intr):
                print("-> Global event number:{0}, interactions: {1}. Skipping...".format(i_glb, nIntr))
                continue
            
            
            #------------------------------------------------------------------
            # Determine S2 Window
            #------------------------------------------------------------------
            
            left  = -1
            right = -1
            
            if (event.main_s2):
                
                left  = event.main_s2.left
                right = event.main_s2.right
                
            else:
                
                true_left  = self.df_all.at[i_glb, 't_first_electron_true']
                true_right = self.df_all.at[i_glb, 't_last_photon_true']
                true_nels  = self.df_all.at[i_glb, 'n_electrons_true']
                true_nphs  = self.df_all.at[i_glb, 'n_photons_true']
                s2_left    = self.df_all.at[i_glb, 's2_left']
                s2_center  = self.df_all.at[i_glb, 's2_center_time']
                s2_width   = 2*(s2_center - s2_left)
                s2_right   = s2_left + s2_width   
                
                print(self.df_all.at[0, 's2_left'])
                print(i_glb)
                print(self.df_all.at[i_glb, 's2_left'])
                
 

                true_width  = true_right + 1 - true_left
                window_left = min(true_left , s2_left )
                #window_right = min(max(true_right, s2_right)+1, window_left + n_samples_max)
                #window_width = window_right - window_left
            
                assert(event.duration() == self.df_all.at[i_glb, 'event_duration'])
            
                left  = s2_left
                right = s2_right
                

            assert(left >= 0 and right >= left)

            print()
            print(i_glb)
            print("true left:  {0}".format(true_left))
            print("true right: {0}".format(true_right))
            print("true width: {0}".format(true_width))
            print()
            print("left:  {0}".format(s2_left))
            print("right: {0}".format(s2_right))
            print("width: {0}".format(s2_width))
            print()
                
                
            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            df_evt, df_chans = process_event.process_evt(
                event,
                cfg,
                left,
                right,
                i_zip,
                i_pkl,
                self.n_intr,
                None,
                isStrict=self.isStrict
            )
            
            df_merged = df_merged.append(df_evt)

            
            #------------------------------------------------------------------
            # Save Waveform Dataframes
            #------------------------------------------------------------------
            
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
        print("\nSaving dataframe (shape: {0}) to file: {1}".format(df_merged.shape, f_out))
        print()
        print("\nSaving structured array (shape: {0}) to file: {1}".format(strArr.shape, f_out))
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
    
    mrg = mergePax()
    mrg.main()
    
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def fill_sarr(strArr2, df, idx_arr):

    return 

    #df = utils_fax.nsToSamples(df, 't_first_electron_true')
    #df = utils_fax.nsToSamples(df, 't_last_electron_true')
    #df = utils_fax.nsToSamples(df, 't_first_photon_true')
    #df = utils_fax.nsToSamples(df, 't_last_photon_true')
    #df = utils_fax.nsToSamples(df, 's2_center_time')
        
        
    #assert(df is not None)
    #
    #idx_df = idx_arr 
    #
    #try:
    #    strArr2[idx_arr]['true_left']    = df.at[idx_df, 't_first_electron_true']
    #    strArr2[idx_arr]['true_right']   = df.at[idx_df, 't_last_photon_true']
    #    strArr2[idx_arr]['true_nels']    = df.at[idx_df, 'n_electrons_true']
    #    strArr2[idx_arr]['true_nphs']    = df.at[idx_df, 'n_photons_true']
    #    strArr2[idx_arr]['x_ins']        = df.at[idx_df, 'x_ins']
    #    strArr2[idx_arr]['y_ins']        = df.at[idx_df, 'y_ins']            
    #    strArr2[idx_arr]['x_true']       = df.at[idx_df, 'x_true']
    #    strArr2[idx_arr]['y_true']       = df.at[idx_df, 'y_true']
    #except Exception as e:
    #    print("Exception Filling Truth!")
    #    print(e)
    #    
    #try:
    #    strArr2[idx_arr]['x_s2']         = df.at[idx_df, 's2_x']
    #    strArr2[idx_arr]['y_s2']         = df.at[idx_df, 's2_y']
    #    strArr2[idx_arr]['s2_left']      = df.at[idx_df, 's2_left']
    #    strArr2[idx_arr]['s2_right']     = 2*(df.at[idx_df, 's2_center_time' ] - df.at[idx_df, 's2_left' ])
    #    strArr2[idx_arr]['s2_area']      = df.at[idx_df, 's2_area']
    #except Exception as e:
    #    print("Exception Filling Reco!")
    #    print(e)
    #    
#
    #    #strArr2[idx_arr]['index_left']   = min(strArr2[idx_arr]['true_left'] , strArr2[idx_arr]['s2_left'] )
    #    ##strArr2[idx_arr]['s2_truncated'] = num_s2_samples_truncated
    #    #strArr2[idx_arr]['s2_area']      = df.at[idx_df, 's2_area']
    #    #strArr2[idx_arr]['image']        = arr2d_s2
    #    #strArr2[idx_arr]['s2_areas']     = arr_s2_areas
    #  
    #    
    #return

    
  