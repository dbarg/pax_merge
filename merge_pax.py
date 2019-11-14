
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

import merge_looper as looper
import merge_process_event as process_event


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

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class mergePax():

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __init__(self):
        
        self.n_intr = 0
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
                
                s2_left   = event.s2_left
                s2_center = dft.at[ipklfile, 's2_center_time']
                s2_width  = 2*(s2_center - s2_left)
                s2_right  = s2_left + s2_width    
                
                left  = s2_left
                right = s2_right

                
            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            df_evt, df_chans = process_event.process_evt(event, cfg, left, right, i_zip, i_pkl, None, isStrict=True)
            df_merged        = df_merged.append(df_evt)

            
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
        df_merged.to_pickle('df_merge.pkl')
            
        #f_out = dir_out + '/strArr_dir{0}'.format(i_dir)
        f_out = './strArr_dir{0}'.format(i_zip)
        print("\nOutfile: {0}".format(f_out))
        np.save(f_out, strArr)
        

        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        return

   
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def main(self):
            
        #--------------------------------------------------------------------------
        # Parse Arguments
        #--------------------------------------------------------------------------
    
        args        = parse_arguments()
        dir_in      = args.dir_in
        dir_fmt     = args.dir_fmt
        zip_fmt     = args.zip_fmt
        self.n_intr = args.n_intr
        
            
        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------
            
        looper.looper(dir_in, dir_fmt, zip_fmt, self.zip_callback_pax)
        
        return

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-dir_in' , required=True)
    parser.add_argument('-dir_fmt', required=True)
    parser.add_argument('-zip_fmt', required=True)
    parser.add_argument('-n_intr' , required=True, type=int)

    return parser.parse_args()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):
    
    mrg = mergePax()
    mrg.main()
    
  