
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

class mergePax():

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __init__(self):
        return
        

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def zip_callback_pax(self, zipname, i_dir, i_zip):
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        jsonfilename  = os.path.dirname(zipname) + '/pax_info.json'
        cfg           = utils_event.getConfig(jsonfilename)
        zfile         = zipfile.ZipFile(zipname, 'r')
        zip_namelist  = zfile.namelist()
        n_zip         = len(zip_namelist)
        df_merged     = pd.DataFrame()
            
        assert(n_zip == 1000)
        
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        for i_pkl, pkl in enumerate(zip_namelist):
           
            if (i_pkl % 100 == 0):
                print("   PKL File: {0}".format(i_pkl))

            event = pickle.loads(zlib.decompress(zfile.open(str(i_pkl)).read()))

            continue
            
            intrs = event.interactions
            nIntr = len(intrs)
            
            
            if (nIntr < 1):
                continue
        
            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            left  = -1
            right = -1
            
            if (event.main_s2):
                
                left  = event.main_s2.left
                right = event.main_s2.right
            
            else:
                
                left = event.s2_left

                print()
                
            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            df_evt, df_chans = process_event.process_evt(event, cfg, left, right, i_zip, i_pkl, None, isStrict=True)
        
            df_merged = df_merged.append(df_evt)

            
            #------------------------------------------------------------------
            # Save Waveform Dataframes
            #------------------------------------------------------------------
            
            file_out_s2_waveforms = 's2s/event{0:07d}_S2waveforms.pkl'.format(event.event_number)
            df_chans.to_pickle(file_out_s2_waveforms)

            continue
            
            
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
            
        df_merged.reset_index(inplace=True, drop=True)
        df_merged.to_pickle('df_merge.pkl')
        
        return

   
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def main(self):
            
        #--------------------------------------------------------------------------
        # Parse Arguments
        #--------------------------------------------------------------------------
    
        args    = parse_arguments()
        dir_in  = args.dir_in
        dir_fmt = args.dir_fmt
        zip_fmt = args.zip_fmt
        
            
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

    return parser.parse_args()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):
    
    mrg = mergePax()
    mrg.main()
    
  