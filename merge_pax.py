
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import argparse
import glob
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import re
import sys
import time
import zipfile
import zlib

from pax_utils import utils_event
from pax_utils import utils_interaction as interaction_utils
from pax_utils import utils_waveform
from utils_fax import helpers_fax_truth

import merge_looper as looper


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class mergePax():

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def __init__(self):

        #self.n_zip_per_dir = 10 # S2 only
  

        #----------------------------------------------------------------------
        # Parse Arguments
        #----------------------------------------------------------------------
    
        args               = parse_arguments()
        self.n_zip_per_dir = args.n_zip
        self.dir_out       = args.dir_out
        self.dir_in        = args.dir_in
        self.dir_fmt       = args.dir_fmt
        self.zip_fmt       = args.zip_fmt
        self.n_intr        = args.n_intr
        self.isStrict      = args.isStrict
        self.idx_arr       = 0
        self.idx_out       = 0
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        self.n_samples_max = 1000
        self.n_pkl_per_zip = 1000
        
        self.init_data(self.n_pkl_per_zip, n_channels=127, n_samples=1000)
            
        
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
        #----------------------------------------------------------------------

        return
        

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def zip_callback_pax(self, zipname, i_dir, i_zip, verbose=False):
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------

        n_events_modulus = 1000
        jsonfilename     = os.path.dirname(zipname) + '/pax_info.json'
        cfg              = utils_event.getConfig(jsonfilename)
        zfile            = zipfile.ZipFile(zipname, 'r')
        zip_namelist     = zfile.namelist()
        n_pkl_per_zip    = len(zip_namelist)
        df_merged        = pd.DataFrame()
        faxfile          = re.sub(r'/zip/.*', '', os.path.abspath(zipname))
        faxfile          += '/fax_truth/fax_truth_{0:05d}.csv'.format(i_dir)

        dt21, dt32, dt43, dt54, dt65, dt76, dt87, dt98 = (0 for i in range(8))
        
        assert(n_pkl_per_zip == 1000)
       
        
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        for i_pkl, pklfilename in enumerate(zip_namelist):
               
            #------------------------------------------------------------------
            #------------------------------------------------------------------

            t1      = time.time()
            i_glb   = i_dir*self.n_zip_per_dir*n_pkl_per_zip + i_zip*n_pkl_per_zip + i_pkl
            pklfile = zfile.open(pklfilename)
            event   = pickle.loads(zlib.decompress(pklfile.read()))
            intrs   = event.interactions
            nIntr   = len(intrs)
            trueS2  = True
            t2      = time.time()
            dt21    += t2 - t1
            
            if (False and i_pkl == 0):
                
                print("i_pkl: {0}".format(i_pkl))
                print("i_zip: {0}".format(i_zip))
                print("i_dir: {0}".format(i_dir))
                print("i_glb: {0}".format(i_glb))
                print("i_glb: {0}".format(i_glb))
                
            
            #------------------------------------------------------------------
            #------------------------------------------------------------------

            df_fax = None
            
            try:
                
                df_fax = pd.read_csv(faxfile)
                df_fax = df_fax[df_fax['event']     == i_pkl]
                df_fax = df_fax[df_fax['peak_type'] == 's2' ]

                df_fax = helpers_fax_truth.nsToSamples(df_fax, 't_first_electron')
                df_fax = helpers_fax_truth.nsToSamples(df_fax, 't_last_electron')
                df_fax = helpers_fax_truth.nsToSamples(df_fax, 't_first_photon')
                df_fax = helpers_fax_truth.nsToSamples(df_fax, 't_last_photon')
                rows   = df_fax.shape[0] 
                
                if (rows == 0):
                    trueS2 = False
                elif (rows == 1):
                    df_fax          = df_fax.iloc[0,:]
                    self.true_x     = df_fax.loc['x']
                    self.true_y     = df_fax.loc['y']
                    self.true_z     = df_fax.loc['z']
                    self.true_left  = df_fax.loc['t_first_photon']
                    self.true_right = df_fax.loc['t_last_photon']
                    self.true_nels  = df_fax.loc['n_electrons']
                    self.true_nphs  = df_fax.loc['n_photons']
                    self.true_width = self.true_right - self.true_left
                else:
                    trueS2 = False
                    print(rows)
                    
            except Exception as ex:
                
                if (verbose):
                    print("      Exception! Could not load FAX data file:\n         {0}".format(ex))
            
            
            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            if (i_pkl % 100 == 0):
                print("      PKL File {0}: {1}".format(i_pkl, pklfilename))
            
            if (nIntr < self.n_intr or not trueS2):
                if (verbose):
                    print("      -> Global event number:{0}, {1} interactions. Skipping...".format(i_glb, nIntr))
                continue


            #------------------------------------------------------------------
            # Determine S2 Window
            #------------------------------------------------------------------
            
            t3            = time.time()
            dt32          += t3 - t2
            self.duration = event.duration()
            
            if (event.main_s2):

                pax_nn   = [x for x in event.main_s2.reconstructed_positions if x.algorithm == 'PosRecNeuralNet']
                pax_tpf  = [x for x in event.main_s2.reconstructed_positions if x.algorithm == 'PosRecTopPatternFit']
                pax_tpff = [x for x in event.main_s2.reconstructed_positions if x.algorithm == 'PosRecTopPatternFunctionFit']
                
                assert(len(pax_nn)   == 1)
                assert(len(pax_tpf)  == 1)
                assert(len(pax_tpff) == 1)
                    
                pax_nn = pax_nn[0]
                    
                self.left         = event.main_s2.left
                self.right        = event.main_s2.right
                self.s2_area      = event.main_s2.area
                self.s2_aft       = event.main_s2.area_fraction_top
                self.s2_area_top  = self.s2_aft*self.s2_area
                self.width        = self.right - self.left
                self.window_left  = self.left
                self.window_right = self.right
                self.window_width = self.width
                self.x            = pax_nn.x
                self.y            = pax_nn.y
                self.z            = pax_nn.z
                self.event        = event.event_number
                
            else:
                
                if (verbose):
                    print("      Using data from: '{0}'...".format(self.f_df_all))
                
                event             = utils_event.getVerifiedEvent(event, verbose=False)
                
                self.x            = self.df_all.at[i_glb, 's2_x']
                self.y            = self.df_all.at[i_glb, 's2_y']
                self.true_x       = self.df_all.at[i_glb, 'x_true']
                self.true_y       = self.df_all.at[i_glb, 'y_true']
                self.true_left    = self.df_all.at[i_glb, 't_first_electron_true']
                self.true_right   = self.df_all.at[i_glb, 't_last_photon_true']
                self.true_nels    = self.df_all.at[i_glb, 'n_electrons_true']
                self.true_nphs    = self.df_all.at[i_glb, 'n_photons_true']
                self.true_width   = self.true_right - self.true_left
                self.left         = self.df_all.at[i_glb, 's2_left']
                self.s2_center    = self.df_all.at[i_glb, 's2_center_time']
                self.s2_width     = 2*(self.s2_center - self.left)
                self.right        = self.left + self.s2_width   
                self.s2_area      = self.df_all.at[i_glb, 's2_area']
                self.s2_aft       = self.df_all.at[i_glb, 's2_area_fraction_top']
                self.s2_area_top  = self.s2_aft*self.s2_area
                self.window_left  = min(self.true_left, self.left )
                self.window_right = min(max(self.true_right, self.right), self.window_left + self.n_samples_max)
                self.window_width = self.window_right - self.window_left
                
                assert(event.duration() == self.df_all.at[i_glb, 'event_duration'])

                #self.left  = self.window_left
                #self.right = self.window_right
                

            assert(self.left >= 0 and self.right >= self.left)


            #------------------------------------------------------------------
            # Load Data
            #------------------------------------------------------------------
        
            t4                     = time.time()
            dt43                   += t4 - t3
            df_pkl                 = utils_event.getEventDataFrameFromEvent(event)
            df_pkl['event_number'] = event.event_number
            df_pkl_intr            = interaction_utils.getInteractionDataFrameFromEvent(event)
            df_pkl_s2s             = utils_waveform.getS2integralsDataFrame(event, 127)
            df_pkl.merge(df_pkl_intr).merge(df_pkl_s2s)
            t5                     = time.time()
            dt54                   += t5 - t4

         
            #----------------------------------------------------------------------
            # Get summed S2 waveform PAX from event
            #----------------------------------------------------------------------
            
            arr_sum_wf_top_evt = utils_waveform.GetSummedWaveformFromEvent(event)
            arr_sum_wf_top_evt = arr_sum_wf_top_evt[self.left:self.right]
            wf_sum_evt         = np.sum(arr_sum_wf_top_evt)
            arr_s2areas_evt    = np.zeros(127)
            if (event.main_s2):
                arr_s2areas_evt = event.main_s2.area_per_channel[0:127]
            sum_s2_areas_evt   = np.sum(arr_s2areas_evt)
            t6                     = time.time()
            dt65                   += t6 - t5
            
            
            #------------------------------------------------------------------
            # Get channels & summed waveform as dataframe
            #------------------------------------------------------------------
            
            df_chs_wfs_top    = utils_waveform.getChannelsWaveformsDataFrame(event, cfg)
            arr_sum_wf_top_df = utils_waveform.getSummedWaveformFromDataFrame(df_chs_wfs_top, event.length())
            arr_sum_wf_top_df = arr_sum_wf_top_df[self.left:self.right]
            wf_sum_df         = np.sum(arr_sum_wf_top_df)
            #s2area_df         = utils_waveform.getS2areaFromDataFrame(df_chs_wfs_top , event.length(), self.left, self.right)
            arr_s2areas_df    = utils_waveform.getS2areasFromDataFrame(df_chs_wfs_top, event.length(), self.left, self.right)
            sum_s2_areas_df   = np.sum(arr_s2areas_df)
            t7                = time.time()
            dt76              += t7 - t6
            
            
            #------------------------------------------------------------------
            # Truncate to fixed S2 Width
            #------------------------------------------------------------------
            
            arr2d    = utils_waveform.covertChannelWaveformsDataFrametoArray(df_chs_wfs_top, 0, event.length())
            arr2d_s2 = np.zeros(shape=(127, self.n_samples_max))
            
            idx_max  = min(self.window_right, self.window_left + self.n_samples_max)
            arr2d_s2[:, 0:self.window_width] = arr2d[:, self.window_left:idx_max]
            
            #if (self.window_width <= self.n_samples_max):
            #    arr2d_s2[:, 0:self.window_width] = arr2d[:, self.window_left:self.window_right]
            #else:
            #    arr2d_s2[:, 0:self.window_width] = arr2d[:, self.window_left:self.window_left+self.n_samples_max]
            
            t8                     = time.time()
            dt87                   += t8 - t7
                
            
            #------------------------------------------------------------------
            # Sanity
            #------------------------------------------------------------------
            
            if (True):
                
                i0 = self.window_left + self.n_samples_max
                i1 = max(self.right, i0)
                
                #arr2d_added_left        = arr2d[:, self.window_left:self.left]
                #arr2d_trunc_right       = arr2d[:, i0:i1]
                #arr_s2areas_to_subtract = np.sum(arr2d_added_left , axis=1)
                #arr_s2areas_to_add      = np.sum(arr2d_trunc_right, axis=1)
                #s2area_to_subtract      = np.sum(arr_s2areas_to_subtract)
                #s2area_to_add           = np.sum(arr_s2areas_to_add)
                #assert(
                #    np.allclose(arr_s2areas_df, np.sum(arr2d_s2, axis=1) - arr_s2areas_to_subtract + arr_s2areas_to_add)
                #)
                
                
                
            #------------------------------------------------------------------
            # Sanity - Check Summed Waveform from the event & dataframe are equal
            # To Do: Check s2 areas from df
            # To Do: Deal with truncated data
            #------------------------------------------------------------------
            
            marg      = 1e-1
            diff1     = wf_sum_evt - wf_sum_df   
            pct       = max(diff1/wf_sum_df, diff1/wf_sum_evt)
            arr_diff2 = arr_sum_wf_top_evt - arr_sum_wf_top_df
            
            eq1 = np.isclose(diff1, 0, atol=marg, rtol=marg) and (pct < 1e-1)
            eq2 = np.allclose(arr_diff2, np.zeros(arr_diff2.size), atol=marg, rtol=marg)
            eq3 = np.isclose(self.s2_area_top, sum_s2_areas_df, atol=marg, rtol=marg)
            eq4 = np.isclose(wf_sum_evt, sum_s2_areas_df , atol=marg, rtol=marg)
            eq5 = np.isclose(self.s2_area_top, wf_sum_evt, atol=marg, rtol=marg)
            
            if (not eq1):
                
                print("\n-> Sum of Summed waveform unequal for evt:{0} & df: {1}. Diff: {2}, Pct: {3}\n".format(
                    wf_sum_evt, wf_sum_df, diff1, pct))
                          
            assert(pct < 1e-1)
                
            if (not eq2):
                print()
                print("NOT eq2")
                print(np.amin(arr_diff2))
                print(np.amax(arr_diff2))
                print(wf_sum_evt)
                print(wf_sum_df)

            #if (not eq3):
            #    print("\nError! Event {0}: S2 area NOT EQUAL to Sum of S2 areas.".format(event.event_number))
            #    #print("   S2 Area (evt):        {0:.3f}".format(self.s2_area))
            #    print("   True Els:             {0}".format( self.true_nels ))
            #    print("   S2 Area Top (evt):    {0:.1f}".format(self.s2_area_top))
            #    #print("   S2 Area Top (df):     {0:.1f}".format(s2area_df))
            #    print("   Sum of S2 Areas (df): {0:.1f}".format(sum_s2_areas_df))
            #    print("   Sum of Sum WF (evt):  {0:.1f}".format(wf_sum_evt))
            #    print("   Sum of Sum WF (df):   {0:.1f}".format(wf_sum_df))

            if (not eq4):
                print("NOT eq4")
                
            #if (not eq5):
            #    print("\nError! S2 area NOT EQUAL to Sum of Sum Waveform from event.")
            #    print("   event: {0}".format(event.event_number))
            #    print("   S2 Area Top:         {0:.1f}".format(self.s2_area_top))
            #    print("   S2 AFT:              {0:.2f}".format(self.s2_aft))
            #    print("   Sum of Sum WF (evt): {0:.1f}".format(wf_sum_evt))
            #    print("   Sum of Sum WF (df):  {0:.1f}".format(wf_sum_df))
            #    print("   Sum of S2 Areas:     {0:.1f}".format(sum_s2_areas_evt))
                
            assert(eq1)
            assert(eq2)
            #assert(eq3)
            assert(eq4)
            #assert(eq5)
            
            t9                     = time.time()
            dt98                   += t9 - t8

            
            #------------------------------------------------------------------
            # Save Waveform Dataframes
            #------------------------------------------------------------------

            if(self.idx_out % n_events_modulus == 0 and self.idx_out != 0):
            
                f_idx        = int(self.idx_out/n_events_modulus) - 1
                f_out_strArr = self.dir_out + '/strArr{0}'.format(f_idx)
                
                assert(f_idx >= 0)
                
                print("\nSaving structured array (shape: {0}) to file: {1}".format(self.strArr.shape, f_out_strArr))
                print("\n idx_glb={0}, idx_out={1}, idx_arr={2}".format(i_glb, self.idx_arr, self.idx_out))

                self.idx_arr = 0
                np.savez(f_out_strArr, self.strArr)
            
            self.fill_strArr(self.idx_arr, arr2d_s2, arr_s2areas_evt)
            self.idx_arr += 1
            self.idx_out += 1
            
            
            #------------------------------------------------------------------
            # Save Waveform Dataframes
            #------------------------------------------------------------------
            
            if (False):
                df_merged = df_merged.append(df_pkl)
                file_out_s2_waveforms = 's2s/event{0:07d}_S2waveforms.pkl'.format(event.event_number)
                df_chans.to_pickle(file_out_s2_waveforms)


            #------------------------------------------------------------------
            # End loop on PKL files
            #------------------------------------------------------------------

            #break
            continue
            
        
        #----------------------------------------------------------------------
        # Performance
        #----------------------------------------------------------------------

        if (True):
            print()
            print("Zip: {0:.1f} s".format(dt21))
            print("FAX: {0:.1f} s".format(dt32))
            print("DF:  {0:.1f} s".format(dt54))
            print("PAX: {0:.1f} s".format(dt65))
            print("DF:  {0:.1f} s".format(dt76))
            print("Arr: {0:.1f} s".format(dt87))
            print("San: {0:.1f} s".format(dt98))

        
        #----------------------------------------------------------------------
        # Save
        #----------------------------------------------------------------------

        f_out_df = self.dir_out + '/df_merge_dir{0}.hdf'.format(i_zip)
        
        print("\nSaving dataframe (shape: {0}) to file: {1}".format(df_merged.shape, f_out_df))
        
        df_merged.reset_index(inplace=True, drop=True)
        df_merged.to_pickle(f_out_df)


        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        
        return

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def fill_strArr(self, i_arr, arr2d_s2, arr_s2areas_evt):

        #self.strArr[i_arr]['x_ins']        = df.at[idx_df, 'x_ins']
        #self.strArr[i_arr]['y_ins']        = df.at[idx_df, 'y_ins']  
        self.strArr[i_arr]['idx_out']        = self.idx_out
        self.strArr[i_arr]['event']          = self.event
        self.strArr[i_arr]['duration']       = self.duration
        self.strArr[i_arr]['true_nels']      = self.true_nels
        self.strArr[i_arr]['true_nphs']      = self.true_nphs
        self.strArr[i_arr]['true_left']      = self.true_left
        self.strArr[i_arr]['true_right']     = self.true_right
        self.strArr[i_arr]['true_x']         = self.true_x
        self.strArr[i_arr]['true_y']         = self.true_y
        self.strArr[i_arr]['x']              = self.x
        self.strArr[i_arr]['y']              = self.y
        self.strArr[i_arr]['left']           = self.left
        self.strArr[i_arr]['right']          = self.right
        self.strArr[i_arr]['left_index']     = self.window_left
        self.strArr[i_arr]['image']          = arr2d_s2.astype(np.float16)
        self.strArr[i_arr]['s2_areas']       = arr_s2areas_evt
        self.strArr[i_arr]['s2_area']        = self.s2_area
        self.strArr[i_arr]['s2_aft']         = self.s2_aft
        self.strArr[i_arr]['s2_area_top']    = self.s2_area_top
        #self.strArr[i_arr]['s2_n_truncated'] = num_s2_samples_truncated
        
        return
    
        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def init_data(self, n_rows, n_channels=127, n_samples=1000):
        
        self.idx_out      = 0
        self.event        = 0
        self.duration     = np.nan
        self.x            = np.nan
        self.y            = np.nan
        self.true_x       = np.nan
        self.true_y       = np.nan
        self.true_z       = np.nan
        self.true_left    = np.nan
        self.true_right   = np.nan
        self.true_nels    = np.nan
        self.true_nphs    = np.nan
        self.left         = np.nan
        self.right        = np.nan
        self.width        = np.nan
        self.window_left  = np.nan
        self.window_right = np.nan
        self.window_width = np.nan
        self.s2_area      = np.nan
        self.s2_area_top  = np.nan
        self.s2_aft       = np.nan
        
        self.strArr = np.zeros(
            n_rows,
            dtype=[
                ('idx_out'     , np.int32),
                ('event'       , np.int32),
                ('duration'    , np.float32),
                ('x_ins'       , np.float32), # Truth
                ('y_ins'       , np.float32),
                ('true_left'   , np.int32),
                ('true_right'  , np.int32),
                ('true_x'      , np.float32),
                ('true_y'      , np.float32),
                ('true_nels'   , np.int32),
                ('true_nphs'   , np.int32),
                ('left_index'  , np.int32),   # Reco
                ('left'        , np.int32),
                ('right'       , np.int32),
                ('x'           , np.float32),
                ('y'           , np.float32),
                ('z'           , np.float32),
                ('dt'          , np.float32),
                ('s2_area'     , np.float32),
                ('s2_area_top' , np.float32),
                ('s2_aft'      , np.float32),
                ('s2_areas'    , np.float16, n_channels),
                ('image'       , np.float16, (n_channels, n_samples) ),
                ('s2_n_truncated', np.int32),
                ('s2_area_truncated', np.float32),
                
            ]
        )
        
        
        #----------------------------------------------------------------------
        # Kludge
        #----------------------------------------------------------------------
        
        try:

            self.f_df_all = self.dir_in + '/data_new.hdf5'
            self.df_all   = pd.DataFrame()

            assert(os.path.exists(self.f_df_all))
            
            self.df_all = pd.read_hdf(self.f_df_all)
            self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 's2_center_time')
            self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_first_electron_true')
            self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_last_electron_true')
            self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_first_photon_true')
            self.df_all = helpers_fax_truth.nsToSamples(self.df_all, 't_last_photon_true')
            
        except Exception as ex:
            print(ex)
            

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
            self.zip_callback_pax)
        
        return

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dir_out'    , required=True)
    parser.add_argument('-dir_in'     , required=True)
    parser.add_argument('-dir_fmt'    , required=True)
    parser.add_argument('-zip_fmt'    , required=True)
    parser.add_argument('-n_zip'      , required=True, type=int)
    parser.add_argument('-n_intr'     , required=True, type=int)
    parser.add_argument('-isStrict'   , required=True, default=True, type=lambda x: (str(x).lower() == 'true'))

    return parser.parse_args()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):
    
    print("Starting...")

    t1 = time.time()
    mrg = mergePax()
    mrg.main()
    t2 = time.time()
    
    print("Done in {0:.1f} min".format( (t2-t1)/60 ))
    
    
    