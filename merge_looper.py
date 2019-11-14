#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import glob
import os
import pickle
import sys
import zipfile
import zlib


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def looper(dir_in, dir_fmt, zip_fmt, func):

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    print("Input path: {0}".format(dir_in) )
    
    assert(os.path.exists(dir_in))
    
    lst_dir = glob.glob(dir_in + '/' + dir_fmt)
    lst_dir = sorted(lst_dir)
    n_dir   = len(lst_dir)
    
    print("{0} Directories found in {1}".format(n_dir, dir_in))
    
    
    #--------------------------------------------------------------------------
    # Loop over directories
    #--------------------------------------------------------------------------

    for i_dir, dirname in enumerate(lst_dir):
        
        print("\nDirectory: {0}\n".format(dirname))
    
        #----------------------------------------------------------------------
        # Zip Files
        #----------------------------------------------------------------------
    
        lst_zip = glob.glob(dirname + '/' + zip_fmt)
        lst_zip = sorted(lst_zip)
        n_zip   = len(lst_zip)
        
        print("   {0} ZIP file(s) found".format(n_zip))
        
        for i_zip, zipname in enumerate(lst_zip):

            print("   " + zipname)
            
            func(zipname, i_dir, i_zip)
            
            continue
                
        #----------------------------------------------------------------------
        # End loop on directories
        #----------------------------------------------------------------------

        print("")
        
        continue
    
    return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if (__name__ == "__main__"):
    
    main()