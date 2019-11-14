#!/bin/bash

python ./merge_pax.py -dir_out ./temp_pax -dir_in ../../data-xe1t/pax2019-11-14/zip/ -dir_fmt instructions_[0-9]* -zip_fmt *.zip -n_intr 1 -isStrict True

