#!/bin/bash

python ./merge_pax.py -dir_out ./temp_s2 -dir_in ../../data-xe1t/s2only/fax/ -dir_fmt [0-9]* -zip_fmt sim_s2s/*.zip -n_intr 0

