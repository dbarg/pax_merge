#!/bin/bash

python ./merge_pax.py -dir_in ../pax_run/test/zip -dir_fmt instructions_[0-9]* -zip_fmt *.zip -n_intr 1

