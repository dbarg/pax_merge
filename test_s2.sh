#!/bin/bash

python ./merge.py -dir_in ../../data-xe1t/s2only/fax/ -dir_out ./temp -dir_fmt [0-9]* -isStrict True -isS2only True -max_dirs 1

