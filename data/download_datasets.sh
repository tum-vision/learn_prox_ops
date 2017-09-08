#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $BASEDIR

#
# deblurring grey
#
mkdir deblurring_grey
wget http://www.cs.ubc.ca/labs/imager/tr/2014/FlexISP/FlexISP_Supplement_Heide2014.zip
unzip FlexISP_Supplement_Heide2014.zip
mv supplemental/supplementalWeb/deconvolution/images/gray/* deblurring_grey/
mv deblurring_grey/experiment_1 deblurring_grey/experiment_a
mv deblurring_grey/experiment_2 deblurring_grey/experiment_b
mv deblurring_grey/experiment_3 deblurring_grey/experiment_c
mv deblurring_grey/experiment_4 deblurring_grey/experiment_d
mv deblurring_grey/experiment_5 deblurring_grey/experiment_e
rm -rf supplemental FlexISP_Supplement_Heide2014.zip

#
# demosaicking
#

# McMaster
mkdir demosaicking
wget http://www4.comp.polyu.edu.hk/~cslzhang/DATA/McM.zip
unzip -P McM_CDM McM.zip
mv McM demosaicking/mc_master
rm McM.zip

# Kodak
mkdir demosaicking/kodak
cd demosaicking/kodak
wget -r -np -nH --cut-dirs=3 -R index.html http://www.natrox.org/img/ref/kodak/