#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $BASEDIR

mkdir bsds_500

# color images
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz
tar -xf BSR_full.tgz
mv BSR/BSDS500/data/images bsds_500/color_images
rm -rf BSR_full.tgz BSR

# greyscale images
wget https://www.dropbox.com/s/vuxitdiawlt0go9/bsds_500_greyscale.zip
unzip bsds_500_greyscale.zip
mv greyscale_images bsds_500/
rm bsds_500_greyscale.zip