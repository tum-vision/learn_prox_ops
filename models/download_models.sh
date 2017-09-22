#!/usr/bin/env bash

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR

wget https://www.dropbox.com/s/fr7evby4h7q4bo3/DNCNN__gaussian_0.02.zip
unzip DNCNN__gaussian_0.02.zip
rm DNCNN__gaussian_0.02.zip

cd $INITIAL_DIR