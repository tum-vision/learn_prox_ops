#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $BASEDIR

cd tensorflow
wget https://www.dropbox.com/s/fr7evby4h7q4bo3/DNCNN__gaussian_0.02.zip
unzip DNCNN__gaussian_0.02.zip
rm DNCNN__gaussian_0.02.zip