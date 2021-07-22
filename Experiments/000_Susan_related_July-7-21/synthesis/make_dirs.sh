#!/bin/bash
DIR=$1
echo "Making subdirectories named $DIR in chks/, outs/, verilog/, ddc/, vg/, reports/"
mkdir -p chks/$DIR
mkdir -p outs/$DIR
mkdir -p verilog/$DIR
mkdir -p ddc/$DIR
mkdir -p vg/$DIR
mkdir -p reports/$DIR
