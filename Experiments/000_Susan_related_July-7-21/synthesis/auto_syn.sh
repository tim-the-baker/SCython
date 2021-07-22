#!/bin/bash

for n in {8..16}; do
  make syn n=$n q=$n file_ID="FIR1" grouped=0 core=$c
  make clean
done
