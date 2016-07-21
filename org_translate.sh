#!/bin/sh

# theano device
device=gpu2

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/hltsrv0/chatterjee/nematus-master

test_file_name="tst2014.de-en.tc.de.subword"
model_name="model.npz.dev.npz"
work_dir=/hltsrv0/chatterjee/cettolo

model="$work_dir/$model_name"
input="$work_dir/$test_file_name"
output="$work_dir/$test_file_name.output"
align="$work_dir/$test_file_name.align"

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m $model \
     -i $input \
     -o $output \
     -a $align \
     -k 12 -n -p 1
