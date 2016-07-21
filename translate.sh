#!/bin/sh

nematus=/hltsrv1/software/nematus-master
mosesdecoder=/hltsrv1/software/moses/moses-20150228_kenlm_cmph_xmlrpc_irstlm_master

model=$1 # path of the model.npz file
test=$2  # test input file
ref=$3   # reference file
device=$4 # gpu device

# decode
THEANO_FLAGS="mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn,force_device=True,lib.cnmem=0.0,dnn.enabled=False" OMP_NUM_THREADS=1 python $nematus/nematus/translate.py \
     -m $model \
     -i $test \
     -o $test.output \
     -a $test.output.align \
     -k 12 -n -p 1

$nematus/postprocess-dev.sh < $test.output > $test.output.postprocessed

$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $test.output.postprocessed