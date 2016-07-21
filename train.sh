#!/bin/bash 

# set some parameters for qsub

# -N name of the job
#$ -N NMT

# -q name of the queue to use
#$ -q gpgpu.q

# -l mf=amount of memory requested (this is a MANDATORY parameter), use carefully.
#$ -l mf=500G,gpu=1

#$ -S /bin/sh

script_dir="/hltsrv1/software/nematus-master"
config=$1
device=$2
THEANO_FLAGS="mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn,force_device=True,lib.cnmem=0.0,dnn.enabled=False" OMP_NUM_THREADS=1 python $script_dir/train.py $config $device
