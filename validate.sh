#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/hltsrv1/software/nematus-master

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/hltsrv1/software/moses/moses-20150228_kenlm_cmph_xmlrpc_irstlm_master

work_dir=$1
dev=$2
ref=$3
device=$4

model="$work_dir/model.npz.dev.npz"
prefix="$work_dir/model.npz"

# decode
THEANO_FLAGS="mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn,force_device=True,lib.cnmem=0.0,dnn.enabled=False" OMP_NUM_THREADS=1 python $nematus/nematus/translate.py \
     -m $model \
     -i $dev \
     -o $dev.output \
     -a $dev.output.align \
     -k 12 -n -p 1

$nematus/postprocess-dev.sh < $dev.output > $dev.output.postprocessed

## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed >> ${prefix}_bleu_scores
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${prefix}_best_bleu
  cp ${prefix}.dev.npz "$work_dir/model.best_bleu.npz"
fi
