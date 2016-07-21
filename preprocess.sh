#!/bin/sh
subword="/hltsrv1/software/subword"
script_dir="/hltsrv1/software/nematus-master"

# directory that contains training data
data_dir=$1

# source and target corpus extention eg. en
src=$2
trg=$3

# directory to save the output files
work_dir=$4

# number of codes to learn from the training data eg. 50000
bpe_operation_src=$5
bpe_operation_trg=$6

apply_bpe=$7

##########################################################################

if [ -s "$work_dir/train.$src" ] ; then
    echo "$work_dir/train.$src already exist"
else
    echo "Copying source corpus"
    cp "$data_dir/train.$src" "$work_dir/train.$src"
    echo "Copied $data_dir/train.$src to $work_dir/train.$src"
fi

if [ -s "$work_dir/train.$trg" ] ; then
    echo "$work_dir/train.$trg already exist"
else
    echo "Copying target corpus"
    cp "$data_dir/train.$trg" "$work_dir/train.$trg"
    echo "Copied $data_dir/train.$trg to $work_dir/train.$trg"
fi

if [ -s "$work_dir/dev.$src" ] ; then
    echo "$work_dir/dev.$src already exist"
else
    echo "Copying source corpus"
    cp "$data_dir/dev.$src" "$work_dir/dev.$src"
    echo "Copied $data_dir/dev.$src to $work_dir/dev.$src"
fi

if [ -s "$work_dir/dev.$trg" ] ; then
    echo "$work_dir/dev.$trg already exist"
else
    echo "Copying target corpus"
    cp "$data_dir/dev.$trg" "$work_dir/dev.$trg"
    echo "Copied $data_dir/dev.$trg to $work_dir/dev.$trg"
fi

train_src="$work_dir/train.$src"
train_trg="$work_dir/train.$trg"
dev_src="$data_dir/dev.$src"
dev_trg="$data_dir/dev.$trg"

if [ $apply_bpe -eq "1" ] ; then
    echo "Running subword preprocessing"
    
    train_code_src="$work_dir/train.code.$src"
    train_code_trg="$work_dir/train.code.$trg"
    train_bpe_src="$work_dir/train.bpe.$src"
    train_bpe_trg="$work_dir/train.bpe.$trg"
    dev_bpe_src="$work_dir/dev.bpe.$src"
    
    # Learn subword codes
    if [ -s "$train_code_src" ] ; then
        echo "$train_code_src already exist"
    else
        echo "Learning codes for the source corpus"
        $subword/learn_bpe.py -s "$bpe_operation_src" < "$train_src" > "$train_code_src"
        echo "Code for the source corpus is saved in $train_code_src"
    fi
    
    if [ -s "$train_code_trg" ] ; then
        echo "$train_code_trg already exist"
    else
        echo "Learning codes for the target corpus"
        $subword/learn_bpe.py -s "$bpe_operation_trg" < "$train_trg" > "$train_code_trg"
        echo "Code for the target corpus is saved in $train_code_trg"
    fi
    
    # Apply the learned codes
    if [ -s "$train_bpe_src" ] ; then
        echo "$train_bpe_src already exist"
    else
        echo "Applying codes for the source corpus"
        $subword/apply_bpe.py -c "$train_code_src" -o "$train_bpe_src" < "$train_src"
        echo "BPE processed source corpus is saved in $train_bpe_src"
    fi
    
    if [ -s "$train_bpe_trg" ] ; then
        echo "$train_bpe_trg already exist"
    else
        echo "Applying codes for the target corpus"
        $subword/apply_bpe.py -c "$train_code_trg" -o "$train_bpe_trg" < "$train_trg"
        echo "BPE processed target corpus is saved in $train_bpe_trg"
    fi
    
    # Apply the learned codes
    if [ -s "$dev_bpe_src" ] ; then
        echo "$dev_bpe_src already exist"
    else
        echo "Applying codes for the source corpus"
        $subword/apply_bpe.py -c "$train_code_src" -o "$dev_bpe_src" < "$dev_src"
        echo "BPE processed source corpus is saved in $dev_bpe_src"
    fi
    
    train_src=$train_bpe_src
    train_trg=$train_bpe_trg
    
fi

# build network dictionary
if [ -s "$train_src.json" ] ; then
    echo "$train_src.json already exist"
else
    $script_dir/data/build_dictionary.py "$train_src"
fi

if [ -s "$train_trg.json" ] ; then
    echo "$train_trg.json already exist"
else
    $script_dir/data/build_dictionary.py "$train_trg"
fi
