import numpy
import os
import sys

# directory that contains train, dev, and test files
data_dir = "/hltsrv0/chatterjee/nematus-master/experiments/cettolo/temp"

# directory to save system generated files and models
work_dir = "/hltsrv0/chatterjee/nematus-master/experiments/cettolo/temp"

vocab_src = 40000
vocab_tgt = 40000

src_ext = "de"
tgt_ext = "en"

train_src = data_dir + '/train.bpe.' + src_ext
train_tgt = data_dir + '/train.bpe.' + tgt_ext
dict_src = data_dir + '/train.bpe.' + src_ext + '.json'
dict_tgt = data_dir + '/train.bpe.' + tgt_ext + '.json'
dev_src = data_dir + '/dev.bpe.' + src_ext
dev_tgt = data_dir + '/dev.' + tgt_ext

from nematus.nmt import train


if __name__ == '__main__':
    validation_script = []
    validation_script.append('./validate.sh')
    validation_script.append(work_dir)
    validation_script.append(dev_src)
    validation_script.append(dev_tgt)

    validerr = train(saveto=work_dir + '/model.npz',
                    reload_=True,
                    dim_word=620,
                    dim=620,
                    n_words=vocab_tgt,
                    n_words_src=vocab_src,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.001,
                    optimizer='adadelta',
                    maxlen=50,
                    batch_size=100,
                    valid_batch_size=100,
                    datasets=[train_src, train_tgt],
                    valid_datasets=[dev_src, dev_tgt],
                    dictionaries=[dict_src, dict_tgt],
                    validFreq=1000,
                    dispFreq=100000,
                    saveFreq=1000,
                    sampleFreq=100000,
                    use_dropout=True,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.1, # dropout source words (0: no dropout)
                    dropout_target=0.1, # dropout target words (0: no dropout)
                    overwrite=False,
                    finish_after=720000,
                    max_epochs=10,
                    external_validation_script=validation_script)
    print validerr
