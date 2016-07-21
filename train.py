#!/usr/bin/env python
'''
    File name: train.py
    Description: this is a wrapper script to preprocess and train a NMT system (it uses Nematus toolkit)
    Author: Rajen Chatterjee
    Email: chatterjee@fbk.eu
    Date created: 17th July, 2016
    Python Version: 2.7
'''

import sys
from subprocess import Popen
from nematus.nmt import train
import config

script_dir = "/hltsrv1/software/nematus-master"
conf = config.Configuration(sys.argv[1])
device = sys.argv[2]

def copy_config(source, destination):
    cmd = []
    cmd.append("cp")
    cmd.append(source)
    cmd.append(destination)
    p = Popen(cmd)
    exit_val = p.wait()
    return exit_val

def get_data_path():
    path = dict()
    if(conf.apply_bpe == "1"):
        path["train_src"] = conf.work_dir + "/train.bpe." + conf.src
        path["train_trg"] = conf.work_dir + "/train.bpe." + conf.trg
        path["dev_src"] = conf.work_dir + "/dev.bpe." + conf.src
        path["dict_src"] = conf.work_dir + "/train.bpe." + conf.src + ".json"
        path["dict_trg"] = conf.work_dir + "/train.bpe." + conf.src + ".json"
    else:
        path["train_src"] = conf.work_dir + "/train." + conf.src
        path["train_trg"] = conf.work_dir + "/train." + conf.trg
        path["dev_src"] = conf.work_dir + "/dev." + conf.src
        path["dict_src"] = conf.work_dir + "/train." + conf.src + ".json"
        path["dict_trg"] = conf.work_dir + "/train." + conf.src + ".json"
    path["dev_trg"] = conf.work_dir + "/dev." + conf.trg
    
    return path


def get_external_validation_script(path):
    external_validation_script = []
    external_validation_script.append(script_dir + "/validate.sh")
    external_validation_script.append(conf.work_dir)
    external_validation_script.append(path["dev_src"])
    external_validation_script.append(path["dev_trg"])
    external_validation_script.append(device)
    print ' '.join(external_validation_script)
    return external_validation_script

def run_subword():
    cmd = []
    cmd.append(script_dir + "/preprocess.sh")
    cmd.append(conf.data_dir)
    cmd.append(conf.src)
    cmd.append(conf.trg)
    cmd.append(conf.work_dir)
    cmd.append(conf.bpe_operation_src)
    cmd.append(conf.bpe_operation_trg)
    cmd.append(conf.apply_bpe)
    preprocess = Popen(cmd)
    exit_val = preprocess.wait()
    return exit_val

def run_nmt(path):
    external_validation_script = get_external_validation_script(path)

    validerr = train(saveto=conf.work_dir + "/model.npz",
                    external_validation_script=external_validation_script,
                    datasets=[path["train_src"], path["train_trg"]],
                    valid_datasets=[path["dev_src"], path["dev_trg"]],
                    dictionaries=[path["dict_src"], path["dict_trg"]],
                    n_words_src=int(conf.n_words_src),
                    n_words=int(conf.n_words),
                    maxlen=int(conf.maxlen),
                    dim_word=int(conf.dim_word),
                    dim=int(conf.dim),
                    batch_size=int(conf.batch_size),
                    valid_batch_size=int(conf.valid_batch_size),
                    reload_=bool(conf.reload_),
                    overwrite=bool(conf.overwrite),
                    optimizer=conf.optimizer,
                    lrate=float(conf.lrate),
                    dispFreq=int(conf.dispFreq),
                    validFreq=int(conf.validFreq),
                    saveFreq=int(conf.saveFreq),
                    sampleFreq=int(conf.sampleFreq),
                    use_dropout=bool(conf.use_dropout),
                    dropout_embedding=float(conf.dropout_embedding),
                    dropout_hidden=float(conf.dropout_hidden),
                    dropout_source=float(conf.dropout_source),
                    dropout_target=float(conf.dropout_target),
                    shuffle_each_epoch=bool(conf.shuffle_each_epoch),
                    max_epochs=int(conf.max_epochs),
                    finish_after=int(conf.finish_after),
                    finetune=bool(conf.finetune),
                    decay_c=float(conf.decay_c),
                    alpha_c=float(conf.alpha_c),
                    clip_c=float(conf.clip_c),
                    patience=int(conf.patience),
                    encoder=conf.encoder,
                    decoder=conf.decoder)
    return validerr

if __name__ == '__main__':
    copy_config(sys.argv[1], conf.work_dir)
    exit_val = run_subword()
    path = get_data_path()
    validerr = run_nmt(path)
