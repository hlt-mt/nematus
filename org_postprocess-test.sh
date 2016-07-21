#/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/hltsrv1/software/moses/moses-20150228_kenlm_cmph_xmlrpc_irstlm_master

# suffix of target language files
lng=en

sed 's/\@\@ //g' | $mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng
# $mosesdecoder/scripts/recaser/detruecase.perl | \

