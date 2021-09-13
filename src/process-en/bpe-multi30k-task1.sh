#!/bin/bash

MOSES="$(dirname ${BASH_SOURCE[0]})/../../external/moses-3a0631a/tokenizer"
export PATH="${MOSES}:$PATH"
BPEPATH="$(dirname ${BASH_SOURCE[0]})/../../external/subword-nmt"
BPEAPPLY=${BPEPATH}/apply_bpe.py
BPELEARN=${BPEPATH}/learn_joint_bpe_and_vocab.py

LLANG=$1
OLANG=$2

BPECODES=$3
BPEVOCAB=$4
INFILE=$5
OUTPREF=$6

TOKFILE="${OUTPREF}.lc.norm.tok.${OLANG}"
OUTFILE="${OUTPREF}.lc.norm.tok.bpe.${OLANG}"

# LLANG=en

cat $INFILE | lowercase.perl | normalize-punctuation.perl -l $LLANG | \
    tokenizer.perl -l $LLANG -threads 2 > $TOKFILE

WC=`cat $TOKFILE | wc`
N_SENTS=`echo $WC | cut -d' ' -f1`
N_WORDS=`echo $WC | cut -d' ' -f2`
N_WORD_PER_SENT=`python3 -c "print('%.1f' % (${N_WORDS} / ${N_SENTS}))"`
echo " ($lang) $N_SENTS sentences, $N_WORDS words, $N_WORD_PER_SENT words/sent"

#  $BPELEARN -s $BPE_MOPS -o $BPEFILE \
#    --input ${TOK}/train.${SUFFIX}.en \
#            ${TOK}/train.${SUFFIX}.${TLANG} \
#    --write-vocabulary \
#            "${BPE}/${LPAIR}/vocab.en" "${BPE}/${LPAIR}/vocab.$TLANG"

$BPEAPPLY -c $BPECODES --vocabulary $BPEVOCAB < $TOKFILE > $OUTFILE

