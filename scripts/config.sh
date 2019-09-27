#!/bin/bash
#
# Set environment variables for the training and testing of stanfordnlp modules.

# Set UDBASE to the location of CoNLL18 folder
# For details, see http://universaldependencies.org/conll18/data.html
export UDBASE=./data/treebanks/ner

# Set directories to store processed training/evaluation files
export DATA_ROOT=./data/output
export TOKENIZE_DATA_DIR=$DATA_ROOT/tokenize
export MWT_DATA_DIR=$DATA_ROOT/mwt
export LEMMA_DATA_DIR=$DATA_ROOT/lemma
export POS_DATA_DIR=$DATA_ROOT/pos
export NER_DATA_DIR=$DATA_ROOT/ner
export DEPPARSE_DATA_DIR=$DATA_ROOT/depparse
export ETE_DATA_DIR=$DATA_ROOT/ete

# Set directories to store external word vector data
export WORDVEC_DIR=./data/wordvec
