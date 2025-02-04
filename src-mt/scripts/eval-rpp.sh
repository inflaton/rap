#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export ORG_NAME=$1
export MODEL=$2
export CHECKPOINT=$3

export MODEL_NAME=$ORG_NAME/$MODEL
export ADAPTER_NAME_OR_PATH=checkpoints/${MODEL}_${CHECKPOINT}

echo Evaluating $MODEL_NAME
python llm_toolkit/eval_rpp.py
