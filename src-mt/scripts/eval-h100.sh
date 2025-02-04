#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

nvidia-smi
uname -a
cat /etc/os-release
lscpu
grep MemTotal /proc/meminfo

# pip install torch torchvision torchaudio
# pip install -r requirements.txt

export MAX_NEW_TOKENS=2048
export START_REPETITION_PENALTY=1.0
export END_REPETITION_PENALTY=1.1

export USING_CHAT_TEMPLATE=false
export RESULTS_PATH=results/mac-results_rpp_with_mnt_2048_generic_prompt.csv

# export USING_CHAT_TEMPLATE=true
# export RESULTS_PATH=results/mac-results_rpp_with_mnt_2048.csv


export BATCH_SIZE=1
export LOAD_IN_4BIT=false

./scripts/eval-rpp.sh microsoft Phi-3.5-mini-instruct checkpoint-210

./scripts/eval-rpp.sh shenzhi-wang Llama3.1-8B-Chinese-Chat checkpoint-105


export LOAD_IN_4BIT=true
./scripts/eval-rpp.sh shenzhi-wang Llama3.1-70B-Chinese-Chat checkpoint-210
