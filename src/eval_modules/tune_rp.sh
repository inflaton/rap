#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/../..
echo Current Directory:
pwd

nvidia-smi
uname -a
cat /etc/os-release
lscpu
grep MemTotal /proc/meminfo

pip install -r requirements.txt

# Evaluating Models (from Smallest to Largest) for the MS MARCO Dataset
./src/eval_modules/eval-hf_v2.sh google gemma-1.1-2b-it true

./src/eval_modules/eval-hf_v2.sh microsoft Phi-3-mini-128k-instruct true

./src/eval_modules/eval-hf_v2.sh google gemma-1.1-7b-it true

./src/eval_modules/eval-hf_v2.sh mistralai Mistral-7B-Instruct-v0.2 true

./src/eval_modules/eval-hf_v2.sh meta-llama Llama-2-7b-chat-hf true

./src/eval_modules/eval-hf_v2.sh meta-llama Llama-2-13b-chat-hf true

./src/eval_modules/eval-hf_v2.sh meta-llama Llama-2-70b-chat-hf true

./src/eval_modules/eval-hf_v2.sh meta-llama Meta-Llama-3-8B-Instruct true

./src/eval_modules/eval-hf_v2.sh meta-llama Meta-Llama-3-70B-Instruct true

# Evaluating Models (from Smallest to Largest) for the WebQSP Dataset
./src/eval_modules/eval-hf_v2.sh google gemma-1.1-2b-it false

./scripts/eval-hf_v2.sh microsoft Phi-3-mini-128k-instruct false

./src/eval_modules/eval-hf_v2.sh google gemma-1.1-7b-it false

./src/eval_modules/eval-hf_v2.sh mistralai Mistral-7B-Instruct-v0.2 false

./src/eval_modules/eval-hf_v2.sh meta-llama Llama-2-7b-chat-hf false

./src/eval_modules/eval-hf_v2.sh meta-llama Llama-2-13b-chat-hf false

./src/eval_modules/eval-hf_v2.sh meta-llama Llama-2-70b-chat-hf false

./src/eval_modules/eval-hf_v2.sh meta-llama Meta-Llama-3-8B-Instruct false

./src/eval_modules/eval-hf_v2.sh meta-llama Meta-Llama-3-70B-Instruct false
