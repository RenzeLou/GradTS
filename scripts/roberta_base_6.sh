#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "train.sh <batch_size> <gpu>"
  exit 1
fi
prefix="mt-dnn-rte"
BATCH_SIZE=$1
gpu=$2
Epoch=$3
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
echo "epoch num=${Epoch}"
tstr=$(date +"%FT%H%M")

train_datasets="rte,qnli,mrpc,sst,cola,stsb"
test_datasets="rte,qnli,mrpc,sst,cola,stsb"
MODEL_ROOT="checkpoints"
BERT_PATH="roberta-base"
DATA_DIR="data/canonical_data/roberta-base"

answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"

model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --epochs ${Epoch} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on
