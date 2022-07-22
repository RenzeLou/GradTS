echo "GradTS-fg running begin!"
BATCH_SIZE=$1
gpu=$2
Epoch=$3
ENCODER_TYPE=1
NUM_LAYER=12
DATA_DIR=./data/canonical_data/bert-base-cased
echo "export CUDA_VISIBLE_DEVICES=$gpu"
echo "epoch num=$Epoch"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")
answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"
grad_accumulation_step=4

python train_with_corr_test.py --main_task rte --method 3 --task_threshold 0.9 --instance_threshold 0.62 --gradient_path ./gradient_files/bert-base-cased --candidate_set wnli --train_datasets rte,wnli --test_datasets rte --data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers ${NUM_LAYER} --epochs ${Epoch} --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} --answer_opt ${answer_opt} --optimizer ${optim} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} --learning_rate ${lr} --output_dir ./checkpoints/2_bert-base-cased/GradTS-fg --bert_model_type bert-base-cased --init_checkpoint bert-base-cased 
python train_with_corr_test.py --main_task wnli --method 3 --task_threshold 0.9 --instance_threshold 0.62 --gradient_path ./gradient_files/bert-base-cased --candidate_set rte --train_datasets wnli,rte --test_datasets wnli --data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers ${NUM_LAYER} --epochs ${Epoch} --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} --answer_opt ${answer_opt} --optimizer ${optim} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} --learning_rate ${lr} --output_dir ./checkpoints/2_bert-base-cased/GradTS-fg --bert_model_type bert-base-cased --init_checkpoint bert-base-cased 
