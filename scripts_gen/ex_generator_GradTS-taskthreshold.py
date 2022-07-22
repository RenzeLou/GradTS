''' this script can be used to generate shell to run mtdnn with the task selection result of AutoSem & GradTS '''
''' note that AutoSem has two stage, we generate these two stage on different gpu respectively'''
''' note that GradTS has three stage, that is threshold-based, trial-based and instance corr-based'''
task_list = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp']  # 8 task
# task_list = ['cola', 'mrpc', 'sst', 'wnli', 'rte', 'qnli']  # 6 task
# task_list = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp', 'meld', 'dmeld']  # 10 task
# task_list = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp', 'stsb', 'pos', 'ner', 'sc']  # 12 task  ## todo
labeled_data = ['ner', 'pos', 'sc', 'meld', 'dmeld']
method_dict = {1: "GradTS-thres", 2: "GradTS-trial", 3: "GradTS-fg"}

# set args
method = 1  # 1,2,3
task_threshold = 0.6  # todo: 0.66,0.59,0.69,0.59,0.57
bert_model_type = "bert-large-cased"  # todo: bert-base-uncased,roberta-base,bert-large-uncased，roberta-large,bert-base-cased,bert-large-cased
main_task = "pos"
# train_datasets = "cola"  # ",".join(task_list)
# test_datasets = main_task if "mnli" not in main_task else "mnli_matched,mnli_mismatched"
candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
instance_threshold = 0.62
# trial_num = 1
gradient_path = "./gradient_files/{}".format(bert_model_type)
output_dir = "./checkpoints/{}/{}".format(str(len(task_list)) + "_" + bert_model_type, method_dict[method])
num_layer = 24 if "large" in bert_model_type else 12
data_dir = "./data/canonical_data/{}".format(bert_model_type)
lr = '5e-5'  ## TODO
wnli_only = False  # due to the instability of wnli, run the wnli seperately
# for main exp, the gradient file path is similar with bert model type
# note epoch num

python_script = 'train_with_corr.py'
encoder_type = 1 if 'roberta' not in bert_model_type else 2

# setting param
template = 'echo "{} running begin!"\n'.format(method_dict[method])
template += 'BATCH_SIZE=$1\ngpu=$2\nEpoch=$3\n'
template += 'ENCODER_TYPE={}\n'.format(encoder_type)
template += "NUM_LAYER={}\n".format(num_layer)
template += "DATA_DIR={}\n".format(data_dir)
template += 'echo "export CUDA_VISIBLE_DEVICES=${gpu}"\necho "epoch num=${Epoch}"\nexport CUDA_VISIBLE_DEVICES=${gpu}\ntstr=$(date +"%FT%H%M")\n'
# template += 'MODEL_ROOT="checkpoints"\nBERT_PATH="{}"\nDATA_DIR="data/canonical_data/{}"\n'.format(model, model)
template += 'answer_opt=1\noptim="adamax"\ngrad_clipping=0\nglobal_grad_clipping=1\nlr="{}"\ngrad_accumulation_step=4\n\n'.format(
    lr)

# for t in task_list:
#     main_task = t
#     candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
#     # training param
#     template += 'python {} '.format(python_script)
#     template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
#         main_task, method, task_threshold, instance_threshold, gradient_path, candidate_set)
#     # template += "--train_datasets {} --test_datasets {} ".format(train_datasets, test_datasets)
#     template += '--data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers ${NUM_LAYER} ' \
#                 '--epochs ${Epoch} --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} ' \
#                 '--answer_opt ${answer_opt} --optimizer ${optim} ' \
#                 '--grad_clipping ${grad_clipping} ' \
#                 '--global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} ' \
#                 '--learning_rate ${lr} '
#     template += '--test_with_label ' if main_task in labeled_data else ""
#     template += "--output_dir {} ".format(output_dir)
#     template += "--bert_model_type {} ".format(bert_model_type)
#     template += "--init_checkpoint {} ".format(bert_model_type)
#     template += '\n'

# for wnli
# ===========================================
# wnli_only = True
# for t in ["wnli"]:
#     main_task = t
#     candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
#     # training param
#     template += 'python {} '.format(python_script)
#     template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
#         main_task, method, task_threshold, instance_threshold, gradient_path, candidate_set)
#     # template += "--train_datasets {} --test_datasets {} ".format(train_datasets, test_datasets)
#     template += '--data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers ${NUM_LAYER} ' \
#                 '--epochs ${Epoch} --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} ' \
#                 '--answer_opt ${answer_opt} --optimizer ${optim} ' \
#                 '--grad_clipping ${grad_clipping} ' \
#                 '--global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} ' \
#                 '--learning_rate ${lr} '
#     template += '--test_with_label ' if main_task in labeled_data else ""
#     template += "--output_dir {} ".format(output_dir)
#     template += "--bert_model_type {} ".format(bert_model_type)
#     template += "--init_checkpoint {} ".format(bert_model_type)
#     template += '\n'
# ===========================================

# shell_name = "wnli_{}_{}_{}_{}.sh".format(method_dict[method], len(task_list), bert_model_type,
#                                           task_threshold) if wnli_only else "{}_{}_{}.sh".format(method_dict[method],
#                                                                                                  len(task_list),
#                                                                                                  bert_model_type)


# for sc
for t in ['sc']:
    main_task = t
    candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
    # training param
    template += 'python {} '.format(python_script)
    template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
        main_task, method, task_threshold, instance_threshold, gradient_path, candidate_set)
    # template += "--train_datasets {} --test_datasets {} ".format(train_datasets, test_datasets)
    template += '--data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers ${NUM_LAYER} ' \
                '--epochs ${Epoch} --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} ' \
                '--answer_opt ${answer_opt} --optimizer ${optim} ' \
                '--grad_clipping ${grad_clipping} ' \
                '--global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} ' \
                '--learning_rate ${lr} '
    template += '--test_with_label ' if main_task in labeled_data else ""
    template += "--output_dir {} ".format(output_dir)
    template += "--bert_model_type {} ".format(bert_model_type)
    template += "--init_checkpoint {} ".format(bert_model_type)
    template += '\n'

shell_name = "sc_{}_{}_{}_{}.sh".format(method_dict[method], len(task_list), bert_model_type,
                                          task_threshold)


print("shell name:",shell_name)

with open(shell_name, "w") as f:
    f.write(template)
