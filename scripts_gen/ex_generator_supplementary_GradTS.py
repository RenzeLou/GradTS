from task_selection import GradTS_fg_bert_base, GradTS_fg_bert_large, GradTS_thres_bert_base, GradTS_thres_bert_large, \
    GradTS_trial_bert_large, GradTS_trial_bert_base

task_list = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp']  # 8 task

# set args
stage = 2  # TODO: {1:thres} {2:trial} {3:fg}
model = 'bert-large-cased'  # TODO: backend model # bert-base-cased,bert-base-uncased,roberta-base,bert-large-uncased,roberta-large,bert-base-cased,bert-large-cased
strategy = GradTS_trial_bert_base  # TODO
ins_threses = GradTS_fg_bert_large  # TODO: only be used when run **fg** method
method = "GradTS-trial-Bert-base"  # TODO: GradTS-thres-Bert-base, GradTS-thres-Bert-large, GradTS-trial-Bert-base, GradTS-trial-Bert-large, GradTS-fg-Bert-base, GradTS-fg-Bert-large

bert_model_type = model
gradient_path = "./gradient_files/{}".format(bert_model_type)
num_layer = 24 if "large" in bert_model_type else 12
data_dir = "./data/canonical_data/{}".format(bert_model_type)
lr = '5e-5'
best_eval_score = True

python_script = 'train_with_corr.py' if stage in [1, 2] else 'train_with_corr_test.py'
encoder_type = 1 if 'roberta' not in bert_model_type else 2

# setting param
template = 'echo "{} running begin!"\n'.format(method)
template += 'BATCH_SIZE=$1\ngpu=$2\nEpoch=$3\n'
template += 'ENCODER_TYPE={}\n'.format(encoder_type)
template += "NUM_LAYER={}\n".format(num_layer)
template += "DATA_DIR={}\n".format(data_dir)
template += 'echo "export CUDA_VISIBLE_DEVICES=${gpu}"\necho "epoch num=${Epoch}"\nexport CUDA_VISIBLE_DEVICES=${gpu}\ntstr=$(date +"%FT%H%M")\n'
template += 'answer_opt=1\noptim="adamax"\ngrad_clipping=0\nglobal_grad_clipping=1\nlr="{}"\ngrad_accumulation_step=4\n\n'.format(
    lr)

for task in task_list:
    main_task = task
    test_datasets = main_task if "mnli" not in main_task else "mnli_matched,mnli_mismatched"
    candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
    train_datasets = [main_task] + strategy[main_task]
    # all_candidate = candidate_set.split(",")
    ins_thres = ins_threses[main_task] if stage == 3 else 1
    method_choice = 2 if stage in [1, 2] else 3
    template += 'echo "main task {} trial exp begin!"\n'.format(main_task)
    if stage in [1, 2]:
        output_dir = "checkpoints_mix_backend/{}/{}/{}".format(method, model, main_task)
    else:
        output_dir = "checkpoints_mix_backend/{}/{}".format(method, model)
    template += 'python {} '.format(python_script)
    template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
        main_task, method_choice, 1, ins_thres, gradient_path, candidate_set)
    template += "--train_datasets {} --test_datasets {} ".format(",".join(train_datasets), test_datasets)
    template += '--data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers ${NUM_LAYER} ' \
                '--epochs ${Epoch} --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} ' \
                '--answer_opt ${answer_opt} --optimizer ${optim} ' \
                '--grad_clipping ${grad_clipping} ' \
                '--global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} ' \
                '--learning_rate ${lr} '
    template += "--record_eval_best " if best_eval_score and stage in [1, 2] else ""
    template += "--output_dir {} ".format(output_dir)
    template += "--bert_model_type {} ".format(bert_model_type)
    template += "--init_checkpoint {} ".format(bert_model_type)
    template += "--force_run " if stage in [1, 2] else ""
    template += '\n'

shell_name = "sp_{}_{}.sh".format(method, bert_model_type)

print("\nshell name:", shell_name)

with open(shell_name, "w") as f:
    f.write(template)
