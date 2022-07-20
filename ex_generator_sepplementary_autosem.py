from task_selection import autosem_p1_lstm, autosem_p1_bert_base, autosem_p1_bert_large, autosem_p2_bert_base, \
    autosem_p2_bert_large, autosem_p2_lstm

# set args
model = 'bert-large-cased'  # TODO: backend model
stage = 1  # TODO:1 or 2
task = autosem_p1_bert_large  # TODO:
ratio = autosem_p2_bert_large  # TODO
method = "autosem-p1-Bert-large"  # TODO: autosem-p1-BiLSTM,autosem-p1-Bert-base,autosem-p1-Bert-large,autosem-BiLSTM,autosem-Bert-base,autosem-Bert-large
lr = '5e-5'

task_group_list = []
auxiliary_ratio_list = []
for main_task, auxiliary_lis in task.items():
    t_l = [main_task] + auxiliary_lis
    task_group_list.append(tuple(t_l))
    if stage == 2:
        auxiliary_ratio_list.append(ratio[main_task])
    else:
        auxiliary_ratio_list.append(list(range(len(t_l))))  # to be compatible

if stage == 2:
    assert len(task_group_list) == len(auxiliary_ratio_list)
print("task group list:", task_group_list)
print("auxiliary ratio list:", auxiliary_ratio_list)

# for main exp, the gradient file path is similar with bert model type

python_script = 'train_with_mix_ratio.py' if stage == 2 else 'train.py'
encoder_type = 1 if 'roberta' not in model else 2

template = 'echo "AutoSem stage {} running begin!"\n'.format(stage)
template += 'BATCH_SIZE=$1\ngpu=$2\nEpoch=$3\n'
template += 'ENCODER_TYPE={}\n'.format(encoder_type)
template += 'echo "export CUDA_VISIBLE_DEVICES=${gpu}"\necho "epoch num=${Epoch}"\nexport CUDA_VISIBLE_DEVICES=${gpu}\ntstr=$(date +"%FT%H%M")\n'
template += 'MODEL_ROOT="checkpoints"\nBERT_PATH="{}"\nDATA_DIR="data/canonical_data/{}"\n'.format(model, model)
template += 'answer_opt=1\noptim="adamax"\ngrad_clipping=0\nglobal_grad_clipping=1\nlr="{}"\ngrad_accumulation_step=4\n\n'.format(
    lr)

# task relevance
for task_group, auxiliary_ratio in zip(task_group_list, auxiliary_ratio_list):
    assert len(task_group) == len(auxiliary_ratio)
    main_task = task_group[0]
    auxiliary_task = task_group[1:]

    train_choice = ','.join(task_group)
    test_choice = 'mnli_matched,mnli_mismatched' if 'mnli' in main_task else main_task
    ratio_choice = ','.join(list(map(lambda x: str(x), auxiliary_ratio)))
    print("task group:{}, ratio:{}".format(train_choice, ratio_choice))

    # task specification
    template += '\necho "====== this is main task {} ======="\n'.format(main_task)
    template += 'prefix="{}-{}"\n'.format(model, main_task)
    template += 'model_dir="checkpoints_mix_backend/{}/{}/{}'.format(method, model,
                                                                     main_task)
    template += '"\n'
    template += 'log_file="${model_dir}/log.log"\n'

    template += 'train_datasets="{}"\n'.format(train_choice)
    template += 'test_datasets="{}"\n'.format(test_choice)
    template += 'python {} '.format(python_script)
    template += '--data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers 24 ' \
                '--init_checkpoint ${BERT_PATH} --epochs ${Epoch} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} ' \
                '--log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} ' \
                '--train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} ' \
                '--global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} ' \
                '--learning_rate ${lr} '
    template += '--mix_ratios {} '.format(ratio_choice) if stage == 2 else ""
    template += '\n'

shell_name = "sp_{}_{}.sh".format(method, model)

print("shell name:", shell_name)

with open(shell_name, "w") as f:
    f.write(template)
