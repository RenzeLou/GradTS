''' this script can be used to generate shell to run mtdnn with the task selection result of AutoSem & GradTS '''
''' note that AutoSem has two stage, we generate these two stage on different gpu respectively'''
''' note that GradTS has three stage, that is threshold-based, trial-based and instance corr-based'''
# task_list = ['cola','mrpc','sst','wnli','mnli','mnli','qnli','rte','qqp']
# auxiliary_set_list = [('qnli','qqp'),('cola','mnli')]
labeled_data = ['ner', 'pos', 'sc', 'meld', 'dmeld']
task_group_list = [('cola', 'qnli', 'qqp'), ('mrpc', 'cola', 'mnli'), ('sst', 'qnli', 'mrpc'), ('wnli', 'mrpc', 'mnli'),
                   ('mnli', 'rte', 'cola'), ('qnli', 'sst', 'rte'),
                   ('rte', 'mrpc', 'cola'), ('qqp', 'cola', 'wnli')]
auxiliary_ratio_list = [[9, 7, 3], [4, 6, 6], [6, 4, 2], [2, 7, 5], [3, 2, 7], [2, 1, 0], [4, 7, 4], [7, 0, 4]]
# task_group_list = [('cola', 'rte', 'dmeld'), ('mrpc', 'cola', 'mnli', 'rte'), ('sst', 'qnli'),
#                    ('wnli', 'qqp', 'mrpc', 'cola'),
#                    ('mnli', 'rte', 'dmeld'), ('qnli', 'sst', 'mnli'),
#                    ('rte', 'cola', 'mnli'), ('qqp', 'sst', 'wnli'),
#                    ('meld', 'qqp', 'mnli'), ('dmeld', 'qqp', 'mnli')]
# auxiliary_ratio_list = [[8, 6, 5], [1, 5, 2, 3], [1, 9], [2, 0, 9, 7], [5, 2, 7], [3, 1, 2], [4, 1, 9], [7, 3, 4],
#                         [5, 3, 6], [6, 4, 0]]
# task_group_list = [('cola', 'sc'), ('mrpc', 'qqp'), ('sst', 'mnli', 'qqp'),
#                    ('wnli', 'ner', 'stsb', 'sst'),
#                    ('mnli', 'mrpc', 'qnli'), ('qnli', 'mnli', 'wnli', 'sst'),
#                    ('rte', 'stsb', 'qnli'), ('qqp', 'wnli'), ('ner', 'qnli', 'rte'),
#                    ('pos', 'cola', 'rte'), ('sc', 'mnli', 'rte'), ('stsb', 'wnli')]
# auxiliary_ratio_list = [[6, 1], [2, 6], [8, 0, 6], [3, 9, 2, 9], [6, 4, 0], [5, 5, 6, 1], [1, 4, 7], [7, 8],
#                         [3, 9, 0], [8, 2, 0], [9, 0, 8], [5, 3]]

assert len(task_group_list) == len(auxiliary_ratio_list)

# set args
stage = 1  # 1 or 2
model = 'bert-base-cased'  # any other bert models
lr = '5e-5'

# for main exp, the gradient file path is similar with bert model type

python_script = 'train_with_mix_ratio.py' if stage == 2 else 'train.py'
method = 'autosem' if stage == 2 else 'autosem-p1'
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
    template += 'model_dir="checkpoints/{}/{}_{}/{}'.format(method, len(task_group_list),
                                                            model,
                                                            main_task)  # method/{task_num}_{model_type} or {exp_type}/{main_task_name}
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
    template += '--test_with_label ' if main_task in labeled_data else ""
    template += '\n'

shell_name = "{}_{}_{}.sh".format(method, len(task_group_list), model)

with open(shell_name, "w") as f:
    f.write(template)
