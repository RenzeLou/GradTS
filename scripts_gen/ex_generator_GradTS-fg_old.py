''' this script can be used to generate shell to run mtdnn with the task selection result of AutoSem & GradTS '''
''' special candidate task chose from trial-based method '''
import os
import pickle
import numpy as np

labeled_data = ['ner', 'pos', 'sc', 'meld', 'dmeld']
method_dict = {1: "GradTS-thres", 2: "GradTS-trial", 3: "GradTS-fg"}

task_list = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp']  # 8 task   ## TODO: may need to change
# task_list = ['cola', 'mrpc', 'sst', 'rte', 'qnli', 'wnli']  # 6 task
# task_list = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp', 'meld', 'dmeld']  # 10 task
# task_list = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp', 'stsb', 'pos', 'ner', 'sc']  # 12 task

###########################  TODO: modify the trial_base_candidate according to its backend
#### for roberta-base-6
# trial_base_candidate = {"cola": ['sst'],
#                         "mrpc": ['cola'],
#                         'qnli': ['rte', 'cola'],
#                         'rte': ['cola', 'mrpc', 'sst'],
#                         'sst': ['cola'],
#                         'wnli': None}  # ['qnli']
##### for bert-base-uncased-6
# trial_base_candidate = {"cola": ['sst', 'qnli'],
#                         "mrpc": ['wnli'],
#                         'qnli': ['sst'],
#                         'rte': ['wnli', 'qnli'],
#                         'sst': ['cola'],
#                         'wnli': ["rte", "mrpc", "qnli", "cola"]}
##### for bert-base-cased-6
# trial_base_candidate = {"cola": ['rte', 'sst'],
#                         "mrpc": ['rte'],
#                         'qnli': ['sst'],
#                         'rte': ['mrpc', 'cola'],
#                         'sst': ['cola', 'qnli'],
#                         'wnli': ['rte', 'mrpc', 'qnli']}
#### for roberta-large-6
# trial_base_candidate = {"cola": ['sst', 'mrpc', 'rte'],
#                         "mrpc": ['sst', 'qnli'],
#                         'qnli': ['mrpc'],
#                         'rte': ['cola'],
#                         'sst': ['cola', 'mrpc'],
#                         'wnli': ['rte']}  # ['rte']
#### for bert-large-cased-6
# trial_base_candidate = {"cola": ['rte'],
#                         "mrpc": ['rte'],
#                         'qnli': ['rte', 'mrpc'],
#                         'rte': ['cola', 'mrpc', 'sst', 'qnli', 'wnli'],
#                         'sst': ['cola', 'mrpc', 'rte', 'qnli'],
#                         'wnli': ["rte", "cola", "qnli", "mrpc"]}
#### for bert-large-uncased-6
# trial_base_candidate = {"cola": ['sst'],
#                         "mrpc": ["rte","wnli"],
#                         'qnli': ['rte'],
#                         'rte': ["mrpc", "cola"],
#                         'sst': ['cola', 'rte', 'qnli', 'wnli'],
#                         'wnli': None}  # ['rte', 'cola', 'qnli', 'mrpc']
#### for bert-base-cased-8
trial_base_candidate = {"cola": ['rte', 'sst'],
                        "mrpc": ['rte'],
                        'qnli': ['qqp', 'mnli'],
                        'rte': ['mrpc', 'mnli', 'qqp', 'cola', 'wnli'],
                        'sst': ['cola', 'qnli', 'mnli'],
                        'wnli': None,
                        'mnli': ['qqp'],
                        'qqp': ['mnli']}
#### for bert-base-cased-10
# trial_base_candidate = {"cola": ['rte', 'sst'],
#                         "mrpc": ['rte'],
#                         'qnli': ['qqp', 'dmeld', 'mnli', 'meld'],
#                         'rte': ['mrpc', 'mnli', 'qqp', 'cola', 'wnli'],
#                         'sst': ['dmeld'],
#                         'wnli': ["rte", "mrpc", "mnli"],
#                         'mnli': ['qqp'],
#                         'qqp': ['mnli'],
#                         'meld': ['dmeld'],
#                         'dmeld': ['meld', 'sst']}
#### for bert-base-cased-12
# trial_base_candidate = {"cola": ['pos'],
#                         "mrpc": ['rte'],
#                         'qnli': ['qqp', 'ner'],
#                         'rte': ['mrpc', 'mnli', 'stsb', 'ner'],
#                         'sst': ['cola', 'qnli', 'mnli', 'qqp'],
#                         'wnli': ["stsb", "pos"],
#                         'mnli': ['qqp'],
#                         'qqp': ['mnli'],
#                         'stsb': ['wnli'],
#                         'pos': ['stsb', 'wnli'],
#                         'ner': ['stsb', 'pos'],
#                         'sc': ['pos']}
#### for testing the subsampling, backend is bert-base-cased
# trial_base_candidate = {'qnli': ['sst'],
#                         'rte': ['mrpc', 'cola'],
#                         'mrpc': ['rte'],
#                         'sst': ['cola', 'qnli']}
# subsampling_list = ['cola']  # for testing subsampling  ## TODO: mod
# subsampling_list = ['mrpc']
# subsampling_list = ["qnli"]
# subsampling_list = ['rte']
# subsampling_list = ['sst']
subsampling_list = ['sc']

# set args
method = 3  # 1,2,3
task_threshold = 0.9
# instance_thresholds = [0.47, 0.52, 0.58, 0.62]  # TODO: 0.4,0.42,0.43; list, if multi ins thres else single
instance_thresholds = [0.55]
# percentage = list(reversed(np.round(np.linspace(0.7, 0.95, 26), 2).tolist()))
percentage = []
## ==================================
# main coverage
temp = 0.7
while temp <= 0.96:
    percentage.append(temp)
    temp += 0.03
## ==================================
# expanded coverage
temp = 0.3
while temp <= 0.7:
    percentage.append(temp)
    temp += 0.03
## ==================================
# trial_num = 1
bert_model_type = "bert-base-cased"  # bert-base-cased,bert-base-uncased,roberta-base,bert-large-uncased,roberta-large  TODO: modify it by hand
gradient_path = "./gradient_files/{}".format(bert_model_type)
# output_dir = "./checkpoints/{}/{}".format(bert_model_type, method_dict[method])
output_dir = "./checkpoints/{}/{}".format(str(len(task_list)) + "_" + bert_model_type, method_dict[method])
num_layer = 24 if "large" in bert_model_type else 12
data_dir = "./data/canonical_data/{}".format(bert_model_type)
lr = '5e-5'  ## TODO: you may need to change lr when running wnli
forced_shell_name = "subsampling_{}_{}_{}_tune.sh".format(subsampling_list[0], len(trial_base_candidate),
                                                          bert_model_type)  # Todo: None, if normal exp
# forced_shell_name = None
# for main exp, the gradient file path is similar with bert model type
# note epoch num

# python_script = 'train_with_corr.py'
if forced_shell_name is None:
    python_script = "train_with_corr_test.py"
else:
    python_script = "train_with_corr_test_subsample.py"

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

multi_thres = False
### multi instance thres
# multi_thres = True
# for instance_threshold in instance_thresholds:
#     # training param
#     for task in task_list:
#         main_task = task
#         candidate_lis = [t for t in task_list if t != main_task]  # imply the exp type
#         candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
#         # check
#         if trial_base_candidate[main_task] is None:
#             print("skip task {}".format(main_task))
#             continue
#         for tt in trial_base_candidate[main_task]:
#             assert tt in task_list, "{} is invalid!".format(tt)
#         train_datasets = ",".join([main_task] + trial_base_candidate[main_task])
#         test_datasets = main_task if "mnli" not in main_task else "mnli_matched,mnli_mismatched"
#
#         # training param
#         template += 'python {} '.format(python_script)
#         template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
#             main_task, method, task_threshold, instance_threshold, gradient_path, candidate_set)
#         template += "--train_datasets {} --test_datasets {} ".format(train_datasets, test_datasets)
#         template += '--data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers ${NUM_LAYER} ' \
#                     '--epochs ${Epoch} --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} ' \
#                     '--answer_opt ${answer_opt} --optimizer ${optim} ' \
#                     '--grad_clipping ${grad_clipping} ' \
#                     '--global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} ' \
#                     '--learning_rate ${lr} '
#         template += '--test_with_label ' if main_task in labeled_data else ""
#         template += "--output_dir {} ".format(output_dir)
#         template += "--bert_model_type {} ".format(bert_model_type)
#         template += "--init_checkpoint {} ".format(bert_model_type)
#         template += '\n'

### single instance thres
# training param
# instance_threshold = instance_thresholds[0]
# for task in task_list:
#     main_task = task
#     candidate_lis = [t for t in task_list if t != main_task]  # imply the exp type
#     candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
#     # check
#     if trial_base_candidate[main_task] is None:
#         print("skip task {}".format(main_task))
#         continue
#     for tt in trial_base_candidate[main_task]:
#         assert tt in task_list, "{} is invalid!".format(tt)
#     train_datasets = ",".join([main_task] + trial_base_candidate[main_task])
#     test_datasets = main_task if "mnli" not in main_task else "mnli_matched,mnli_mismatched"
#
#     # training param
#     template += 'python {} '.format(python_script)
#     template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
#         main_task, method, task_threshold, instance_threshold, gradient_path, candidate_set)
#     template += "--train_datasets {} --test_datasets {} ".format(train_datasets, test_datasets)
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

# usage-performance
auxiliary_tasks = trial_base_candidate[subsampling_list[0]]
for main_task in subsampling_list:
    # chose auxiliary task threshold according to the training percentage
    print("===> for main task {}".format(main_task))
    for p in percentage:
        print("\n++ this is percentage {}".format(p))
        auxiliary_thres = []
        for task in auxiliary_tasks:
            corr_path = "gradient_files"
            corr_name = os.path.join(corr_path, bert_model_type, main_task, "gradient_correlation",
                                     "{}.pkl".format(task))
            with open(corr_name, "rb") as f:
                corr = pickle.load(f)
            corr_sorted = sorted(corr, reverse=True)
            choice = int(len(corr) * p)
            threshold = corr_sorted[choice]
            auxiliary_thres.append(threshold)
            print("for auxiliary task {}, use thres {}".format(task, threshold))
        # candidate_lis = [t for t in task_list if t != main_task]  # imply the exp type
        candidate_set = ",".join(auxiliary_tasks)  # imply the exp type
        train_datasets = ",".join([main_task] + trial_base_candidate[main_task])
        test_datasets = main_task if "mnli" not in main_task else "mnli_matched,mnli_mismatched"
        assert len(train_datasets.split(",")) == len(auxiliary_thres) + 1
        auxiliary_thres = list(map(lambda x: str(x), auxiliary_thres))
        instance_threshold = ",".join(auxiliary_thres)
        # training param
        template += 'python {} '.format(python_script)
        template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
            main_task, method, task_threshold, instance_threshold, gradient_path, candidate_set)
        template += "--train_datasets {} --test_datasets {} ".format(train_datasets, test_datasets)
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

## del ##
# main_task = "cola"
# candidate_lis = [t for t in task_list if t != main_task]  # imply the exp type
# candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
# # check
# for tt in trial_base_candidate[main_task]:
#     assert tt in task_list
# train_datasets = ",".join([main_task]+trial_base_candidate[main_task])
# test_datasets = main_task if "mnli" not in main_task else "mnli_matched,mnli_mismatched"
#
# # training param
# template += 'python {} '.format(python_script)
# template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
#     main_task, method, task_threshold, instance_threshold, gradient_path, candidate_set)
# template += "--train_datasets {} --test_datasets {} ".format(train_datasets, test_datasets)
# template += '--data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers ${NUM_LAYER} ' \
#             '--epochs ${Epoch} --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} ' \
#             '--answer_opt ${answer_opt} --optimizer ${optim} ' \
#             '--grad_clipping ${grad_clipping} ' \
#             '--global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} ' \
#             '--learning_rate ${lr} '
# template += '--test_with_label ' if main_task in labeled_data else ""
# template += "--output_dir {} ".format(output_dir)
# template += "--bert_model_type {} ".format(bert_model_type)
# template += "--init_checkpoint {} ".format(bert_model_type)
# template += '\n'
## del ##

## all tasks chosed  =========================================================
# for task in task_list:
#     main_task = task
#     candidate_lis = [t for t in task_list if t != main_task]  # imply the exp type
#     candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
#     # check
#     for tt in trial_base_candidate[main_task]:
#         assert tt in task_list
#     train_datasets = ",".join(t for t in task_list)
#     test_datasets = main_task if "mnli" not in main_task else "mnli_matched,mnli_mismatched"
#
#     # training param
#     template += 'python {} '.format(python_script)
#     template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
#         main_task, method, task_threshold, instance_threshold, gradient_path, candidate_set)
#     template += "--train_datasets {} --test_datasets {} ".format(train_datasets, test_datasets)
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

# ============================================================================================

if forced_shell_name is not None:
    shell_name = forced_shell_name
else:
    if not multi_thres:
        shell_name = "{}_{}_{}_{}.sh".format(method_dict[method], len(task_list), bert_model_type, instance_threshold)
    else:
        shell_name = "{}_{}_{}_{}.sh".format(method_dict[method], len(task_list), bert_model_type,
                                             "_".join(list(map(str, instance_thresholds))))

with open(shell_name, "w") as f:
    f.write(template)

print("\nshell name:", shell_name)
