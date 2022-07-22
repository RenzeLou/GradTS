''' based on the auxiliary task selection results from GradTS-trial
use the subset of auxiliary tasks
'''
import os
import pickle
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('-T','--tasks',type=str,default='cola,mrpc,sst,wnli,mnli,qnli,rte,qqp')
parser.add_argument('-B','--bert_model_type',type=str,default="bert-base-cased")
parser.add_argument('-L','--labeled_data',type=str,default='ner,pos,meld,dmeld',
                    help = 'There are only dev set for glue tasks. ' + 
                    'If there is any task with labeled test data in your task list, make sure mention it.')
parser.add_argument('--thres',type=float,default=0.55)

args = parser.parse_args()

task_list = [t.strip() for t in args.tasks.split(',')]
labeled_data = [t.strip() for t in args.labeled_data.split(',')]
method_dict = {1: "GradTS-thres", 2: "GradTS-trial", 3: "GradTS-fg"}

root = 'checkpoints/{}_{}/GradTS-trial'.format(len(task_list),args.bert_model_type)
trial_base_candidate = dict()

for task in task_list:
    root_path = os.path.join(root,task)
    try:
        with open(root_path+'/best_task_selection.json','r',encoding='utf-8') as f:
            task_order = json.load(f)
    except FileNotFoundError:
        print('{} is not found, pls run the read_trial_results.py in advance'.format(root_path+'/best_task_selection.json'))
    trial_base_candidate[task] = task_order[1:]
    
print(trial_base_candidate)
#### for bert-base-cased-8
# trial_base_candidate = {"cola": ['rte', 'sst'],
#                         "mrpc": ['rte'],
#                         'qnli': ['qqp', 'mnli'],
#                         'rte': ['mrpc', 'mnli', 'qqp', 'cola', 'wnli'],
#                         'sst': ['cola', 'qnli', 'mnli'],
#                         'wnli': None,
#                         'mnli': ['qqp'],
#                         'qqp': ['mnli']}

# subsampling_list = ['cola']  # for testing subsampling 
# subsampling_list = ['mrpc']
# subsampling_list = ["qnli"]
# subsampling_list = ['rte']
# subsampling_list = ['sst']
# subsampling_list = ['sc']

# set args
method = 3  # 1,2,3
task_threshold = 0.9
# instance_thresholds = [0.47, 0.52, 0.58, 0.62]  # TODO: 0.4,0.42,0.43; list, if multi ins thres else single
# instance_thresholds = [0.55]
instance_thresholds = [args.thres]
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
bert_model_type = args.bert_model_type  # bert-base-cased,bert-base-uncased,roberta-base,bert-large-uncased,roberta-large  TODO: modify it by hand
gradient_path = "./gradient_files/{}".format(bert_model_type)
# output_dir = "./checkpoints/{}/{}".format(bert_model_type, method_dict[method])
output_dir = "./checkpoints/{}/{}".format(str(len(task_list)) + "_" + bert_model_type, method_dict[method])
num_layer = 24 if "large" in bert_model_type else 12
data_dir = "./data/canonical_data/{}".format(bert_model_type)
lr = '5e-5'  ## TODO: you may need to change lr when running wnli
# forced_shell_name = "subsampling_{}_{}_{}_tune.sh".format(subsampling_list[0], len(trial_base_candidate),
#                                                           bert_model_type)  # Todo: None, if normal exp
forced_shell_name = None
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
template += 'echo "export CUDA_VISIBLE_DEVICES=$gpu"\necho "epoch num=$Epoch"\nexport CUDA_VISIBLE_DEVICES=${gpu}\ntstr=$(date +"%FT%H%M")\n'
# template += 'MODEL_ROOT="checkpoints"\nBERT_PATH="{}"\nDATA_DIR="data/canonical_data/{}"\n'.format(model, model)
template += 'answer_opt=1\noptim="adamax"\ngrad_clipping=0\nglobal_grad_clipping=1\nlr="{}"\ngrad_accumulation_step=4\n\n'.format(
    lr)

multi_thres = False

# single instance thres
## training param
instance_threshold = instance_thresholds[0]
for task in task_list:
    main_task = task
    candidate_lis = [t for t in task_list if t != main_task]  # imply the exp type
    candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
    # check
    if trial_base_candidate[main_task] is None:
        print("skip task {}".format(main_task))
        continue
    for tt in trial_base_candidate[main_task]:
        assert tt in task_list, "{} is invalid!".format(tt)
    train_datasets = ",".join([main_task] + trial_base_candidate[main_task])
    test_datasets = main_task if "mnli" not in main_task else "mnli_matched,mnli_mismatched"

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
