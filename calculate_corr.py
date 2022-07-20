''' this script is used to load all tasks' head importance matrices and calculate the task-task, task-instance kendall correlation '''
import argparse
import os
import warnings
import pickle
from tqdm import tqdm, trange

import numpy as np
from scipy.stats import kendalltau

parser = argparse.ArgumentParser()
parser.add_argument("--bert_model_type", type=str, required=True)
parser.add_argument("--task_list", type=str, required=True)
args, unparsed = parser.parse_known_args()
if unparsed:
    raise ValueError(unparsed)

task_list = args.task_list.split(",")
bert_model_type = args.bert_model_type
# task_list = ["mnli", "rte", "qqp", "qnli", "mrpc", "sst", "cola", "wnli", "meld", "dmeld", "ner", "pos", "sc", "stsb"]
# assert len(task_list) == 14
# task_list = ["rte", "qnli", "mrpc", "sst", "cola", "wnli"]  # 6 task
# assert len(task_list) == 6

# bert_model_type = "roberta-large"  # bert-base-cased,roberta-base,bert-large-uncased,bert-large-cased,roberta-large
gradient_path = "./gradient_files"

gradient_path = os.path.join(gradient_path,bert_model_type)  # path should named with task name

task_head_matrices = dict()
instance_head_matrices = dict()

total_head_num = 24*16 if "large" in bert_model_type else 12*12

# read all head matrices to RAM
for task in tqdm(task_list):
    file_path = os.path.join(gradient_path, task)
    task_gradient_file = os.path.join(file_path, "gradient_layer_normal_task.pkl")
    instance_gradient_file = os.path.join(file_path, "gradient_layer_normal_ins.pkl")
    with open(task_gradient_file, "rb") as f:
        task_head_matrix = pickle.load(f)
    with open(instance_gradient_file, "rb") as f:
        instance_head_matrix = pickle.load(f)

    print("shape:",task_head_matrix.shape)
    task_head_matrices[task] = task_head_matrix.reshape(total_head_num, ).tolist()
    new_instance_matrix = []
    for i in range(instance_head_matrix.shape[0]):
        new_instance_matrix.append(instance_head_matrix[i].reshape(total_head_num, ).tolist())
    instance_head_matrices[task] = new_instance_matrix

# generate task-task corr
print('\n'+'+' * 5 + "task-level corr begin!" + '+' * 5)
for task, main_vector in task_head_matrices.items():
    save_name = "task_level_corr.pkl"
    file_path = os.path.join(gradient_path, task)
    file_path = os.path.join(file_path, "gradient_correlation")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    save_path = os.path.join(file_path, save_name)
    task_level_corr = dict()
    for auxiliary_task, auxiliary_vector in task_head_matrices.items():
        if auxiliary_task == task:
            continue
        corr, p_value = kendalltau(main_vector, auxiliary_vector)
        task_level_corr[auxiliary_task] = corr

    with open(save_path, "wb") as f:
        pickle.dump(task_level_corr, f)
    print("task {}'s corr has been save to {}".format(task, save_path))

# generate task-instance corr
print('\n' + '+' * 5 + "instance-level corr begin!" + '+' * 5)
for task, main_vector in task_head_matrices.items():
    file_path = os.path.join(gradient_path, task)
    file_path = os.path.join(file_path, "gradient_correlation")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    for auxiliary_task, auxiliary_vectors in instance_head_matrices.items():
        if auxiliary_task == task:
            continue
        save_name = "{}.pkl".format(auxiliary_task)
        save_path = os.path.join(file_path, save_name)
        if os.path.isfile(save_path):
            warnings.warn("there is already {} file, you'd better to check it".format(save_path))
        ins_corr = []
        for auxiliary_vector in auxiliary_vectors:
            corr, p_value = kendalltau(main_vector, auxiliary_vector)
            ins_corr.append(corr)

        with open(save_path, "wb") as f:
            pickle.dump(ins_corr, f)

        print("task {} - {}'s instance corr has been save to {}".format(task, auxiliary_task, save_path))
