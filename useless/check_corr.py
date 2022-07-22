''' this script is used to load all tasks' head importance matrices and calculate the task-task, task-instance kendall correlation '''
import os
import warnings
import pickle
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# task_list = ["cola", "qqp", "wnli", "mrpc", "qnli", "rte", "mnli", "sst"]  # 8 task
# task_list = ["cola", "qqp", "wnli", "mrpc", "qnli", "rte", "mnli", "sst", 'meld', 'dmeld']  # 10 task
task_list = ["cola", "qqp", "wnli", "mrpc", "qnli", "rte", "mnli", "sst", 'stsb', 'pos', 'ner', 'sc']  # 12 task
# task_list = ["cola", "wnli", "mrpc", "qnli", "rte", "sst", ]
# task_list = ["mnli", "rte", "qqp", "qnli", "mrpc", "sst", "cola", "wnli", "meld", "dmeld", "ner", "pos", "sc", "stsb"]
# task_list = ["mrpc", "rte"]
# task_list = ["mnli", "rte", "qqp", "qnli", "mrpc", "sst", "cola", "wnli", "meld", "dmeld", "ner", "pos", "sc", "stsb"]
# assert len(task_list) == 9
bert_model_type = "bert-base-cased"
gradient_path = "./gradient_files"
gradient_path = os.path.join(gradient_path, bert_model_type)  # path should named with task name
task_head_matrices = dict()
instance_head_matrices = dict()

all_head_num = 144 if "large" not in bert_model_type else 24 * 16
corr_heat_map = np.zeros((len(task_list), len(task_list)), dtype=np.float32)
# read all head matrices to RAM
for task in tqdm(task_list):
    file_path = os.path.join(gradient_path, task)
    task_gradient_file = os.path.join(file_path, "gradient_layer_normal_task.pkl")
    instance_gradient_file = os.path.join(file_path, "gradient_layer_normal_ins.pkl")
    with open(task_gradient_file, "rb") as f:
        task_head_matrix = pickle.load(f)
    # with open(instance_gradient_file, "rb") as f:
    #     instance_head_matrix = pickle.load(f)

    task_head_matrices[task] = task_head_matrix.reshape(all_head_num, ).tolist()

for i, task in enumerate(task_list):
    for j, auxiliary_task in enumerate(task_list):
        main_vector = task_head_matrices[task]
        auxiliary_vector = task_head_matrices[auxiliary_task]
        corr, p_value = kendalltau(main_vector, auxiliary_vector)
        corr_heat_map[i][j] = corr  # observe this variable

# select auxiliary by eyes
for i, task in enumerate(task_list):
    print("==>for main task {}".format(task))
    corr_dic = dict()
    for j, auxiliary_task in enumerate(task_list):
        main_vector = task_head_matrices[task]
        auxiliary_vector = task_head_matrices[auxiliary_task]
        corr, p_value = kendalltau(main_vector, auxiliary_vector)
        corr_dic[auxiliary_task] = corr
    sorted_corr_dic=dict(sorted(corr_dic.items(),key=lambda x:x[1],reverse=True))
    print("sorted_corr:{}\n".format(sorted_corr_dic))
    wait = True
task_thres = 0.6
# select auxiliary by eyes
for i, task in enumerate(task_list):
    print("==>for main task {}".format(task))
    corr_dic = dict()
    for j, auxiliary_task in enumerate(task_list):
        main_vector = task_head_matrices[task]
        auxiliary_vector = task_head_matrices[auxiliary_task]
        corr, p_value = kendalltau(main_vector, auxiliary_vector)
        if corr >= task_thres:
            corr_dic[auxiliary_task] = corr
    sorted_corr_dic=dict(sorted(corr_dic.items(),key=lambda x:x[1],reverse=True))
    print("sorted_corr:{}\n".format(sorted_corr_dic))
    wait = True

ori_corr = pd.read_csv("./9_task_corr_weicheng.csv")
reference_heat_map = ori_corr.iloc[:, 1:].values

corr_corr, _ = kendalltau(corr_heat_map.reshape(corr_heat_map.shape[0] * corr_heat_map.shape[0], ).tolist(),
                          reference_heat_map.reshape(
                              reference_heat_map.shape[0] * reference_heat_map.shape[0], ).tolist())

print("task ranking corr between weicheng's corr matrix and mine")
for i in range(corr_heat_map.shape[0]):
    task_corr, _ = kendalltau(corr_heat_map[i].tolist(), reference_heat_map[i].tolist())
    print("task {}'s ranking similarity is {}".format(task_list[i], task_corr))
