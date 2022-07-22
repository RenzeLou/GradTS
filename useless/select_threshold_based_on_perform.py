import os
import pickle
import numpy as np
import pandas as pd
import json
from scipy import stats, integrate
import seaborn as sns
import matplotlib.pyplot as plt


def FindAllSuffix(path: str, suffix: str, verbose: bool = False) -> list:
    ''' find all files have specific suffix under the path

    :param path: target path
    :param suffix: file suffix. e.g. ".json"/"json"
    :param verbose: whether print the found path
    :return: a list contain all corresponding file path (relative path)
    '''
    result = []
    if not suffix.startswith("."):
        suffix = "." + suffix
    for root, dirs, files in os.walk(path, topdown=False):
        # print(root, dirs, files)
        for file in files:
            if suffix in file:
                file_path = os.path.join(root, file)
                result.append(file_path)
                if verbose:
                    print(file_path)

    return result


metric_dic = {"mrpc": "F1", "qqp": "F1", "qnli": "ACC", "rte": "ACC", "sst": "ACC", "mnli": "ACC", "wnli": "ACC",
              "cola": "MCC", "stsb": "Pearson", "pos": "SeqEval", "ner": "SeqEval", "sc": "SeqEval", "meld": "F1MAC",
              "dmeld": "F1MAC"}
bert_model_type = "bert-base-cased"
main_tasks = "rte"
auxiliary_task = ["mrpc","cola"]
# thresholds = [0.45,0.47,0.51,0.57]
thresholds = list(reversed(np.round(np.linspace(0.7, 0.95, 26), 2).tolist()))
task_num = 6

score_path = os.path.join(
    "checkpoints/{}/GradTS-fg/{}/{}".format(str(task_num)+"_"+bert_model_type, main_tasks, "_".join([main_tasks] + auxiliary_task)))

all_performance = []
for threshold in thresholds:
    score_path_spe = os.path.join(score_path, str(threshold))
    score_name = os.path.join(score_path_spe, "eval_score.json")
    with open(score_name, "r") as f:
        score_dic = json.load(f)
        score = score_dic["metrics"][metric_dic[main_tasks]]
        all_performance.append(score)

plt.figure()
plt.plot(thresholds, all_performance, color="r", label='intra', linewidth=2, markersize=10,
         alpha=0.8, mec='black', mew=0.7)
plt.xlabel('instance-threshold', fontsize=18)
plt.ylabel('score', fontsize=18)
plt.title("{}-{}".format(bert_model_type,main_tasks))
plt.legend(numpoints=2, fontsize='large',loc=4)
plt.grid(linestyle='--')

plt.show()
save_name = "out.pdf"
# plt.savefig(os.path.join(path, save_name), format='pdf', bbox_inches='tight')

# for main_task in main_tasks:
#     print("===> For main task {}".format(main_task))
#     auxiliary_task = [t for t in main_tasks if t != main_task]
#     corr_path = "gradient_files"
#     save_path = "corr_distribution"
#     save_path = os.path.join(save_path, bert_model_type, main_task)
#     os.makedirs(save_path, exist_ok=True)
#     for task in auxiliary_task:
#         corr_name = os.path.join(corr_path, bert_model_type, main_task, "gradient_correlation", "{}.pkl".format(task))
#         with open(corr_name, "rb") as f:
#             corr = pickle.load(f)
#             fig, ax = plt.subplots(figsize=(9, 9))
#             # sns.set(font_scale=1.4)
#             res = sns.distplot(corr, kde=False, fit=stats.gamma)
#             sns_plot = res.get_figure()
#             save_name = os.path.join(save_path, "{}.png".format(task))
#             # sns_plot.savefig(save_name)
#             plt.savefig(save_name)
#             print("auxiliary task {} saved to {}".format(task, save_name))
#             # sns.plt.show()
#             # wait = True
