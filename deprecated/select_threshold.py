''' generate auxiliary data corr distribution '''
import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats, integrate
import seaborn as sns
import matplotlib.pyplot as plt

bert_model_type = "bert-base-cased"
# main_tasks = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp', "stsb", "ner", "pos", "sc", "meld", "dmeld"]
# main_tasks = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp']
main_tasks = ['cola', 'mrpc', 'sst', 'wnli', 'qnli', 'rte']
subsample_task = ["rte"]

for main_task in main_tasks:
    print("===> For main task {}".format(main_task))
    auxiliary_task = [t for t in main_tasks if t != main_task]
    corr_path = "gradient_files"
    save_path = "corr_distribution"
    save_path = os.path.join(save_path, bert_model_type, main_task)
    os.makedirs(save_path, exist_ok=True)
    for task in auxiliary_task:
        corr_name = os.path.join(corr_path, bert_model_type, main_task, "gradient_correlation", "{}.pkl".format(task))
        with open(corr_name, "rb") as f:
            corr = pickle.load(f)
            fig, ax = plt.subplots(figsize=(9, 9))
            # sns.set(font_scale=1.4)
            res = sns.distplot(corr, kde=False, fit=stats.gamma)
            sns_plot = res.get_figure()
            save_name = os.path.join(save_path, "{}.png".format(task))
            # sns_plot.savefig(save_name)
            plt.savefig(save_name)
            print("auxiliary task {} saved to {}".format(task, save_name))
            # sns.plt.show()
            # wait = True
# for main_task in subsample_task:
#     print("===> For main task {}".format(main_task))
#     auxiliary_task = [t for t in main_tasks if t != main_task]
#     corr_path = "gradient_files"
#     save_path = "corr_distribution"
#     save_path = os.path.join(save_path, str(len(main_tasks))+"_"+bert_model_type, main_task)
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