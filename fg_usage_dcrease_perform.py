import os
import pickle
import json
import numpy as np

from matplotlib import pyplot as plt

#### for testing the subsampling, backend is bert-base-cased
# trial_base_candidate = {'qnli': ['sst'],
#                         'rte': ['mrpc', 'cola'],
#                         'mrpc': ['rte'],
#                         'sst': ['cola', 'qnli']}

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
#                         'wnli': ['rte']}
#### for bert-large-cased-6
# trial_base_candidate = {"cola": ['rte'],
#                         "mrpc": ['rte'],
#                         'qnli': ['rte', 'mrpc'],
#                         'rte': ['cola', 'mrpc', 'sst', 'qnli', 'wnli'],
#                         'sst': ['cola', 'mrpc', 'rte', 'qnli'],
#                         'wnli': ['rte', 'cola', 'qnli', 'mrpc']}
#### for bert-large-uncased-6
# trial_base_candidate = {"cola": ['sst'],
#                         "mrpc": ["rte","wnli"],
#                         'qnli': ['rte'],
#                         'rte': ["mrpc", "cola"],
#                         'sst': ['cola', 'rte', 'qnli', 'wnli'],
#                         'wnli': None}  # ['rte', 'cola', 'qnli', 'mrpc']
#### for bert-base-cased-8
# trial_base_candidate = {"cola": ['rte', 'sst'],
#                         "mrpc": ['rte'],
#                         'qnli': ['qqp', 'mnli'],
#                         'rte': ['mrpc', 'mnli', 'qqp', 'cola', 'wnli'],
#                         'sst': ['cola', 'qnli', 'mnli'],
#                         'wnli': None,
#                         'mnli': ['qqp'],
#                         'qqp': ['mnli']}
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
### for bert-base-cased-12
trial_base_candidate = {"cola": ['pos'],
                        "mrpc": ['rte'],
                        'qnli': ['qqp', 'ner'],
                        'rte': ['mrpc', 'mnli', 'stsb', 'ner'],
                        'sst': ['cola', 'qnli', 'mnli', 'qqp'],
                        'wnli': ["stsb", "pos"],
                        'mnli': ['qqp'],
                        'qqp': ['mnli'],
                        'stsb': ['wnli'],
                        'pos': ['stsb', 'wnli'],
                        'ner': ['stsb', 'pos'],
                        'sc': ["pos"]}
#### for testing the subsampling, backend is bert-base-cased
# trial_base_candidate = {'qnli': ['sst'],
#                         'rte': ['mrpc', 'cola'],
#                         'mrpc': ['rte'],
#                         'sst': ['cola', 'qnli']}

# subsampling_list = ['qnli','sst','rte', 'mrpc']  # for testing subsampling
subsampling_list = ["sc"]  ## TODO
whole_usage_perform = {'mrpc': 85.97,
                       'sst': 91.97,
                       'rte': 69.31,
                       'qnli': 90.99,
                       "wnli": 100,
                       "cola": 50}  # the best performance of GradTS-trial(100% usage)

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
temp2 = 0.7
while temp2 <= 0.96:
    percentage.append(temp2)
    temp2 += 0.03
## ==================================
bert_model_type = "bert-base-cased"  # bert-base-cased,bert-base-uncased,roberta-base,bert-large-uncased,roberta-large  TODO: modify it by hand
gradient_path = "./gradient_files/{}".format(bert_model_type)
# output_dir = "./checkpoints/{}/{}".format(bert_model_type, method_dict[method])
# output_dir = "./checkpoints/{}/{}".format(str(len(task_list)) + "_" + bert_model_type, method_dict[method])
num_layer = 24 if "large" in bert_model_type else 12
data_dir = "./data/canonical_data/{}".format(bert_model_type)
lr = '5e-5'
forced_shell_name = "subsampling_{}_{}_tune.sh".format(subsampling_list[0],
                                                       bert_model_type)
# for main exp, the gradient file path is similar with bert model type
# note epoch num

# python_script = 'train_with_corr.py'
if forced_shell_name is None:
    python_script = "train_with_corr_test.py"
else:
    python_script = "train_with_corr_test_subsample.py"

encoder_type = 1 if 'roberta' not in bert_model_type else 2

# score_template = "checkpoints/12_bert-base-cased/GradTS-fg/{}/{}/{}/eval_score.json"  ## TODO: you may need to change the exp
score_template = "checkpoints/12_bert-base-cased/GradTS-fg/{}/{}/{}/eval_score_best.json"  ## TODO: you may need to change the exp
save_path = "fg_usage/" + score_template.split("/")[1]
# auxiliary_tasks = trial_base_candidate[subsampling_list[0]]
percen2score = dict()
for main_task in subsampling_list:
    # chose auxiliary task threshold according to the training percentage
    print("===> for main task {}".format(main_task))
    auxiliary_tasks = trial_base_candidate[main_task]
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
        # score_file = score_template.format(main_task, "_".join([main_task] + auxiliary_tasks),
        #                                    "_".join(list(map(lambda x: str(x), auxiliary_thres))))
        score_file = score_template.format(main_task, "_".join([main_task] + auxiliary_tasks),
                                           "_".join(list(map(lambda x: str(x), auxiliary_thres))), main_task)
        try:
            with open(score_file, "r") as f:
                score_dict = json.load(f)["metrics"]
        except:
            percen2score_sorted = dict(sorted(percen2score.items(), key=lambda x: x[1], reverse=True))
            print()
            print("fail!")
            exit()
        avg_mec_score = np.mean(list(score_dict.values())[0])  ## TODO: be carefull about the metric scores
        percen2score[p] = avg_mec_score

    # percen2score = dict(reversed(list(percen2score.items())))
    percen2score[1.00] = whole_usage_perform[main_task]
    percen2score_sorted = dict(sorted(percen2score.items(), key=lambda x: x[1], reverse=True))
    plt.figure()
    plt.plot(percen2score.keys(), percen2score.values(), color='r', linewidth=2, markersize=10,
             alpha=0.8, mec='black', mew=0.7)
    plt.xlabel('auxiliary usage', fontsize=18)
    plt.ylabel('performance', fontsize=18)
    plt.title("{} auxiliary usage".format(main_task.upper()))
    plt.legend(numpoints=2, fontsize='large', loc=4)
    plt.grid(linestyle='--')

    plt.show()

    save_name = "{}.pdf".format(main_task)
    save_name = os.path.join(save_path, save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_name, format='pdf', bbox_inches='tight')

# save_name = "out.pdf"
#
