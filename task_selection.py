''' contain all task selection results (used for supplementary exp, that is, 8 exp) '''
task_list = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp']  # 8 task

# AutoSem
## Bi-LSTM
autosem_p1_lstm = {
    "cola": ["wnli", "mnli"],
    "mrpc": ["mnli", "rte"],
    "mnli": ["mrpc", 'cola', 'rte'],
    "qnli": ["wnli", "mnli"],
    "qqp": ["mnli"],
    "rte": ["qqp", "mnli"],
    "sst": ["mnli", 'mrpc', 'wnli'],
    "wnli": ["mnli", "cola", "qnli"]
}
autosem_p2_lstm = {
    "cola": [20, 0, 5],
    "mrpc": [9, 4, 1],
    "mnli": [8, 5, 0, 8],
    "qnli": [20, 0, 5],
    "qqp": [1, 4],
    "rte": [1, 5, 5],
    "sst": [13, 5, 0, 0],
    "wnli": [2, 5, 4, 0]
}
## bert-base-cased
autosem_p1_bert_base = {
    "cola": ["qnli", "qqp"],
    "mrpc": ["cola", "mnli"],
    "mnli": ["rte", "sst", "cola"],
    "qnli": ["sst", "rte"],
    "qqp": ["cola", "wnli"],
    "rte": ["mrpc", "cola"],
    "sst": ["qnli", "mrpc"],
    "wnli": ["mrpc", "mnli"]
}
autosem_p2_bert_base = {
    "cola": [9, 7, 3],
    "mrpc": [4, 6, 6],
    "mnli": [3, 2, 0, 7],
    "qnli": [2, 1, 0],
    "qqp": [7, 0, 4],
    "rte": [4, 7, 4],
    "sst": [6, 4, 2],
    "wnli": [2, 7, 5]
}
## bert-large-cased
autosem_p1_bert_large = {
    "cola": ["qqp", "rte"],
    "mrpc": ["wnli", "qqp"],
    "mnli": ["cola", "rte"],
    "qnli": ["mnli", "cola", "rte"],
    "qqp": ["rte", "sst", "qnli"],
    "rte": ["sst", "mrpc", "cola"],
    "sst": ["rte", "qqp", "mrpc"],
    "wnli": ["qqp", "mrpc"]
}
autosem_p2_bert_large = {
    "cola": [2, 1, 1],
    "mrpc": [5, 8, 8],
    "mnli": [7, 1, 8],
    "qnli": [2, 5, 5, 4],
    "qqp": [2, 4, 2, 0],
    "rte": [2, 7, 4, 4],
    "sst": [2, 2, 7, 9],
    "wnli": [1, 4, 6]
}

# GradTS
## bert-base-cased
GradTS_thres_bert_base = {
    "cola": ["rte", "sst"],
    "mrpc": ["rte", "mnli", "qqp", "wnli"],
    "mnli": ["qqp", "rte", "mrpc", "qnli"],
    "qnli": ["qqp", "mnli"],
    "qqp": ["mnli", "mrpc", "qnli", "rte"],
    "rte": ["mrpc", "mnli", "qqp", "cola", "wnli"],
    "sst": ["cola"],
    "wnli": ["rte", "mrpc"]
}  # with task_thres == 0.6
GradTS_trial_bert_base = {
    "cola": ["rte", "sst"],
    "mrpc": ["rte"],
    "mnli": ["qqp"],
    "qnli": ["qqp", "mnli"],
    "qqp": ["mnli"],
    "rte": ["mrpc", "mnli", "qqp", "cola", "wnli"],
    "sst": ["cola", "qnli", "mnli"],
    "wnli": ["rte"]
}
GradTS_fg_bert_base = {
    "cola": 0.4,
    "mrpc": 0.4,
    "mnli": 0.43,
    "qnli": 0.42,
    "qqp": 0.42,
    "rte": 0.42,
    "sst": 0.4,
    "wnli": 0.4
}
## bert-large-cased
GradTS_thres_bert_large = {
    "cola": ["rte", "sst", "qnli"],
    "mrpc": ["rte", "qnli", "sst"],
    "mnli": ["qqp", "qnli"],
    "qnli": ["rte", "mnli", "qqp", "mrpc", "sst", "cola"],
    "qqp": ["mnli", "qnli"],
    "rte": ["cola", "qnli", "mrpc", "sst"],
    "sst": ["cola", "rte", "qnli", "mrpc"],
    "wnli": []
}  # with task_thres == 0.62
GradTS_trial_bert_large = {
    "cola": ["rte", "sst"],
    "mrpc": ["rte", "qnli"],
    "mnli": ["qqp"],
    "qnli": ["rte"],
    "qqp": ["mnli", "qnli"],
    "rte": ["cola", "qnli"],
    "sst": ["cola", "rte", "qnli"],
    "wnli": ["rte", "cola"]
}
GradTS_fg_bert_large = {
    "cola": [""],
    "mrpc": [""],
    "mnli": [""],
    "qnli": [""],
    "qqp": [""],
    "rte": [""],
    "sst": [""],
    "wnli": [""]
}
