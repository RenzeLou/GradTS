'''read and load the task selection results of GradTS-trial'''

import os
import json
import sys
import argparse
from unittest import result

parser = argparse.ArgumentParser()

parser.add_argument('-T','--tasks',type=str,default='cola,mrpc,sst,wnli,mnli,qnli,rte,qqp')
parser.add_argument('-B','--bert_model_type',type=str,default="bert-base-cased")

args = parser.parse_args()

def FindAllSuffix(path: str, suffix: str, tgt:str, verbose: bool = False) -> list:
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
            if suffix in file and tgt in file:
                file_path = os.path.join(root, file)
                result.append(file_path)
                if verbose:
                    print(file_path)

    return result

task_list = [t.strip() for t in args.tasks.split(',')]

root = 'checkpoints/{}_{}/GradTS-trial'.format(len(task_list),args.bert_model_type)

for task in task_list:
    root_path = os.path.join(root,task)
    results = FindAllSuffix(root_path,'json','score_best')
    
    best_score = -1
    best_selection = None
    for file in results:
        with open(file,'r',encoding='utf-8') as f:
            score = json.load(f)
            score = list(score['metrics'].values())[0]
            if score >= best_score:
                best_score = score
                best_selection = file.split('/')[-2].split('_')
    # print(results)
    # sys.exit(0)
    save_name = os.path.join(root_path,'best_task_selection.json')
    # print(save_name)
    with open(save_name,'w',encoding='utf-8') as tgt:
        json.dump(best_selection,tgt)
    print("For main task {}, the best task selection is \n{}".format(task,best_selection))