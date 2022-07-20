# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
# this file modified from train.py which can train the MT-DNN model with specific mix ratios
import argparse
import json
import pickle
import os
import warnings

''' used for training mt-dnn with head matrices correlation '''

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import random
import time
from datetime import datetime
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from pretrained_models import *
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from experiments.exp_def import TaskDefs
from mt_dnn.inference import eval_model, extract_encoding
from data_utils.log_wrapper import create_logger
from data_utils.task_def import EncoderModelType
from data_utils.utils import set_environment
from mt_dnn.batcher import SingleTaskDataset, MultiTaskDataset, Collater, MultiTaskBatchSampler, \
    DistMultiTaskBatchSampler, DistSingleTaskBatchSampler
from mt_dnn.batcher import DistTaskDataset
from mt_dnn.model import MTDNNModel


def model_config(parser):
    parser.add_argument('--update_bert_opt', default=0, type=int)
    parser.add_argument('--multi_gpu_on', type=bool, default=True)
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_num_turn', type=int, default=5)
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--max_answer_len', type=int, default=10)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=1, help='0,1')
    parser.add_argument('--pooler_actf', type=str, default='tanh',
                        help='tanh/relu/gelu')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--init_ratio', type=float, default=1)
    parser.add_argument('--encoder_type', type=int, default=1)
    parser.add_argument('--num_hidden_layers', type=int, default=-1)

    # BERT pre-training
    parser.add_argument('--bert_model_type', type=str, default='bert-base-cased')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15)
    parser.add_argument('--short_seq_prob', type=float, default=0.2)
    parser.add_argument('--max_predictions_per_seq', type=int, default=128)

    # bin samples
    parser.add_argument('--bin_on', action='store_true')
    parser.add_argument('--bin_size', type=int, default=64)
    parser.add_argument('--bin_grow_ratio', type=int, default=0.5)

    # dist training
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--world_size", type=int, default=1, help="For distributed training: world size")
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="6600")
    parser.add_argument("--backend", type=str, default="nccl")
    return parser


def data_config(parser):
    parser.add_argument('--log_file', default='mt-dnn-train.log', help='path for log file.')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--tensorboard_logdir', default='tensorboard_logdir')
    parser.add_argument("--init_checkpoint", default='bert-base-cased', type=str)
    parser.add_argument('--data_dir', default='data/canonical_data/bert-base-cased')
    parser.add_argument('--data_sort_on', action='store_true')
    parser.add_argument('--name', default='farmer')
    parser.add_argument('--task_def', type=str, default="experiments/glue/glue_task_def.yml")
    parser.add_argument('--train_datasets', type=str, default='cola,mrpc,rte')
    parser.add_argument('--test_datasets', default='cola')
    parser.add_argument('--glue_format_on', action='store_true')
    parser.add_argument('--mkd-opt', type=int, default=0,
                        help=">0 to turn on knowledge distillation, requires 'softlabel' column in input data")
    parser.add_argument('--do_padding', action='store_true')
    parser.add_argument("--test_with_label", action='store_true', help="the task which has test label")
    return parser


def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--save_per_updates', type=int, default=10000)
    parser.add_argument('--save_per_updates_on', action='store_true')
    parser.add_argument('--epochs', type=int, default=1)  # default tune 3 epochs
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')
    parser.add_argument('--adam_eps', type=float, default=1e-6)

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)

    # loading
    parser.add_argument("--model_ckpt", default='checkpoints/model_0.pt', type=str)
    parser.add_argument("--resume", action='store_true')

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    # parser.add_argument('--feature_based_on', action='store_true')
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='checkpoint/8_bert-base-cased')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--grad_accumulation_step', type=int, default=4)

    # fp 16
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # adv training
    parser.add_argument('--adv_train', action='store_true')
    # the current release only includes smart perturbation
    parser.add_argument('--adv_opt', default=0, type=int)
    parser.add_argument('--adv_norm_level', default=0, type=int)
    parser.add_argument('--adv_p_norm', default='inf', type=str)
    parser.add_argument('--adv_alpha', default=1, type=float)
    parser.add_argument('--adv_k', default=1, type=int)
    parser.add_argument('--adv_step_size', default=1e-5, type=float)
    parser.add_argument('--adv_noise_var', default=1e-5, type=float)
    parser.add_argument('--adv_epsilon', default=1e-6, type=float)
    parser.add_argument('--encode_mode', action='store_true', help="only encode test data")
    parser.add_argument('--debug', action='store_true', help="print debug info")

    # GradTS
    parser.add_argument('--method', type=int, default=1,
                        help="the GradTS strategies, include " +
                             "1: task-threshold based" +
                             "2: task-trial based" +
                             "3: instance-threshold based")
    parser.add_argument("--main_task", type=str, required=True)
    parser.add_argument("--task_threshold", type=float, default=0.47)
    parser.add_argument("--instance_threshold", type=float, default=0.62)
    parser.add_argument("--trial_num", type=int, default=1,
                        help="the greedy trial num that had been ran, also indicate the auxiliary task num")
    parser.add_argument("--gradient_path", type=str, default="./gradient_files/bert-base-cased")
    parser.add_argument("--candidate_set", type=str, default="cola,mrpc,sst,wnli,mnli,qnli,rte,qqp")
    parser.add_argument('--record_eval_best', action='store_true', help="if true,save the best score among all epochs")
    parser.add_argument("--force_run", action='store_true', help="if true, force scripts to run GradTS-trial")

    return parser


parser = argparse.ArgumentParser()
parser = data_config(parser)
parser = model_config(parser)
parser = train_config(parser)

args = parser.parse_args()

data_dir = args.data_dir
args.train_datasets = args.train_datasets.split(',')  ###########
args.test_datasets = args.test_datasets.split(',')  ##########

set_environment(args.seed, args.cuda)

task_defs = TaskDefs(args.task_def)
encoder_type = args.encoder_type


def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def evaluation(model, datasets, data_list, task_defs, output_dir='checkpoints', epoch=0, n_updates=-1, with_label=False,
               tensorboard=None, glue_format_on=False, test_on=False, device=None, logger=None, eval_score_file=None):
    # eval on rank 1
    print_message(logger, "Evaluation")
    test_prefix = "Test" if test_on else "Dev"
    if n_updates > 0:
        updates_str = "updates"
    else:
        updates_str = "epoch"
    updates = model.updates if n_updates > 0 else epoch
    # print("\n============================================evaluation====================================")
    # print(datasets)
    assert len(data_list) == len(datasets)
    for idx, dataset in enumerate(datasets):
        # print("\n======================this is the {} task {}".format(idx,data_list[idx]))
        prefix = dataset.split('_')[0]
        task_def = task_defs.get_task_def(prefix)
        label_dict = task_def.label_vocab
        # print("task_def.label_vocab\n",vars(task_def.label_vocab))
        test_data = data_list[idx]
        if test_data is not None:
            # print("test_data is not None\n",test_data)
            with torch.no_grad():
                test_metrics, test_predictions, test_scores, test_golds, test_ids = eval_model(model,
                                                                                               test_data,
                                                                                               metric_meta=task_def.metric_meta,
                                                                                               device=device,
                                                                                               with_label=with_label,
                                                                                               label_mapper=label_dict,
                                                                                               task_type=task_def.task_type)
            for key, val in test_metrics.items():
                if tensorboard:
                    tensorboard.add_scalar('{}/{}/{}'.format(test_prefix, dataset, key), val, global_step=updates)
                if isinstance(val, str):
                    print_message(logger, 'Task {0} -- {1} {2} -- {3} {4}: {5}'.format(dataset, updates_str, updates,
                                                                                       test_prefix, key, val), level=1)
                elif isinstance(val, float):
                    print_message(logger,
                                  'Task {0} -- {1} {2} -- {3} {4}: {5:.3f}'.format(dataset, updates_str, updates,
                                                                                   test_prefix, key, val), level=1)
                else:
                    test_metrics[key] = str(val)
                    print_message(logger, 'Task {0} -- {1} {2} -- {3} {4}: \n{5}'.format(dataset, updates_str, updates,
                                                                                         test_prefix, key, val),
                                  level=1)

            if args.local_rank in [-1, 0]:
                score_file = os.path.join(output_dir,
                                          '{}_{}_scores_{}_{}.json'.format(dataset, test_prefix.lower(), updates_str,
                                                                           updates))
                results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids,
                           'scores': test_scores}
                dump(score_file, results)
                if eval_score_file:
                    dump(eval_score_file, results)
                    best_eval_score_file = eval_score_file.rsplit(".", 1)[0] + "_best.json"
                    print("best eval score path:", best_eval_score_file)
                    best_score = list(test_metrics.values())[0]
                    print("best score:", best_score)
                    # exit()
                    save_flag = False
                    if not os.path.isfile(best_eval_score_file) or epoch == 0:
                        save_flag = True
                    else:
                        with open(best_eval_score_file, "r") as f:
                            last_best = json.load(f)
                        last_best_score = list(last_best["metrics"].values())[0]
                        save_flag = best_score > last_best_score
                    if save_flag:
                        print("the the best score:", best_score)
                        best_results = {"metrics": test_metrics}
                        dump(best_eval_score_file, best_results)
                    else:
                        print("do not save the current score %.5f" % best_score,
                              " last best score is %.5f" % last_best_score)
                if glue_format_on:
                    from experiments.glue.glue_utils import submit
                    official_score_file = os.path.join(output_dir,
                                                       '{}_{}_scores_{}.tsv'.format(dataset, test_prefix.lower(),
                                                                                    updates_str))
                    submit(official_score_file, results, label_dict)


def initialize_distributed(args):
    """Initialize torch.distributed."""
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    if os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'):
        # We are using (OpenMPI) mpirun for launching distributed data parallel processes
        local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        local_size = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))
        args.local_rank = local_rank
        args.rank = nodeid * local_size + local_rank
        args.world_size = num_nodes * local_size
    # args.batch_size = args.batch_size * args.world_size

    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    device = torch.device('cuda', args.local_rank)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6600')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)
    return device


def print_message(logger, message, level=0):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            do_logging = True
        else:
            do_logging = False
    else:
        do_logging = True
    if do_logging:
        if level == 1:
            logger.warning(message)
        else:
            logger.info(message)


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


def CheckRemainFile(target_file: list, exist_file: list) -> list:
    filter_for_file = lambda x: x.rsplit("/", 1)[-1]
    exist_file = list(map(filter_for_file, exist_file))
    train_need = []
    test_need = []
    dev_need = []
    for target_task in target_file:
        train_target = target_task + "_train.json"
        if not train_target in exist_file:
            train_need.append(train_target)
        if "mnli" not in target_task:
            test_target, dev_target = target_task + "_test.json", target_task + "_dev.json"
            if not test_target in exist_file:
                test_need.append(test_target)
            if not dev_target in exist_file:
                dev_need.append(dev_target)
        else:
            test_target, dev_target = target_task + "_matched_test.json", target_task + "_matched_dev.json"
            if not test_target in exist_file:
                test_need.append(test_target)
            if not dev_target in exist_file:
                dev_need.append(dev_target)
            test_target, dev_target = target_task + "_mismatched_test.json", target_task + "_mismatched_dev.json"
            if not test_target in exist_file:
                test_need.append(test_target)
            if not dev_target in exist_file:
                dev_need.append(dev_target)

    return train_need + test_need + dev_need


def generate_copy_subset(ori_data_dir: str, data_dir: str, need_file: str, corr_path: str, ins_thres: float):
    # with open(os.path.join(ori_data_dir, need_file), "r") as rf:
    #     origin = json.load(rf)
    ori_data = []
    ori_data_cnt = 0
    with open(os.path.join(ori_data_dir, need_file), 'r', encoding='utf-8') as reader:
        for line in reader:
            sample = json.loads(line)
            ori_data.append(sample)
            ori_data_cnt += 1
    if "test" in need_file or "dev" in need_file:
        # copy file directly
        with open(os.path.join(data_dir, need_file), "w", encoding="utf-8") as writer:
            for i, sample in enumerate(ori_data):
                writer.write('{}\n'.format(json.dumps(sample)))
        print("successfully copy file from {} to {}".format(os.path.join(ori_data_dir, need_file),
                                                            os.path.join(data_dir, need_file)))
    elif "train" in need_file:
        # launch subset
        new_data_cnt = 0
        task_name = need_file.rsplit("_")[0]
        corr_file = os.path.join(corr_path, "{}.pkl".format(task_name))
        with open(corr_file, 'rb') as f:
            ins_corr = pickle.load(f)
        assert len(ins_corr) == len(ori_data)
        with open(os.path.join(data_dir, need_file), "w", encoding="utf-8") as writer:
            for i, sample in enumerate(ori_data):
                if ins_corr[i] >= ins_thres:
                    writer.write('{}\n'.format(json.dumps(sample)))
                    new_data_cnt += 1
        print("{}/{} have been chosen, launch subset at {} successfully".format(new_data_cnt, ori_data_cnt,
                                                                                os.path.join(data_dir, need_file)))
        # save the chosen result
        with open(os.path.join(data_dir, "{}_subset.txt".format(task_name)), "w") as tf:
            tf.write("{}/{} have been chosen, launch subset at {} successfully".format(new_data_cnt, ori_data_cnt,
                                                                                       os.path.join(data_dir,
                                                                                                    need_file)))
    else:
        raise RuntimeError


def main():
    # set up dist
    device = torch.device("cuda")
    if args.local_rank > -1:
        device = initialize_distributed(args)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    main_metrix_lis = ["MCC", "F1", "SeqEval", "F1MAC", "Pearson"]
    main_score = -10000
    output_dir = args.output_dir  # modify the save path
    data_dir = args.data_dir

    candidate_task = args.candidate_set.split(",")

    if args.method == 1:
        print("GradTS task-threshold based exp begin!")
        gradient_path = os.path.join(args.gradient_path, args.main_task)
        corr_path = os.path.join(gradient_path, "gradient_correlation")
        task_corr_file = os.path.join(corr_path, "task_level_corr.pkl")
        with open(task_corr_file, "rb") as f:
            task_corr = pickle.load(f)
        # gradient_path = os.path.join(args.gradient_path, args.main_task)
        # corr_path = os.path.join(gradient_path, "gradient_correlation")
        # task_corr_file = os.path.join(corr_path, "task_level_corr.pkl")
        # candidate_task = args.candidate_set.split(",")
        # with open(task_corr_file, "rb") as f:
        #     task_corr = pickle.load(f)
        candidate = [task for task, corr in task_corr.items() if corr >= args.task_threshold and task in candidate_task]
        # print corr to check
        print("the main task is {} and threshold is {}".format(args.main_task, args.task_threshold))
        for task, corr in task_corr.items():
            if task in candidate:
                print("=> task {} added, corr is {}".format(task, corr))
        args.train_datasets = [args.main_task] + candidate
        args.test_datasets = []
        for i, task in enumerate(args.train_datasets):
            if i > 0:
                break  # only test main task
            if "mnli" in task:
                args.test_datasets += ["mnli_matched", "mnli_mismatched"]
            else:
                args.test_datasets.append(task)
        print("train_datasets: {} ; test_datasets:{}".format(args.train_datasets, args.test_datasets))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, args.main_task)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = os.path.join(output_dir, str(args.task_threshold))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        print("save checkpoint to {}".format(output_dir))
    elif args.method == 2:
        print("GradTS task-trial based exp begin!")
        # load latest result, if no latest result or run_flag == False, exit
        run_flag = False
        if not args.force_run:
            if len(args.train_datasets) == 2:
                print("train_datasets: {} ; test_datasets:{}".format(args.train_datasets, args.test_datasets))
                # make sure to run this trial
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                output_dir = os.path.join(output_dir, args.main_task)
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                output_dir = os.path.join(output_dir, "_".join(args.train_datasets))
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                print("save checkpoint to {}".format(output_dir))
            else:
                latest_path = os.path.join(output_dir, args.main_task)
                latest_path = os.path.join(latest_path, "_".join(args.train_datasets[:-1]))
                latest_file = os.path.join(latest_path, "eval_score.json")
                latest_continue_file = os.path.join(latest_path, "whether_continue.json")
                if not os.path.exists(latest_file):
                    warnings.warn("latest file don't exist, end this exp!")
                    exit(0)
                else:
                    with open(latest_continue_file, "r") as f:
                        latest_continue = json.load(f)
                    print("latest_continue_file:", latest_continue_file)
                    print("whether continue:", latest_continue)
                    run_flag = latest_continue["continue_flag"]

                    with open(latest_file, "r") as f:
                        latest_result = json.load(f)
                    latest_score = latest_result["metrics"]
                    if len(latest_score) == 1:
                        main_score = list(latest_score.values())[0]
                    else:
                        for k, v in latest_score.items():
                            if k in main_metrix_lis:
                                main_score = v
                    if not run_flag:
                        warnings.warn("eval score has already decrease, end this exp!")
                        # avoid the influence of existing file
                        current_continue = os.path.join(output_dir, args.main_task)
                        current_continue = os.path.join(current_continue, "_".join(args.train_datasets))
                        current_continue = os.path.join(current_continue, "whether_continue.json")
                        if os.path.isfile(current_continue):
                            warnings.warn(
                                "the 'whether_continue_flag' has existed in current direction,modify it to False to avoid any breaks!")
                            whether_continue = {"continue_flag": False}
                            with open(current_continue, "w") as f:
                                json.dump(whether_continue, f)
                        exit(0)
                    print("train_datasets: {} ; test_datasets:{}".format(args.train_datasets, args.test_datasets))
                    # make sure to run this trial
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    output_dir = os.path.join(output_dir, args.main_task)
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    output_dir = os.path.join(output_dir, "_".join(args.train_datasets))
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    print("save checkpoint to {}".format(output_dir))
        else:
            print("supplementary task, be forced to run!")
    elif args.method == 3:
        print("GradTS instance-threshold based exp begin!")
        data_dir = os.path.join(args.data_dir, "subset")
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        data_dir = os.path.join(data_dir, args.main_task)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        data_dir = os.path.join(data_dir, str(args.instance_threshold))
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            # generate subset of all candidate training data, the dev,test data will retain the original edition
            # TODO
        exit(0)
    else:
        raise NotImplementedError

    os.makedirs(output_dir, exist_ok=True)
    # output_dir = os.path.abspath(output_dir)
    log_path = output_dir + "/log.log"
    logger = create_logger(__name__, to_disk=True, log_file=log_path)

    opt = vars(args)
    # update data dir
    opt['data_dir'] = args.data_dir
    batch_size = args.batch_size
    print_message(logger, 'Launching the MT-DNN training')
    # return
    tasks = {}
    task_def_list = []
    dropout_list = []
    printable = args.local_rank in [-1, 0]

    train_datasets = []
    for dataset in args.train_datasets:
        prefix = dataset.split('_')[0]
        if prefix in tasks:
            continue
        task_id = len(tasks)
        tasks[prefix] = task_id
        task_def = task_defs.get_task_def(prefix)
        task_def_list.append(task_def)
        train_path = os.path.join(data_dir, '{}_train.json'.format(dataset))
        print_message(logger, 'Loading {} as task {}'.format(train_path, task_id))
        train_data_set = SingleTaskDataset(train_path, True, maxlen=args.max_seq_len, task_id=task_id,
                                           task_def=task_def, printable=printable)
        train_datasets.append(train_data_set)
    train_collater = Collater(dropout_w=args.dropout_w, encoder_type=encoder_type, soft_label=args.mkd_opt > 0,
                              max_seq_len=args.max_seq_len, do_padding=args.do_padding)
    multi_task_train_dataset = MultiTaskDataset(train_datasets)
    if args.local_rank != -1:
        multi_task_batch_sampler = DistMultiTaskBatchSampler(train_datasets, args.batch_size, args.mix_opt, args.ratio,
                                                             rank=args.local_rank, world_size=args.world_size)
    else:
        multi_task_batch_sampler = MultiTaskBatchSampler(train_datasets, args.batch_size, args.mix_opt, args.ratio,
                                                         bin_on=args.bin_on, bin_size=args.bin_size,
                                                         bin_grow_ratio=args.bin_grow_ratio)
    multi_task_train_data = DataLoader(multi_task_train_dataset, batch_sampler=multi_task_batch_sampler,
                                       collate_fn=train_collater.collate_fn, pin_memory=args.cuda)

    opt['task_def_list'] = task_def_list

    dev_data_list = []
    test_data_list = []
    test_collater = Collater(is_train=False, encoder_type=encoder_type, max_seq_len=args.max_seq_len,
                             do_padding=args.do_padding)
    for dataset in args.test_datasets:
        # print("dataset",dataset)
        prefix = dataset.split('_')[0]
        task_def = task_defs.get_task_def(prefix)
        task_id = tasks[prefix]
        task_type = task_def.task_type
        data_type = task_def.data_type

        dev_path = os.path.join(data_dir, '{}_dev.json'.format(dataset))
        dev_data = None
        # assert os.path.exists(dev_path)  # TODO:del it
        if os.path.exists(dev_path):
            dev_data_set = SingleTaskDataset(dev_path, False, maxlen=args.max_seq_len, task_id=task_id,
                                             task_def=task_def, printable=printable)
            if args.local_rank != -1:
                dev_data_set = DistTaskDataset(dev_data_set, task_id)
                single_task_batch_sampler = DistSingleTaskBatchSampler(dev_data_set, args.batch_size_eval,
                                                                       rank=args.local_rank, world_size=args.world_size)
                dev_data = DataLoader(dev_data_set, batch_sampler=single_task_batch_sampler,
                                      collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
            else:
                dev_data = DataLoader(dev_data_set, batch_size=args.batch_size_eval,
                                      collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
        dev_data_list.append(dev_data)

        test_path = os.path.join(data_dir, '{}_test.json'.format(dataset))
        test_data = None
        if os.path.exists(test_path):
            test_data_set = SingleTaskDataset(test_path, False, maxlen=args.max_seq_len, task_id=task_id,
                                              task_def=task_def, printable=printable)
            if args.local_rank != -1:
                test_data_set = DistTaskDataset(test_data_set, task_id)
                single_task_batch_sampler = DistSingleTaskBatchSampler(test_data_set, args.batch_size_eval,
                                                                       rank=args.local_rank, world_size=args.world_size)
                test_data = DataLoader(test_data_set, batch_sampler=single_task_batch_sampler,
                                       collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
            else:
                test_data = DataLoader(test_data_set, batch_size=args.batch_size_eval,
                                       collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
        test_data_list.append(test_data)

    print("\n=====dev_data len:+", len(dev_data_list))
    print("\n=====test_data len:+", len(test_data_list))

    print_message(logger, '#' * 20)
    print_message(logger, opt)
    print_message(logger, '#' * 20)

    # div number of grad accumulation.
    num_all_batches = args.epochs * len(multi_task_train_data) // args.grad_accumulation_step
    print_message(logger, '############# Gradient Accumulation Info #############')
    print_message(logger, 'number of step: {}'.format(args.epochs * len(multi_task_train_data)))
    print_message(logger, 'number of grad grad_accumulation step: {}'.format(args.grad_accumulation_step))
    print_message(logger, 'adjusted number of step: {}'.format(num_all_batches))
    print_message(logger, '############# Gradient Accumulation Info #############')

    init_model = args.init_checkpoint
    state_dict = None

    if os.path.exists(init_model):
        if encoder_type == EncoderModelType.BERT or \
                encoder_type == EncoderModelType.DEBERTA or \
                encoder_type == EncoderModelType.ELECTRA:
            state_dict = torch.load(init_model, map_location=device)
            config = state_dict['config']
        elif encoder_type == EncoderModelType.ROBERTA or encoder_type == EncoderModelType.XLM:
            model_path = '{}/model.pt'.format(init_model)
            state_dict = torch.load(model_path, map_location=device)
            arch = state_dict['args'].arch
            arch = arch.replace('_', '-')
            if encoder_type == EncoderModelType.XLM:
                arch = "xlm-{}".format(arch)
            # convert model arch
            from data_utils.roberta_utils import update_roberta_keys
            from data_utils.roberta_utils import patch_name_dict
            state = update_roberta_keys(state_dict['model'], nlayer=state_dict['args'].encoder_layers)
            state = patch_name_dict(state)
            literal_encoder_type = EncoderModelType(opt['encoder_type']).name.lower()
            config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
            config = config_class.from_pretrained(arch).to_dict()
            state_dict = {'state': state}
    else:
        if opt['encoder_type'] not in EncoderModelType._value2member_map_:
            raise ValueError("encoder_type is out of pre-defined types")
        literal_encoder_type = EncoderModelType(opt['encoder_type']).name.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
        config = config_class.from_pretrained(init_model).to_dict()

    config['attention_probs_dropout_prob'] = args.bert_dropout_p
    config['hidden_dropout_prob'] = args.bert_dropout_p
    config['multi_gpu_on'] = opt["multi_gpu_on"]
    if args.num_hidden_layers > 0:
        config['num_hidden_layers'] = args.num_hidden_layers

    opt.update(config)

    model = MTDNNModel(opt, device=device, state_dict=state_dict, num_train_step=num_all_batches)
    if args.resume and args.model_ckpt:
        print_message(logger, 'loading model from {}'.format(args.model_ckpt))
        model.load(args.model_ckpt)

    #### model meta str
    headline = '############# Model Arch of MT-DNN #############'
    ### print network
    print_message(logger, '\n{}\n{}\n'.format(headline, model.network))

    # dump config
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as writer:
        writer.write('{}\n'.format(json.dumps(opt)))
        writer.write('\n{}\n{}\n'.format(headline, model.network))

    print_message(logger, "Total number of params: {}".format(model.total_param))

    # don't use tensorboard
    tensorboard = None
    args.tensorboard = None
    if args.tensorboard:
        args.tensorboard_logdir = os.path.join(args.output_dir, args.tensorboard_logdir)
        tensorboard = SummaryWriter(log_dir=args.tensorboard_logdir)

    if args.encode_mode:
        for idx, dataset in enumerate(args.test_datasets):
            prefix = dataset.split('_')[0]
            test_data = test_data_list[idx]
            with torch.no_grad():
                encoding = extract_encoding(model, test_data, use_cuda=args.cuda)
            torch.save(encoding, os.path.join(output_dir, '{}_encoding.pt'.format(dataset)))
        return

    # record the time consumption
    begin_time = time.time()
    for epoch in range(0, args.epochs):
        print_message(logger, 'At epoch {}'.format(epoch), level=1)
        start = datetime.now()

        for i, (batch_meta, batch_data) in enumerate(multi_task_train_data):
            batch_meta, batch_data = Collater.patch_data(device, batch_meta, batch_data)
            task_id = batch_meta['task_id']
            model.update(batch_meta, batch_data)

            if (model.updates) % (args.log_per_updates) == 0 or model.updates == 1:
                ramaining_time = \
                    str((datetime.now() - start) / (i + 1) * (len(multi_task_train_data) - i - 1)).split('.')[0]
                if args.adv_train and args.debug:
                    debug_info = ' adv loss[%.5f] emb val[%.8f] eff_perturb[%.8f] ' % (
                        model.adv_loss.avg,
                        model.emb_val.avg,
                        model.eff_perturb.avg
                    )
                else:
                    debug_info = ' '
                print_message(logger, 'Task [{0:2}] updates[{1:6}] train loss[{2:.5f}]{3}remaining[{4}]'.format(task_id,
                                                                                                                model.updates,
                                                                                                                model.train_loss.avg,
                                                                                                                debug_info,
                                                                                                                ramaining_time))
                if args.tensorboard:
                    tensorboard.add_scalar('train/loss', model.train_loss.avg, global_step=model.updates)

            if args.save_per_updates_on and ((model.local_updates) % (
                    args.save_per_updates * args.grad_accumulation_step) == 0) and args.local_rank in [-1, 0]:
                model_file = os.path.join(output_dir, 'model_{}_{}.pt'.format(epoch, model.updates))
                evaluation(model, args.test_datasets, dev_data_list, task_defs, output_dir, epoch,
                           n_updates=args.save_per_updates, with_label=True, tensorboard=tensorboard,
                           glue_format_on=args.glue_format_on, test_on=False, device=device, logger=logger)
                if args.test_with_label:
                    print("test with label!!!")
                evaluation(model, args.test_datasets, test_data_list, task_defs, output_dir, epoch,
                           n_updates=args.save_per_updates, with_label=args.test_with_label, tensorboard=tensorboard,
                           glue_format_on=args.glue_format_on, test_on=True, device=device, logger=logger)
                print_message(logger, 'Saving mt-dnn model to {}'.format(model_file))
                # model.save(model_file)

        # save the final eval score for the convenience of greedy trial
        eval_score_file = os.path.join(output_dir, "eval_score.json")

        evaluation(model, args.test_datasets, dev_data_list, task_defs, output_dir, epoch, with_label=True,
                   tensorboard=tensorboard, glue_format_on=args.glue_format_on, test_on=False, device=device,
                   logger=logger, eval_score_file=eval_score_file)
        if args.test_with_label:
            print("test with label!!!")
        evaluation(model, args.test_datasets, test_data_list, task_defs, output_dir, epoch,
                   with_label=args.test_with_label,
                   tensorboard=tensorboard, glue_format_on=args.glue_format_on, test_on=True, device=device,
                   logger=logger)
        print_message(logger, '[new test scores at {} saved.]'.format(epoch))
        # if args.local_rank in [-1, 0]:
        #     model_file = os.path.join(output_dir, 'model_{}.pt'.format(epoch))
        #     model.save(model_file)
    if args.tensorboard:
        tensorboard.close()

    end_time = time.time()
    run_time = (end_time - begin_time) / 60.0
    time_con = "\nTotal time consumption: %.3f (minutes)" % run_time
    print_message(logger, time_con)

    if args.method == 2:
        # trial-based should additionally save "whether continue "
        if not args.record_eval_best:
            now_file = os.path.join(output_dir, "eval_score.json")
        else:
            now_file = os.path.join(output_dir, "eval_score_best.json")
        with open(now_file, "r") as f:
            now_result = json.load(f)
        now_score = now_result["metrics"]
        main_score_now = -9999999
        if len(now_score) == 1:
            main_score_now = list(now_score.values())[0]
        else:
            for k, v in now_score.items():
                if k in main_metrix_lis:
                    main_score_now = v
        continue_flag = main_score_now >= main_score
        whether_continue = {"continue_flag": continue_flag}
        with open(os.path.join(output_dir, "whether_continue.json"), "w") as f:
            json.dump(whether_continue, f)
        print("next exp whether continue:{}".format(whether_continue))


if __name__ == '__main__':
    main()
