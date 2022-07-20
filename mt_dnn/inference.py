# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
from data_utils.metrics import calc_metrics
from mt_dnn.batcher import Collater
from data_utils.task_def import TaskType
import torch
from tqdm import tqdm
import pickle
import numpy as np


def extract_encoding(model, data, use_cuda=True):
    if use_cuda:
        model.cuda()
    sequence_outputs = []
    max_seq_len = 0
    for idx, (batch_info, batch_data) in enumerate(data):
        batch_info, batch_data = Collater.patch_data(use_cuda, batch_info, batch_data)
        sequence_output = model.encode(batch_info, batch_data)
        sequence_outputs.append(sequence_output)
        max_seq_len = max(max_seq_len, sequence_output.shape[1])

    new_sequence_outputs = []
    for sequence_output in sequence_outputs:
        new_sequence_output = torch.zeros(sequence_output.shape[0], max_seq_len, sequence_output.shape[2])
        new_sequence_output[:, :sequence_output.shape[1], :] = sequence_output
        new_sequence_outputs.append(new_sequence_output)

    return torch.cat(new_sequence_outputs)


def eval_model(model, data, metric_meta, device, with_label=True, label_mapper=None, task_type=TaskType.Classification):
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for (batch_info, batch_data) in data:
        batch_info, batch_data = Collater.patch_data(device, batch_info, batch_data)
        score, pred, gold = model.predict(batch_info, batch_data)
        predictions.extend(pred)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_info['uids'])

    if task_type == TaskType.Span:
        from experiments.squad import squad_utils
        golds = squad_utils.merge_answers(ids, golds)
        predictions, scores = squad_utils.select_answers(ids, predictions, scores)
    if with_label:
        metrics = calc_metrics(metric_meta, golds, predictions, scores, label_mapper)
    return metrics, predictions, scores, golds, ids


# def get_head_matrix(model, data, device, save_path):
#     attention_gradients = {}
#     nb_eval_steps = 0
#     for (batch_info, batch_data) in data:
#         batch_info, batch_data = Collater.patch_data(device, batch_info, batch_data)
#         attention_gradient, nb_eval_steps = model.update_no_grad(batch_info, batch_data, attention_gradients,
#                                                                   nb_eval_steps)
#         for k, v in attention_gradient.items():
#             if k in attention_gradients.keys():
#                 attention_gradients[k] += v
#             else:
#                 attention_gradients[k] = v
#     for k, v in attention_gradients.items():
#         attention_gradients[k] = v / nb_eval_steps
#
#     save_path += ".pkl" if not save_path.endswith(".pkl") else ""
#     pickle.dump(attention_gradients, open(save_path, "wb"))
#     print("Gradient saved in ", save_path)

# def avg_array(a, b):
#     ''' point-wise average two numpy array(2-D) '''
#     assert a.shape == b.shape
#     assert len(a.shape) == 2
#     at = a.reshape(*a.shape, 1)
#     bt = b.reshape(*b.shape, 1)
#     c = np.concatenate((at, bt), axis=2)
#     return np.mean(c, axis=2)
#
# def avg_array_list(array_list):

# def get_head_matrix(model, data, device, save_path):
#     attention_gradients = {}
#     nb_eval_steps = 0
#     for (batch_info, batch_data) in data:
#         attention_gradient_pass = {}
#         batch_info, batch_data = Collater.patch_data(device, batch_info, batch_data)
#         g, nb_eval_steps = model.update_no_grad(batch_info, batch_data, attention_gradient_pass,
#                                                                  nb_eval_steps)
#         for k, v in g.items():
#             # print("\n=======attention_gradients==========\n",attention_gradients)
#             # print("k",k)
#             # print("attention_gradients.keys()",list(attention_gradients.keys()))
#             if k in attention_gradients.keys():
#                 # print("attention_gradients[k]",attention_gradients[k])
#                 # print("type(attention_gradients[k])",type(attention_gradients[k]))
#                 attention_gradients[k].append(v.tolist())
#             else:
#                 # print("v.tolist()",v.tolist())
#                 attention_gradients[k] = [v.tolist()]
#     for k, v in attention_gradients.items():
#         attention_gradients[k] = np.mean(np.array(v),axis=0)
#
#     save_path += ".pkl" if not save_path.endswith(".pkl") else ""
#     pickle.dump(attention_gradients, open(save_path, "wb"))
#     print("Gradient saved in ", save_path)
def get_head_matrix(model, data, device, save_path):
    all_matrix_list = []
    nb_eval_steps = 0
    for (batch_info, batch_data) in data:
        attention_gradient_pass = {}
        batch_info, batch_data = Collater.patch_data(device, batch_info, batch_data)
        g, nb_eval_steps = model.update_no_grad(batch_info, batch_data, attention_gradient_pass,
                                                nb_eval_steps)
        assert type(g) == dict
        head_matrix_batch = avg_head(g)  # just avg K\Q\V, get head matrix
        all_matrix_list.append(head_matrix_batch.tolist())

    all_batch_matrices = np.array(all_matrix_list, dtype=np.float32)
    task_head_matrix = np.mean(all_batch_matrices, axis=0)
    gradient_matrix = layer_normal(task_head_matrix)

    save_path += ".pkl" if not save_path.endswith(".pkl") else ""
    pickle.dump(gradient_matrix, open(save_path, "wb"))
    print("Gradient saved in ", save_path)


def get_head_matrix_ins(model, data, device, save_path):
    # don't save the medium procession, transfer the head matrix directly
    all_head_matrix_ins = []
    for (batch_info, batch_data) in data:
        # tune on each training instance
        attention_gradients = {}
        batch_info, batch_data = Collater.patch_data(device, batch_info, batch_data)
        attention_gradients, _ = model.update_no_grad(batch_info, batch_data, attention_gradients, 0)
        assert type(attention_gradients) == dict
        head_matrix_ins = avg_head_layer_normal(attention_gradients)
        all_head_matrix_ins.append(head_matrix_ins.tolist())  # append sequentially
        # wait = True
    instance_num = len(all_head_matrix_ins)
    all_head_matrix_ins = np.array(all_head_matrix_ins, dtype=np.float32)

    save_path += ".pkl" if not save_path.endswith(".pkl") else ""
    pickle.dump(all_head_matrix_ins, open(save_path, "wb"))
    print("Gradient saved in ", save_path)
    print("Totally {} instances".format(instance_num))
    print("Head matrix shape:{}".format(all_head_matrix_ins.shape))


def avg_head_layer_normal(gradient_dict: dict, abs: bool = False):
    ''' average the head gradient dict and apply layer normalization, global scale'''
    model_keys = list(gradient_dict.keys())
    # max_layers = int(len(model_keys) / 3)
    gradient_matrix = []
    for layer in range(0, len(model_keys), 3):  # should Q, K, V order
        K_V = gradient_dict[model_keys[layer + 1]] + gradient_dict[model_keys[layer + 2]]  # key + value
        # take average.
        head_num = len(K_V)
        # head_vector = [np.mean(K_V[head_index]) for head_index in range(head_num)]
        if abs:
            head_vector = [np.abs(np.mean(K_V[head_index])) for head_index in range(head_num)]
        else:
            head_vector = [np.mean(K_V[head_index]) for head_index in range(head_num)]
        # TODO: layer sum normalization
        # for each batch gradient matrix([12,12]), we normalize every head in a layer with the sum of these heads
        head_vector = [head_value / float(sum(head_vector)) for head_value in head_vector]
        gradient_matrix.append(head_vector)  # add in final

    def norm(a: np.ndarray):
        a = (a - np.min(a)) / float(np.max(a) - np.min(a))
        return a

    # TODO: use min_max normalization, map the whole gradient matrix to [0,1]
    gradient_matrix = norm(np.array(gradient_matrix))

    return gradient_matrix


def avg_head(gradient_dict: dict, abs: bool = False):
    ''' just avg head, mainly be used to batch-wise head importance matrix average'''
    model_keys = list(gradient_dict.keys())
    # max_layers = int(len(model_keys) / 3)
    gradient_matrix = []
    for layer in range(0, len(model_keys), 3):  # should Q, K, V order
        K_V = gradient_dict[model_keys[layer + 1]] + gradient_dict[model_keys[layer + 2]]  # key + value
        # take average.
        head_num = len(K_V)
        # head_vector = [np.mean(K_V[head_index]) for head_index in range(head_num)]
        if abs:
            head_vector = [np.abs(np.mean(K_V[head_index])) for head_index in range(head_num)]
        else:
            head_vector = [np.mean(K_V[head_index]) for head_index in range(head_num)]
        # TODO: layer sum normalization
        # for each batch gradient matrix([12,12]), we normalize every head in a layer with the sum of these heads
        # head_vector = [head_value / float(sum(head_vector)) for head_value in head_vector]
        gradient_matrix.append(head_vector)  # add in final

    def norm(a: np.ndarray):
        a = (a - np.min(a)) / float(np.max(a) - np.min(a))
        return a

    # gradient_matrix = norm(np.array(gradient_matrix))

    return np.array(gradient_matrix, dtype=np.float32)


def layer_normal(gradient_matrix: np.array):
    ''' layer normalization and global scale the ori gradient matrix'''
    gradient_vectors = []
    for layer in range(gradient_matrix.shape[0]):  # should Q, K, V order
        line = (gradient_matrix[layer] / (np.sum(gradient_matrix[layer]))).tolist()
        gradient_vectors.append(line)

    def norm(a: np.ndarray):
        a = (a - np.min(a)) / float(np.max(a) - np.min(a))
        return a

    # TODO: use min_max normalization, map the whole gradient matrix to [0,1]
    gradient_matrix_new = norm(np.array(gradient_vectors, dtype=np.float32))

    return gradient_matrix_new
