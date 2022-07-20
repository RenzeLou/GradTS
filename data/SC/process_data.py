''' process ner to json each instance has sequence and all token labels, and the format is similar to other SLCS task to unify'''
import copy
import json

all_label = []
train_file = []
dev_file = []
test_file = []

tokens = []
labels = []

with open("train.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        # print(i,line)
        if line == "\n":
            # achieve a sentence
            assert len(tokens) == len(labels)
            sentence = " ".join(tokens)
            if sentence != "":
                instance = dict()
                instance["index"], instance['seq1'], instance['seq2'], instance["label"] = str(i), tokens, -1, labels
                train_file.append(instance)
            tokens = []
            labels = []
            continue
        line_info = tuple(line.strip().split(" "))
        if len(line_info) == 3:
            token, _, tag = line_info
        else:
            token, tag = line_info
        tokens.append(token)
        if tag not in all_label:
            all_label.append(tag)
        lb = all_label.index(tag)
        labels.append(lb)
if len(tokens) != 0:
    sentence = " ".join(tokens)
    if sentence != "":
        instance = dict()
        instance["index"], instance['seq1'], instance['seq2'], instance["label"] = str(i), tokens, -1, labels
        train_file.append(instance)
tokens = []
labels = []

with open("dev.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        # print(i,line)
        if line == "\n":
            # achieve a sentence
            assert len(tokens) == len(labels)
            sentence = " ".join(tokens)
            if sentence != "":
                instance = dict()
                instance["index"], instance['seq1'], instance['seq2'], instance["label"] = str(
                    i), tokens, -1, labels
                dev_file.append(instance)
            tokens = []
            labels = []
            continue
        line_info = tuple(line.strip().split(" "))
        if len(line_info) == 3:
            token, _, tag = line_info
        else:
            token, tag = line_info
        tokens.append(token)
        if tag not in all_label:
            all_label.append(tag)
        lb = all_label.index(tag)
        labels.append(lb)
if len(tokens) != 0:
    sentence = " ".join(tokens)
    if sentence != "":
        instance = dict()
        instance["index"], instance['seq1'], instance['seq2'], instance["label"] = str(i), tokens, -1, labels
        dev_file.append(instance)
tokens = []
labels = []

with open("test.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        # print(i,line)
        if line == "\n":
            # achieve a sentence
            assert len(tokens) == len(labels)
            sentence = " ".join(tokens)
            if sentence != "":
                instance = dict()
                instance["index"], instance['seq1'], instance['seq2'], instance["label"] = str(
                    i), tokens, -1, labels
                test_file.append(instance)
            tokens = []
            labels = []
            continue
        line_info=tuple(line.strip().split(" "))
        if len(line_info)==3:
            token, _, tag = line_info
        else:
            token,tag = line_info
        tokens.append(token)
        if tag not in all_label:
            all_label.append(tag)
        lb = all_label.index(tag)
        labels.append(lb)
if len(tokens) != 0:
    sentence = " ".join(tokens)
    if sentence != "":
        instance = dict()
        instance["index"], instance['seq1'], instance['seq2'], instance["label"] = str(i), tokens, -1, labels
        test_file.append(instance)
tokens = []
labels = []

print("train num:", len(train_file))
print("dev num:", len(dev_file))
print("test num:", len(test_file))

with open("train_format.json", "w") as f:
    json.dump(train_file, f)
with open("dev_format.json", "w") as f:
    json.dump(dev_file, f)
with open("test_format.json", "w") as f:
    json.dump(test_file, f)
with open("data_info.json", "w") as f:
    lb2dict = dict()
    lb2dict["labels"] = dict([(str(i), str(lb)) for i, lb in enumerate(all_label)])
    json.dump(lb2dict, f)

print("all_labels", all_label,"all_label num", len(all_label))
