import json
import pandas as pd
import argparse
import os
from transformers import *

all_data = ["wnli", "stsb", "sst", "snli", "rte", "qqp", "qnli", "mrpc", "mnli_matched", "mnli_mismatched", "cola"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bert-base-uncased", required=True)
# parser.add_argument("--data", type=str, required=True)

args, unparsed = parser.parse_known_args()
if unparsed:
    raise ValueError(unparsed)

out = args.model
model = args.model
tokenizer = BertTokenizer.from_pretrained(model)

if not os.path.exists(out):
    os.mkdir(out)

for data in all_data:
    print('process data {} of model {}'.format(data, model))
    source = data
    if "mnli" in data:
        train_data = "mnli"
    else:
        train_data = data
    train = pd.read_csv('{}_train.tsv'.format(train_data), header=0, delimiter='\t')
    dev = pd.read_csv('{}_dev.tsv'.format(source), header=0, delimiter='\t')
    test = pd.read_csv('{}_test.tsv'.format(source), header=0, delimiter='\t')

    train_file = '{}/{}_train.json'.format(out, source)
    if not os.path.isfile(train_file):
        text = train['text'].tolist()
        labels = train['labels'].tolist()
        with open(train_file, 'w') as f:
            for i in range(len(text)):
                line_dict = {}
                line_dict["uid"] = str(i)
                line_dict["label"] = labels[i]
                if type(text[i]) == str:
                    tokenization_res = tokenizer.encode_plus(text[i].lower())
                    line_dict["token_id"] = tokenization_res['input_ids']
                    line_dict["type_id"] = tokenization_res['token_type_ids']
                    json.dump(line_dict, f)
                    f.write('\n')

    dev_file = '{}/{}_dev.json'.format(out, source)
    if not os.path.isfile(dev_file):
        text = dev['text'].tolist()
        labels = dev['labels'].tolist()
        with open(dev_file, 'w') as f:
            for i in range(len(text)):
                line_dict = {}
                line_dict["uid"] = str(i)
                line_dict["label"] = labels[i]
                if type(text[i]) == str:
                    tokenization_res = tokenizer.encode_plus(text[i].lower())
                    line_dict["token_id"] = tokenization_res['input_ids']
                    line_dict["type_id"] = tokenization_res['token_type_ids']
                    json.dump(line_dict, f)
                    f.write('\n')

    test_file = '{}/{}_test.json'.format(out, source)
    if not os.path.isfile(test_file):
        text = test['text'].tolist()
        labels = test['labels'].tolist()
        with open(test_file, 'w') as f:
            for i in range(len(text)):
                line_dict = {}
                line_dict["uid"] = str(i)
                line_dict["label"] = labels[i]
                if type(text[i]) == str:
                    tokenization_res = tokenizer.encode_plus(text[i].lower())
                    line_dict["token_id"] = tokenization_res['input_ids']
                    line_dict["type_id"] = tokenization_res['token_type_ids']
                    json.dump(line_dict, f)
                    f.write('\n')
