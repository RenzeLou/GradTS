''' use this scripts to process the five additional datasets for mt-dnn, that is MELD,D-MELD and three SL task '''
''' when running these five tasks, we need to add argument '--test_with_label', that directly generate test score rather than pred label '''
# due to the setting of our experiments, we only use bert-base-cased
import sys
import os
import json
import argparse
import tqdm

from transformers import AutoTokenizer, BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocessing the additional five datasets.')
    parser.add_argument('--model', type=str, default='bert-base-cased',
                        help='can be any hugging-face bert model, but bert-base-cased here by default.')
    parser.add_argument('--do_lower_case', type=bool, default=False)
    parser.add_argument('--do_padding', action='store_true')
    parser.add_argument('--root_dir', type=str, default='data')
    # parser.add_argument('--task_def', type=str, default="experiments/glue/glue_task_def.yml")
    parser.add_argument("--data_name", type=str, default="SC")
    parser.add_argument("--bpe_pad", type=bool, default=True,
                        help="for SL task, use this arg to keep additional bpe token")
    # all data name:
    # meld, d-meld, pos, ner, sc

    args = parser.parse_args()
    return args


def read_meld(data_dir, tokenizer: BertTokenizer):
    # return a tokenized dict list
    train_file = open(os.path.join(data_dir, "train_format.json"), "r", encoding="utf-8")
    dev_file = open(os.path.join(data_dir, "dev_format.json"), "r", encoding="utf-8")
    test_file = open(os.path.join(data_dir, "test_format.json"), "r", encoding="utf-8")

    train_data = json.load(train_file)
    dev_data = json.load(dev_file)
    test_data = json.load(test_file)

    train_feature = []
    for i, item in enumerate(train_data):
        token_list = tokenizer.tokenize(item['seq1'])
        token_id_list = tokenizer.convert_tokens_to_ids(token_list)
        token_id_list = tokenizer.build_inputs_with_special_tokens(token_ids_0=token_id_list)
        token_type_list = tokenizer.create_token_type_ids_from_sequences(token_id_list)
        attention_list = [1] * len(token_type_list)  # no pad
        new_item = {"uid": i, "label": item["label"], "token_id": token_id_list, "type_id": token_type_list,
                    "attention_mask": attention_list}
        train_feature.append(new_item)

    dev_feature = []
    for i, item in enumerate(dev_data):
        token_list = tokenizer.tokenize(item['seq1'])
        token_id_list = tokenizer.convert_tokens_to_ids(token_list)
        token_id_list = tokenizer.build_inputs_with_special_tokens(token_ids_0=token_id_list)
        token_type_list = tokenizer.create_token_type_ids_from_sequences(token_id_list)
        attention_list = [1] * len(token_type_list)  # no pad
        new_item = {"uid": i, "label": item["label"], "token_id": token_id_list, "type_id": token_type_list,
                    "attention_mask": attention_list}
        dev_feature.append(new_item)

    test_feature = []
    for i, item in enumerate(test_data):
        token_list = tokenizer.tokenize(item['seq1'])
        token_id_list = tokenizer.convert_tokens_to_ids(token_list)
        token_id_list = tokenizer.build_inputs_with_special_tokens(token_ids_0=token_id_list)
        token_type_list = tokenizer.create_token_type_ids_from_sequences(token_id_list)
        attention_list = [1] * len(token_type_list)  # no pad
        new_item = {"uid": i, "label": item["label"], "token_id": token_id_list, "type_id": token_type_list,
                    "attention_mask": attention_list}
        test_feature.append(new_item)

    train_file.close()
    dev_file.close()
    test_file.close()

    return train_feature, dev_feature, test_feature


def read_sl(data_dir, tokenizer: BertTokenizer):
    # return a tokenized dict list
    train_file = open(os.path.join(data_dir, "train_format.json"), "r", encoding="utf-8")
    dev_file = open(os.path.join(data_dir, "dev_format.json"), "r", encoding="utf-8")
    test_file = open(os.path.join(data_dir, "test_format.json"), "r", encoding="utf-8")

    data_info_file = open(os.path.join(data_dir, "data_info.json"), "r", encoding="utf-8")

    train_data = json.load(train_file)
    dev_data = json.load(dev_file)
    test_data = json.load(test_file)

    data_info = json.load(data_info_file)
    label_num = len(data_info["labels"].items())

    train_feature = []
    for i, item in enumerate(train_data):
        seq = item['seq1']
        lbs = item['label']
        assert len(seq) == len(lbs)
        token_list = []
        label_list = []
        for idx, tk in enumerate(seq):
            lb = lbs[idx]
            tked_list = tokenizer.tokenize(tk)
            tked = tked_list[0]  ## assert taking the first tk to represent the ori token
            token_list.append(tked)
            label_list.append(lb)
        token_id_list = tokenizer.convert_tokens_to_ids(token_list)
        token_id_list = tokenizer.build_inputs_with_special_tokens(token_ids_0=token_id_list)
        label_list = [label_num] + label_list + [label_num + 1]
        assert len(label_list) == len(token_id_list)
        token_type_list = tokenizer.create_token_type_ids_from_sequences(token_id_list)
        # attention_list = [1] * len(token_type_list)  # no pad
        new_item = {"uid": i, "label": label_list, "token_id": token_id_list, "type_id": token_type_list}
        train_feature.append(new_item)

    dev_feature = []
    for i, item in enumerate(dev_data):
        seq = item['seq1']
        lbs = item['label']
        assert len(seq) == len(lbs)
        token_list = []
        label_list = []
        for idx, tk in enumerate(seq):
            lb = lbs[idx]
            tked_list = tokenizer.tokenize(tk)
            tked = tked_list[0]  ## assert taking the first tk to represent the ori token
            token_list.append(tked)
            label_list.append(lb)
        token_id_list = tokenizer.convert_tokens_to_ids(token_list)
        token_id_list = tokenizer.build_inputs_with_special_tokens(token_ids_0=token_id_list)
        label_list = [label_num] + label_list + [label_num + 1]
        assert len(label_list) == len(token_id_list)
        token_type_list = tokenizer.create_token_type_ids_from_sequences(token_id_list)
        # attention_list = [1] * len(token_type_list)  # no pad
        new_item = {"uid": i, "label": label_list, "token_id": token_id_list, "type_id": token_type_list}
        dev_feature.append(new_item)

    test_feature = []
    for i, item in enumerate(test_data):
        seq = item['seq1']
        lbs = item['label']
        assert len(seq) == len(lbs)
        token_list = []
        label_list = []
        for idx, tk in enumerate(seq):
            lb = lbs[idx]
            tked_list = tokenizer.tokenize(tk)
            tked = tked_list[0]  ## assert taking the first tk to represent the ori token
            token_list.append(tked)
            label_list.append(lb)
        token_id_list = tokenizer.convert_tokens_to_ids(token_list)
        token_id_list = tokenizer.build_inputs_with_special_tokens(token_ids_0=token_id_list)
        label_list = [label_num] + label_list + [label_num + 1]
        assert len(label_list) == len(token_id_list)
        token_type_list = tokenizer.create_token_type_ids_from_sequences(token_id_list)
        # attention_list = [1] * len(token_type_list)  # no pad
        new_item = {"uid": i, "label": label_list, "token_id": token_id_list, "type_id": token_type_list}
        test_feature.append(new_item)

    train_file.close()
    dev_file.close()
    test_file.close()

    data_info_file.close()

    return train_feature, dev_feature, test_feature


def read_sl_new(data_dir, tokenizer: BertTokenizer):
    ''' the only difference of this procedure is that we don't ignore the additional token generated by BPE '''
    # return a tokenized dict list
    train_file = open(os.path.join(data_dir, "train_format.json"), "r", encoding="utf-8")
    dev_file = open(os.path.join(data_dir, "dev_format.json"), "r", encoding="utf-8")
    test_file = open(os.path.join(data_dir, "test_format.json"), "r", encoding="utf-8")

    data_info_file = open(os.path.join(data_dir, "data_info.json"), "r", encoding="utf-8")

    train_data = json.load(train_file)
    dev_data = json.load(dev_file)
    test_data = json.load(test_file)

    data_info = json.load(data_info_file)
    label_num = len(data_info["labels"].items())

    train_feature = []
    for i, item in enumerate(train_data):
        seq = item['seq1']
        lbs = item['label']
        assert len(seq) == len(lbs)
        token_list = []
        label_list = []
        for idx, tk in enumerate(seq):
            tked_list = tokenizer.tokenize(tk)
            tked = tked_list  # take all the BPE token
            token_list += tked
            lb = [lbs[idx]] + [-1] * (len(tked) - 1)  # append the additional tokens' label with -1
            label_list += lb
        assert len(token_list) == len(label_list)
        token_id_list = tokenizer.convert_tokens_to_ids(token_list)
        token_id_list = tokenizer.build_inputs_with_special_tokens(token_ids_0=token_id_list)
        label_list = [label_num] + label_list + [label_num + 1]
        assert len(label_list) == len(token_id_list)
        token_type_list = tokenizer.create_token_type_ids_from_sequences(token_id_list)
        # attention_list = [1] * len(token_type_list)  # no pad
        new_item = {"uid": i, "label": label_list, "token_id": token_id_list, "type_id": token_type_list}
        train_feature.append(new_item)

    dev_feature = []
    for i, item in enumerate(dev_data):
        seq = item['seq1']
        lbs = item['label']
        assert len(seq) == len(lbs)
        token_list = []
        label_list = []
        for idx, tk in enumerate(seq):
            tked_list = tokenizer.tokenize(tk)
            tked = tked_list  # take all the BPE token
            token_list += tked
            lb = [lbs[idx]] + [-1] * (len(tked) - 1)  # append the additional tokens' label with -1
            label_list += lb
        assert len(token_list) == len(label_list)
        token_id_list = tokenizer.convert_tokens_to_ids(token_list)
        token_id_list = tokenizer.build_inputs_with_special_tokens(token_ids_0=token_id_list)
        label_list = [label_num] + label_list + [label_num + 1]
        assert len(label_list) == len(token_id_list)
        token_type_list = tokenizer.create_token_type_ids_from_sequences(token_id_list)
        # attention_list = [1] * len(token_type_list)  # no pad
        new_item = {"uid": i, "label": label_list, "token_id": token_id_list, "type_id": token_type_list}
        dev_feature.append(new_item)

    test_feature = []
    for i, item in enumerate(test_data):
        seq = item['seq1']
        lbs = item['label']
        assert len(seq) == len(lbs)
        token_list = []
        label_list = []
        for idx, tk in enumerate(seq):
            tked_list = tokenizer.tokenize(tk)
            tked = tked_list  # take all the BPE token
            token_list += tked
            lb = [lbs[idx]] + [-1] * (len(tked) - 1)  # append the additional tokens' label with -1
            label_list += lb
        assert len(token_list) == len(label_list)
        token_id_list = tokenizer.convert_tokens_to_ids(token_list)
        token_id_list = tokenizer.build_inputs_with_special_tokens(token_ids_0=token_id_list)
        label_list = [label_num] + label_list + [label_num + 1]
        assert len(label_list) == len(token_id_list)
        token_type_list = tokenizer.create_token_type_ids_from_sequences(token_id_list)
        # attention_list = [1] * len(token_type_list)  # no pad
        new_item = {"uid": i, "label": label_list, "token_id": token_id_list, "type_id": token_type_list}
        test_feature.append(new_item)

    train_file.close()
    dev_file.close()
    test_file.close()

    data_info_file.close()

    return train_feature, dev_feature, test_feature


def main(args):
    # hyper param
    root = os.path.join(args.root_dir, args.data_name)
    assert os.path.exists(root), "there is no such dir of {}".format(root)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    canonical_data_root = os.path.join(args.root_dir, "canonical_data")
    mt_dnn_root = os.path.join(canonical_data_root, args.model)
    if not os.path.isdir(mt_dnn_root):
        os.makedirs(mt_dnn_root)

    data_dir = os.path.join("data", args.data_name)

    # task_defs = TaskDefs(args.task_def)
    #
    # for task in task_defs.get_task_names():
    #     task_def = task_defs.get_task_def(task)
    #     logger.info("Task %s" % task)
    #     for split_name in task_def.split_names:
    #         file_path = os.path.join(root, "%s_%s.tsv" % (task, split_name))
    #         if not os.path.exists(file_path):
    #             logger.warning("File %s doesnot exit")
    #             sys.exit(1)
    #         rows = load_data(file_path, task_def)
    #         dump_path = os.path.join(mt_dnn_root, "%s_%s.json" % (task, split_name))
    #         logger.info(dump_path)
    #         build_data(
    #             rows,
    #             dump_path,
    #             tokenizer,
    #             task_def.data_type,
    #             lab_dict=task_def.label_vocab)

    train_feature, dev_feature, test_feature = None, None, None
    if args.data_name in ["MELD", "D-MELD"]:
        train_feature, dev_feature, test_feature = read_meld(data_dir, tokenizer)
    else:
        train_feature, dev_feature, test_feature = read_sl(data_dir, tokenizer) if not args.bpe_pad else read_sl_new(
            data_dir, tokenizer)

    task_name = args.data_name.replace("-", "").lower()
    with open(os.path.join(mt_dnn_root, "{}_train.json".format(task_name)), "w", encoding="utf-8") as writer:
        for i, feature in enumerate(train_feature):
            writer.write('{}\n'.format(json.dumps(feature)))
        print("{} train data dumped successfully".format(i))
    with open(os.path.join(mt_dnn_root, "{}_dev.json".format(task_name)), "w", encoding="utf-8") as writer:
        for i, feature in enumerate(dev_feature):
            writer.write('{}\n'.format(json.dumps(feature)))
        print("{} dev data dumped successfully".format(i))
    with open(os.path.join(mt_dnn_root, "{}_test.json".format(task_name)), "w", encoding="utf-8") as writer:
        for i, feature in enumerate(test_feature):
            writer.write('{}\n'.format(json.dumps(feature)))
        print("{} test data dumped successfully".format(i))

    print("{} data process successfully!".format(args.data_name))


if __name__ == '__main__':
    args = parse_args()
    main(args)
