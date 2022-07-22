# tasks_list = ["ner", "pos", "sc", "rte", "qqp", "qnli", "mrpc", "sst", "cola", "wnli", "mnli", "meld", "dmeld", "stsb"] # all 14 tasks, only used in bert-based-cased
tasks_list = ["mnli", "rte", "qqp", "qnli", "mrpc", "sst", "cola", "wnli"]  # 8 tasks
# tasks_list = ["rte", "qnli", "mrpc", "sst", "cola", "wnli"]  # 6 tasks
# tasks_list = ["meld", "dmeld","stsb"]
# tasks_list += ["ner","pos"]
# tasks_list += ["sc"]
# device = 3
train_batch_size = 8  # tune batch size
eval_batch_size = 1  # batch size of getting gradient matrix, instance level don't need it, owe to that we just forward instance one by one
# is_with_label = True
bert_model_type = "bert-base-cased"  # bert-base-uncased,roberta-base,bert-large-uncased,bert-large-cased,roberta-large
data_dir = "data/canonical_data/" + bert_model_type
encoder_type = 2 if "roberta" in bert_model_type else 1
num_hidden_layers = 24 if "large" in bert_model_type else 12


shell_name = "tune_all_level_{}.sh".format(bert_model_type)
# shell_name = "tune_task_level_{}_again.sh".format(bert_model_type)
# shell_name = "tune_task_level_{}.sh".format(bert_model_type)

# all level
command = ""
command += 'gpu=$1\necho "export CUDA_VISIBLE_DEVICES=$gpu"\nexport CUDA_VISIBLE_DEVICES=${gpu}\ntrain_batch=$2\n'
for task in tasks_list:
    # command += "CUDA_VISIBLE_DEVICES=" + str(device) + " "
    command += "python instance_level_matrix_tune.py "
    # command += "--test_with_label " if is_with_label else ""
    command += "--bert_model_type {} --encoder_type {} --num_hidden_layers {} ".format(bert_model_type, encoder_type,
                                                                                       num_hidden_layers)
    command += "--batch_size ${train_batch} "
    command += "--batch_size_eval {} --init_checkpoint {} ".format(1, bert_model_type)
    command += "--answer_opt 1 --optimizer adamax "
    command += "--train_datasets {} --data_dir {} --force\n".format(task, data_dir)
for task in tasks_list:
    # command += "CUDA_VISIBLE_DEVICES=" + str(device) + " "
    command += "python task_level_matrix_tune.py "
    # command += "--test_with_label " if is_with_label else ""
    command += "--batch_size ${train_batch} "
    command += "--batch_size_eval {} --init_checkpoint {} ".format(eval_batch_size, bert_model_type)
    command += "--bert_model_type {} --encoder_type {} --num_hidden_layers {} ".format(bert_model_type, encoder_type,
                                                                                       num_hidden_layers)
    command += "--answer_opt 1 --optimizer adamax "
    command += "--train_datasets {} --data_dir {} --force\n".format(task, data_dir)
# run observe_gradient.py to get gradient matrix
# command += "python observe_gradient.py --task_name {}\n".format(",".join(tasks_list))
command += "python calculate_corr.py --bert_model_type {} --task_list {}\n".format(bert_model_type,
                                                                                   ",".join(tasks_list))
with open(shell_name, "w") as f:
    f.write(command)


# =========================> instance level <=========================

# shell_name = "tune_ins_level_{}.sh".format(bert_model_type)

# command = ""
# command += 'gpu=$1\necho "export CUDA_VISIBLE_DEVICES=$gpu"\nexport CUDA_VISIBLE_DEVICES=${gpu}\ntrain_batch=$2\n'
# for task in tasks_list:
#     # command += "CUDA_VISIBLE_DEVICES=" + str(device) + " "
#     command += "python instance_level_matrix_tune.py "
#     # command += "--test_with_label " if is_with_label else ""
#     command += "--bert_model_type {} --encoder_type {} --num_hidden_layers {} ".format(bert_model_type, encoder_type,
#                                                                                        num_hidden_layers)
#     command += "--batch_size ${train_batch} "
#     command += "--batch_size_eval {} --init_checkpoint {} ".format(1, bert_model_type)
#     command += "--answer_opt 1 --optimizer adamax "
#     command += "--train_datasets {} --data_dir {} --force\n".format(task, data_dir)
# # run observe_gradient.py to get gradient matrix
# # command += "python observe_gradient.py --task_name {}\n".format(",".join(tasks_list))
# command += "python calculate_corr.py --bert_model_type {} --task_list {}\n".format(bert_model_type,
#                                                                                    ",".join(tasks_list))
# with open(shell_name, "w") as f:
#     f.write(command)
    
    
# =========================> task level <=========================

# shell_name = "tune_task_level_{}.sh".format(bert_model_type)

# command = ""
# command += 'gpu=$1\necho "export CUDA_VISIBLE_DEVICES=$gpu"\nexport CUDA_VISIBLE_DEVICES=${gpu}\ntrain_batch=$2\n'
# for task in tasks_list:
#     # command += "CUDA_VISIBLE_DEVICES=" + str(device) + " "
#     command += "python task_level_matrix_tune.py "
#     # command += "--test_with_label " if is_with_label else ""
#     command += "--batch_size ${train_batch} "
#     command += "--batch_size_eval {} --init_checkpoint {} ".format(eval_batch_size, bert_model_type)
#     command += "--bert_model_type {} --encoder_type {} --num_hidden_layers {} ".format(bert_model_type, encoder_type,
#                                                                                        num_hidden_layers)
#     command += "--answer_opt 1 --optimizer adamax "
#     command += "--train_datasets {} --data_dir {} --force\n".format(task, data_dir)
# # run observe_gradient.py to get gradient matrix
# # command += "python observe_gradient.py --task_name {}\n".format(",".join(tasks_list))
# command += "python calculate_corr.py --bert_model_type {} --task_list {}\n".format(bert_model_type,
#                                                                                    ",".join(tasks_list))
# with open(shell_name, "w") as f:
#     f.write(command)
