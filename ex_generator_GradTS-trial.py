''' this script can be used to generate shell to run mtdnn with the task selection result of AutoSem & GradTS '''
import os
import pickle

task_list = ['cola','mrpc','sst','wnli','mnli','qnli','rte','qqp']  # 8 task  ## TODO
# task_list = ['cola', 'mrpc', 'sst', 'wnli', 'rte', 'qnli']  # 6 task
# task_list = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp', 'meld', 'dmeld']  # 10 task
# task_list = ['cola', 'mrpc', 'sst', 'wnli', 'mnli', 'qnli', 'rte', 'qqp', 'stsb', 'pos', 'ner', 'sc']  # 12 task
labeled_data = ['ner', 'pos', 'meld', 'dmeld']  ## TODO: assume sc only has test data
method_dict = {1: "GradTS-thres", 2: "GradTS-trial", 3: "GradTS-fg"}

# set args
method = 2  # 1,2,3
task_threshold = 0.9  ## useless
instance_threshold = 0.62  ## useless
# trial_num = 1
bert_model_type = "bert-large-cased"  # bert-base-cased,bert-base-uncased,roberta-base,bert-large-uncased,roberta-large,bert-base-cased,bert-large-cased  ## TODO
gradient_path = "./gradient_files/{}".format(bert_model_type)
# output_dir = "./checkpoints/{}/{}".format(bert_model_type, method_dict[method])
output_dir = "./checkpoints/{}/{}".format(str(len(task_list)) + "_" + bert_model_type, method_dict[method])
num_layer = 24 if "large" in bert_model_type else 12
data_dir = "./data/canonical_data/{}".format(bert_model_type)
lr = '5e-5'  # 5e-5 ## TODO
best_eval_score = True  # if true,stop the exp according to the best history instead of the last epoch ## TODO
only_wnli = False  ## TODO
only_sc = False  ## TODO
# for main exp, the gradient file path is similar with bert model type
# note epoch num

python_script = 'train_with_corr.py'
encoder_type = 1 if 'roberta' not in bert_model_type else 2

# setting param
template = 'echo "{} running begin!"\n'.format(method_dict[method])
template += 'BATCH_SIZE=$1\ngpu=$2\nEpoch=$3\n'
template += 'ENCODER_TYPE={}\n'.format(encoder_type)
template += "NUM_LAYER={}\n".format(num_layer)
template += "DATA_DIR={}\n".format(data_dir)
template += 'echo "export CUDA_VISIBLE_DEVICES=${gpu}"\necho "epoch num=${Epoch}"\nexport CUDA_VISIBLE_DEVICES=${gpu}\ntstr=$(date +"%FT%H%M")\n'
# template += 'MODEL_ROOT="checkpoints"\nBERT_PATH="{}"\nDATA_DIR="data/canonical_data/{}"\n'.format(model, model)
template += 'answer_opt=1\noptim="adamax"\ngrad_clipping=0\nglobal_grad_clipping=1\nlr="{}"\ngrad_accumulation_step=4\n\n'.format(
    lr)

# training param
if not only_wnli and not only_sc:
    for task in task_list:
        main_task = task
        candidate_lis = [t for t in task_list if t != main_task]  # imply the exp type
        gradient_path = "./gradient_files/{}/{}".format(bert_model_type, main_task)
        corr_path = os.path.join(gradient_path, "gradient_correlation")
        task_corr_file = os.path.join(corr_path, "task_level_corr.pkl")
        with open(task_corr_file, "rb") as f:
            corr = pickle.load(f)
        print("\n==>the corr of the main task:{}".format(corr))
        sorted_candidate = sorted(candidate_lis, key=lambda x: corr[x], reverse=True)
        print("==>for main task {}, the sorted corr is {}".format(main_task, sorted_candidate))
        test_datasets = main_task if "mnli" not in main_task else "mnli_matched,mnli_mismatched"
        candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
        train_datasets = [main_task]  # ",".join(task_list)
        # all_candidate = candidate_set.split(",")
        template += 'echo "main task {} trial exp begin!"\n'.format(main_task)
        for t in sorted_candidate:
            train_datasets.append(t)
            template += 'python {} '.format(python_script)
            template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
                main_task, method, task_threshold, instance_threshold, gradient_path, candidate_set)
            template += "--train_datasets {} --test_datasets {} ".format(",".join(train_datasets), test_datasets)
            template += '--data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers ${NUM_LAYER} ' \
                        '--epochs ${Epoch} --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} ' \
                        '--answer_opt ${answer_opt} --optimizer ${optim} ' \
                        '--grad_clipping ${grad_clipping} ' \
                        '--global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} ' \
                        '--learning_rate ${lr} '
            template += '--test_with_label ' if main_task in labeled_data else ""
            template += "--record_eval_best " if best_eval_score else ""
            template += "--output_dir {} ".format(output_dir)
            template += "--bert_model_type {} ".format(bert_model_type)
            template += "--init_checkpoint {} ".format(bert_model_type)
            template += '\n'

    shell_name = "{}_{}_{}.sh".format(method_dict[method], len(task_list), bert_model_type)
elif only_wnli:
    for task in ["wnli"]:
        main_task = task
        candidate_lis = [t for t in task_list if t != main_task]  # imply the exp type
        gradient_path = "./gradient_files/{}/{}".format(bert_model_type, main_task)
        corr_path = os.path.join(gradient_path, "gradient_correlation")
        task_corr_file = os.path.join(corr_path, "task_level_corr.pkl")
        with open(task_corr_file, "rb") as f:
            corr = pickle.load(f)
        print("\n==>the corr of the main task:{}".format(corr))
        sorted_candidate = sorted(candidate_lis, key=lambda x: corr[x], reverse=True)
        print("==>for main task {}, the sorted corr is {}".format(main_task, sorted_candidate))
        test_datasets = main_task if "mnli" not in main_task else "mnli_matched,mnli_mismatched"
        candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
        train_datasets = [main_task]  # ",".join(task_list)
        # all_candidate = candidate_set.split(",")
        template += 'echo "main task {} trial exp begin!"\n'.format(main_task)
        for t in sorted_candidate:
            train_datasets.append(t)
            template += 'python {} '.format(python_script)
            template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
                main_task, method, task_threshold, instance_threshold, gradient_path, candidate_set)
            template += "--train_datasets {} --test_datasets {} ".format(",".join(train_datasets), test_datasets)
            template += '--data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers ${NUM_LAYER} ' \
                        '--epochs ${Epoch} --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} ' \
                        '--answer_opt ${answer_opt} --optimizer ${optim} ' \
                        '--grad_clipping ${grad_clipping} ' \
                        '--global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} ' \
                        '--learning_rate ${lr} '
            template += '--test_with_label ' if main_task in labeled_data else ""
            template += "--record_eval_best " if best_eval_score else ""
            template += "--output_dir {} ".format(output_dir)
            template += "--bert_model_type {} ".format(bert_model_type)
            template += "--init_checkpoint {} ".format(bert_model_type)
            template += '\n'

    shell_name = "wnli_trial_{}_{}.sh".format(len(task_list), bert_model_type)
elif only_sc:
    for task in ["sc"]:
        main_task = task
        candidate_lis = [t for t in task_list if t != main_task]  # imply the exp type
        gradient_path = "./gradient_files/{}/{}".format(bert_model_type, main_task)
        corr_path = os.path.join(gradient_path, "gradient_correlation")
        task_corr_file = os.path.join(corr_path, "task_level_corr.pkl")
        with open(task_corr_file, "rb") as f:
            corr = pickle.load(f)
        print("\n==>the corr of the main task:{}".format(corr))
        sorted_candidate = sorted(candidate_lis, key=lambda x: corr[x], reverse=True)
        print("==>for main task {}, the sorted corr is {}".format(main_task, sorted_candidate))
        test_datasets = main_task if "mnli" not in main_task else "mnli_matched,mnli_mismatched"
        candidate_set = ",".join([t for t in task_list if t != main_task])  # imply the exp type
        train_datasets = [main_task]  # ",".join(task_list)
        # all_candidate = candidate_set.split(",")
        template += 'echo "main task {} trial exp begin!"\n'.format(main_task)
        for t in sorted_candidate:
            train_datasets.append(t)
            template += 'python {} '.format(python_script)
            template += "--main_task {} --method {} --task_threshold {} --instance_threshold {} --gradient_path {} --candidate_set {} ".format(
                main_task, method, task_threshold, instance_threshold, gradient_path, candidate_set)
            template += "--train_datasets {} --test_datasets {} ".format(",".join(train_datasets), test_datasets)
            template += '--data_dir ${DATA_DIR} --encoder_type ${ENCODER_TYPE} --num_hidden_layers ${NUM_LAYER} ' \
                        '--epochs ${Epoch} --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} ' \
                        '--answer_opt ${answer_opt} --optimizer ${optim} ' \
                        '--grad_clipping ${grad_clipping} ' \
                        '--global_grad_clipping ${global_grad_clipping} --grad_accumulation_step ${grad_accumulation_step} ' \
                        '--learning_rate ${lr} '
            template += '--test_with_label ' if main_task in labeled_data else ""
            template += "--record_eval_best " if best_eval_score else ""
            template += "--output_dir {} ".format(output_dir)
            template += "--bert_model_type {} ".format(bert_model_type)
            template += "--init_checkpoint {} ".format(bert_model_type)
            template += '\n'

    shell_name = "sc_trial_{}_{}.sh".format(len(task_list), bert_model_type)
else:
    raise RuntimeError

print("\nshell name:", shell_name)

with open(shell_name, "w") as f:
    f.write(template)
