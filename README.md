# GradTS: A Gradient-Based Automatic Auxiliary Task Selection Method Based on Transformer Networks.

The source code for [GradTS](https://aclanthology.org/2021.emnlp-main.455/) (appeared in EMNLP 2021), where most of the resources in this repository are based on [mt-dnn](https://github.com/namisan/mt-dnn).

The main system requirements:

- python >= 3.6
- pytorch >= 1.5
- transformers
- numpy

## Preperations

### 1. Environments

We recommend the readers to use the same environment as us.

```bash
conda create -n gradts python==3.6.5
conda activate gradts
sh setup_env.sh
```

### 2. Datases & Processing

Run the following script to download and process datasets for 8 glue tasks. 

```shell
sh setup_data.sh
```

Then, run this script to tokenize all the datasets.

```shell
sh setup_tok.sh
```

Besides the glue tasks, you can use any other task (e.g., pos, ner). But make sure that you:

- Add task definition under the `experiments/`, it should be a `.yml` file that is similar to `glue_task_def.yml`.
- Process your datasets to   `.json` files, which is similar to `data/canonical_data/bert-base-cased/cola_train.json`.

## Quick Start

Various encoders are used in our experiments (e.g., bert, roberta). Here, we set `bert-base-cased` as an example.

### 1. Correlation

To fine-tune PLM and get head importance matrices, run the following commands. Herein, the first and second parameters (i.e., 0, 4) represent the **GPU number** and **training batch size**, respectively. Change them according to your infrastructure.

```shell
sh ./scripts/tune_task_level_bert-base-cased.sh 0 4
sh ./scripts/tune_ins_level_bert-base-cased.sh 0 4
```

Then, use this script to get the task- and instance-level correlation.

```shell
python calculate_corr.py --bert_model_type bert-base-cased --task_list mnli,rte,qqp,qnli,mrpc,sst,cola,wnli
```

### 2. GradTS-trial

For task-level trial-based selection (i.e., GradTS-trial), utilize these two commands below. 

```shell
python ./scripts_gen/ex_generator_GradTS-trial.py -T mnli,rte,qqp,qnli,mrpc,sst,cola,wnli -B bert-base-cased
sh GradTS-trial_8_bert-base-cased.sh 4 0 7  # training batch size, GPU, epoch
```

Then, run the following script.

```shell
python read_trial_results.py -T mnli,rte,qqp,qnli,mrpc,sst,cola,wnli -B bert-base-cased
```

You can find the selection results and the best performance in `checkpoints/8_bert-base-cased/GradTS-trial`. 

### 3. GradTS-fg

For instance-level selection (i.e., GradTS-fg), run these scripts.

```shell
python ./scripts_gen/ex_generator_GradTS-fg.py -T mnli,rte,qqp,qnli,mrpc,sst,cola,wnli -B bert-base-cased --thres 0.62
sh GradTS-fg_8_bert-base-cased_0.62.sh 4 0 7  # training batch size, GPU, epoch
```

You can find the best performance in `checkpoints/8_bert-base-cased/GradTS-fg`.

## Reference

If you use our code for any publications, please cite us:

```bash
@inproceedings{ma2021gradts,
  title={GradTS: A Gradient-Based Automatic Auxiliary Task Selection Method Based on Transformer Networks},
  author={Ma, Weicheng and Lou, Renze and Zhang, Kai and Wang, Lili and Vosoughi, Soroush},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={5621--5632},
  year={2021}
}
```

Feel free to contact us at `weicheng.ma.gr@dartmouth.edu` or `marionojump0722@gmail.com` if there is any confusion.