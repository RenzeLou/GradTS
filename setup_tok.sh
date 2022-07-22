# process all datasets using bert tokenizer
python prepro_std.py --model bert-base-uncased --root_dir data/canonical_data --task_def experiments/glue/glue_task_def.yml --do_lower_case
python prepro_std.py --model bert-base-cased --root_dir data/canonical_data --task_def experiments/glue/glue_task_def.yml
python prepro_std.py --model bert-large-uncased --root_dir data/canonical_data --task_def experiments/glue/glue_task_def.yml --do_lower_case
python prepro_std.py --model bert-large-cased --root_dir data/canonical_data --task_def experiments/glue/glue_task_def.yml
python prepro_std.py --model roberta-large --root_dir data/canonical_data --task_def experiments/glue/glue_task_def.yml
python prepro_std.py --model roberta-base --root_dir data/canonical_data --task_def experiments/glue/glue_task_def.yml
