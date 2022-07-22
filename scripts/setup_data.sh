DATA_DIR=$(pwd)/data

mkdir setup_data
cd setup_data

# download glue
git clone https://github.com/nyu-mll/jiant-v1-legacy.git
cd jiant-v1-legacy
python scripts/download_glue_data.py --data_dir $DATA_DIR --tasks all

cd ../../

python experiments/glue/glue_prepro.py

rm -rf setup_data