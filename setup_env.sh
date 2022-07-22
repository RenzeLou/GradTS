# setup most env
pip install -r requirements.txt

# setup pytorch, we use `pytorch 1.7.0` with `CUDA 11.0`` as default.
mkdir setup_env
cd setup_env
wget https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp36-cp36m-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu110/torchvision-0.8.0-cp36-cp36m-linux_x86_64.whl
pip install torch-1.7.0+cu110-cp36-cp36m-linux_x86_64.whl
pip install torchvision-0.8.0-cp36-cp36m-linux_x86_64.whl

# setup apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ../../
rm -rf setup_env