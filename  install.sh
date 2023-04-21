conda create -n mmdtc1 python=3.8 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda activate mmdtc1
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -v -e .
 pip install wandb 