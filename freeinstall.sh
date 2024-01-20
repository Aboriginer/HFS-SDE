conda_env="myenv"
conda create --name $conda_env python==3.10
# Activate the Conda environment
conda activate $conda_env
# Install Python packages using Conda
conda install cuda -c nvidia/label/cuda-11.6.2
conda install cudatoolkit=11.3.1=h2bc3f7f_2
conda install cudnn=8.2.1=cuda11.3_0

conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install scipy==1.9.3
pip install absl-py==1.3.0
pip install tensorboard
conda install h5py
pip install mat73
pip install tensorflow
pip install ml-collections
pip install torchmetrics
pip install opencv-python

# Deactivate the Conda environment
conda deactivate