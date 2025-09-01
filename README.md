<!-- 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The official implementation of "VAM: A Self-Supervised Vision-Action Model for Visual Navigation Pre-Training".

## Installation
Main libraries:
* [PyTorch](https://www.pytorch.org/): as the main ML framework
* [Comet.ml](https://www.comet.ml): tracking code, logging experiments
* [OmegaConf](https://omegaconf.readthedocs.io/en/latest/): for managing configuration files

First create a virtual env for the project. 
```bash
conda env create -f env.yaml
conda activate vanp-prev
```

Then install the latest version of PyTorch from the [official site](https://www.pytorch.org/). Finally, run the following:
```bash
pip install -r requirements.txt
```
To set up Comet.Ml follow the [official documentations](https://www.comet.ml/docs/).

## Dataset
To download and the dataset please follow [this](docs/data_parser.md) guide.

## Training
To train the Barlow Twins (edit [config](VAM/conf/pretext_config.yaml) first):
```bash
./run.sh train_vanp
```
To train the end-to-end model (edit [config](VAM/conf/config.yaml) first):
```bash
./run.sh train
```

## Acknowledgements
Thanks for [GNM](https://github.com/PrieureDeSion/drive-any-robot) paper repo for making their code public. -->




<!-- ```
conda create -n pose2nav python=3.8
conda activate pose2nav
conda install pytorch torchvision=0.13.0 torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip3 install sdist gdown pyyaml netifaces openpifpaf==0.12.10

pip install mmcv mmdet mmpose

pip install rospkg

pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.2"

``` -->

<!--

conda create -n parser python=3.9 -y
conda install pytorch torchvision=0.13.0 torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"

-->

<!-- ## Launch
1. Launch skeletal keypthon detection
    c
    conda activate pose2nav
    export PYTHONPATH=$(pwd)/pose2nav/skeleton/_monoloco:$PYTHONPATH
    export PYTHONPATH=$(pwd)/pose2nav/skeleton/_mmpose:$PYTHONPATH
    cd scripts/
    python test_pose_estimation.py
    ```
2. Play the ROS bag -->

```
conda env create -f environment.yaml
conda activate learner

export PYTHONPATH=$(pwd)/pose2nav:$PYTHONPATH
export PYTHONPATH=$(pwd)/pose2nav/skeleton/_monoloco:$PYTHONPATH
export PYTHONPATH=$(pwd)/pose2nav/skeleton/_mmpose:$PYTHONPATH

python pose2nav/train_pretext.py --cfg pose2nav/config/train.yaml
python pose2nav/train_downstream.py --cfg pose2nav/config/train.yaml

```