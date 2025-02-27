# Inference Time Policy Guide

## Setup
1. Clone this repo
```bash
git clone git@github.com:EChoudhury/PolicyGuide.git
cd PolicyGuide
```

2. Create conda environment and install required packages
```bash
# Create conda env
conda create -y -n policyguide python=3.10
conda activate policyguide

# Install requirements 
pip install -r requirements.txt 

# Install Pyhash (deprecation issue with pip)
conda install conda-forge::pyhash
```

3. Install CALVIN environment
```bash
git clone --recursive https://github.com/mees/calvin_env.git
cd calvin_env/tacto
pip install -e .
cd ..
pip install -e .
```

## Download CALVIN Dataset
Download your choice of dataset (D|ABC|ABCD|debug) assuming you installed PolicyGuide in your home folder
```bash
cd ~/PolicyGuide/dataset
sh download_data.sh D | ABC | ABCD | debug
```

## Generate Statistics
Diffusion Policy normalization requires dataset statistics to be generated on the entire body of data. If you wish to use normalization, generate the files:
```bash
python -m itpg.scripts.visualize_dataset --path /path/to/dataset
```
Then update or override the stats_path variable in the default configuration file for the model, found at `/conf/model/default.yaml`, to point to your new stats file. 

Also, update the statistics.yaml files in the dataset directories to point to `itpg.utils.transforms....`

## Training
Launch default training.
```bash
python -m itpg.training datamodule.root_data_dir=/path/to/your/calvin/dataset
```
## Evaluation
Launch default evaluation. While rollout is included in the training policy, here you can evaluate a pretrained policy
```bash 
python -m itpg.evaluation.evaluate_policy --dataset_path /path/to/your/calvin/dataset --train_folder /path/to/saved/model/checkpoint/run/folder --debug
```
