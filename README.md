# Arcface Training in Pytorch

This repository is a simplified fork of the original [arcface_torch](https://github.com/coconutbee/arcface_torch) implementation. It currently includes only the ArcFace model and the `rec/casia` dataset for demonstration and training purposes.

**clone the repository by:**
```bash
!git clone https://github.com/KaiCobra/arcface_torch.git
# Do these command below if you are on colab.
!mv /content/arcface_torch/* /content
!rm -rf /content/arcface_torch
```

## Requirements
```bash
!pip install -r requirement.txt
```  
## How to Training

To train a model, execute the `train.py` script with the path to the configuration files. The sample commands provided below demonstrate the process of conducting distributed training.

### 1. Prepare dataset:
```python
!python -m im2rec --list --recursive train <path_to_your_dataset>
!python -m im2rec train.lst <path_to_your_dataset>
```

### 2. Modify the configuration file:
Modify the configuration file to set the path to your dataset and other hyperparameters as needed.

Inside your .py file `./configs/casia.py`:
```python
    config.network = "LResNet50E_IR"  # IR50 model
    config.resume = False             # Whether to resume from a checkpoint
    config.rec = "./rec/casia"        # Path to the folder containing the .rec, .idx, .lst and .bin files 
    config.num_classes = 10572        # ID number of your dataset
    config.num_image = 490623         # Image number of your dataset
    config.val_targets = ['lfw', 'cfp_fp', "agedb_30", 'cplfw', 'calfw'] # Evaluation datasets
```

### 3. Download dataset by:
```bash
!gdown 1Bhg-Kp101Jgfhg3LWd3tpOA-FnV_Dp4s
!unzip casia.zip -d rec/casia
```
After download the dataset, make sure your folder structure be like:
```shell
rec/casia 
    ├── agedb_30.bin    # AgeDB-30 dataset
    ├── calfw.bin       # CALFW dataset
    ├── cfp_fp.bin      # CFP-FP dataset
    ├── cplfw.bin       # CPLFW dataset
    ├── lfw.bin         # LFW dataset
    ├── train.idx       # index file for image positions
    ├── train.lst       # image list file containing paths and labels
    └── train.rec       # binary file storing training images
```

### 4. To run on one GPU:
Validation while training also shows the validation accuracy; focus on 'Accuracy-Highest' value.
```shell
python train.py configs/arcface.py
```