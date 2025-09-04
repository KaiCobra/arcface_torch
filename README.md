# Distributed Arcface Training in Pytorch

The "arcface_torch" repository is the official implementation of the ArcFace algorithm. It supports distributed and sparse training with multiple distributed training examples, including several memory-saving techniques such as mixed precision training and gradient checkpointing. It also supports training for ViT models and datasets including WebFace42M and Glint360K, two of the largest open-source datasets. Additionally, the repository comes with a built-in tool for converting to ONNX format, making it easy to submit to MFR evaluation systems.

## Requirements

To avail the latest features of PyTorch, we have upgraded to version 1.12.0.

- Install [PyTorch](https://pytorch.org/get-started/previous-versions/) (torch>=1.12.0).
- (Optional) Install [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/), our doc for [install_dali.md](docs/install_dali.md).
- `pip install -r requirement.txt`.
  
## How to Training

To train a model, execute the `train_v2.py` script with the path to the configuration files. The sample commands provided below demonstrate the process of conducting distributed training.

### 1. Prepare dataset:
- python -m im2rec --list --recursive train <path_to_your_dataset>
- python -m im2rec train.lst <path_to_your_dataset>

### 3. Modify the configuration file:
Modify the configuration file to set the path to your dataset and other hyperparameters as needed.

```text
configs/dcface.py
  config.network = "LResNet50E_IR" # IR50 model
  config.resume = False # Whether to resume from a checkpoint

  config.rec = "path_to_rec_folder" # Path to the folder containing the .rec, .idx, .lst and .bin files 
          # Training Data (MXNet RecordIO format):
              train.rec – binary file storing training images
              train.idx – index file for image positions
              train.lst – image list file containing paths and labels
          # Evaluation Benchmarks (.bin files):
              agedb_30.bin – AgeDB-30 dataset
              calfw.bin – CALFW dataset
              cfp_fp.bin – CFP-FP dataset
              cplfw.bin – CPLFW dataset
              lfw.bin – LFW dataset
  config.num_classes = 10000 # ID number of your dataset
  config.num_image = 5000000 # Image number of your dataset
  config.val_targets = ['lfw', 'cfp_fp', "agedb_30", 'cplfw', 'calfw'] # Evaluation datasets
```
### 2. To run on one GPU:

```shell
python train_v2.py configs/dcface.py
```

Note:   
It is not recommended to use a single GPU for training, as this may result in longer training times and suboptimal performance. For best results, we suggest using multiple GPUs or a GPU cluster.  


### 2. To run on a machine with 8 GPUs:

```shell
torchrun --nproc_per_node=8 train_v2.py configs/dcface.py
```

## Download Datasets or Prepare Datasets  
- [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57) (87k IDs, 5.8M images)
- [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) (93k IDs, 5.2M images)
- [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) (360k IDs, 17.1M images)
- [WebFace42M](docs/prepare_webface42m.md) (2M IDs, 42.5M images)
- [Your Dataset, Click Here!](docs/prepare_custom_dataset.md)

## Model Zoo

- The models are available for non-commercial research purposes only.  
- All models can be found in here.  
- [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g): e8pw  
- [OneDrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)