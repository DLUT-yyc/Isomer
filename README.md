# Isomer
[ICCV2023] Isomer: Isomerous Transformer for Zero-Shot Video Object Segmentation

## To Do List
- [ ] Release our camera-ready paper. (revising)
- [x] Release our codebase.
- [ ] Release our dataset. (The dataset is a little large and is being uploaded.)
- [x] Release our model checkpoints.
- [x] Release our segmentation maps.

## Introduction

## Installation

The code requires `python>=3.7`, as well as `pytorch>=1.7` and `torchvision>=0.8`. 

## Preparation

Download Pretrained Models, Datasets, Final Checkpoints and Results from [here](https://pan.baidu.com/s/1PJ8JevkmLwaoUVwcScQvCQ) (passwd: iiau).

Please organize the files as follows:

```
dataset/
  TrainSet/
  TestSet/
Isomer/
  checkpoints/
    isomer.pth
  pretrained_model/
    mit_b0.pth
    swin_tiny_patch4_window7_224.pth
  test_results/
    Isomer_Results/
  tools/
  ...
```

## Training

```
# run scripts/train.sh
./scripts/train.sh
```

## Inference

```
# run scripts/infer.sh
./scripts/infer.sh
```

## Evaluation

```
# For ZVOS
python utils/val_zvos.py

# For VSOD
python utils/val_vsod.py
```

## License

The model is licensed under the [Apache 2.0 license](LICENSE).
