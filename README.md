# Skin Cancer Classification using MNIST: HAM10000
Project for **CS6553** Deep Learning
Data:[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download)


**Initially our project was related to nanopore-basecalling, after many hours of work and debug, we reconsidered and shifted our focus to a another area: Skin Cancer Detection/Classification.**


# Requirements
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)

# Table Of Contents
-  [Project Background](#project-background)
-  [Data Description](#data-description)
-  [Model Experimentation Description](#model-experimentation-description)
-  [Summary of Model Generation Instruction](#summary-of-model-generation-instruction)
-  [Expected Data Structure after downloading from source](#expected-data-structure-after-downloading-from-source)
-  [Results](#results)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# Project Background
Pigmented skin lesions can be an indication of skin cancer. However, it is not a guarantee by any means - many such lesions are almost entirely benign. The dataset for this project was comprised of various images of pigmented skin lesions that were categorized by various benign or malignant classes. Our intent is to generate a deep learning model that can accurately predict the type of pigmented skin lesion of any given image.

# Data Description
This dataset contains 10015 dermatoscopic images of pigmented lesions for patients in 7 diagnostic categories. For more than half of the subjects, the diagnosis was confirmed through histopathology and for the rest of the patience through follow-up examinations, expert consensus, or by in-vivo confocal microscopy. More information about the dataset and the diagnosis categories, features and patience conditions besides the links to download the dataset can be found on either [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) or on [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/home).

The categories include; Actinic keratoses and intraepithelial carcinoma / Bowen's disease (AKIEC), basal cell carcinoma (BCC), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, BKL), dermatofibroma (DF), melanoma (MEL), melanocytic nevi (NV) vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, VASC). Of these categories, 3 are cancerous (BCC, AKIEC, MEL) and 4 are non-cancerous (BKL, DF, VASC, NV).

# Model Experimentation Description
We approached our model generation from 2 different model architectures with 2 different data preparations, resulting in 4 different model runs.

## Model Architectures

1. densnet121 - We investigated how other programmers had addressed this problem in the past. We found [the following notebook](https://www.kaggle.com/code/mathewkouch/ham10000-skin-lesion-classifier-82-pytorch), which details using the [densenet121](https://pytorch.org/vision/master/models/generated/torchvision.models.densenet121.html) model for lesion type identification using this dataset. The model came with pretrained data, though the classifier step was altered to be trained specific to the dataset.

2. densenet121/resnet18 hybrid - We found [the following paper](https://ieeexplore.ieee.org/document/8683352) which outlines the hybridization of pre-trained models for identifying lesion types. Following this pattern, we combined the [densenet121](https://pytorch.org/vision/master/models/generated/torchvision.models.densenet121.html) and [resnet18](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18) pretrained models using the same classifier modification step as model architecture 1.

## Data Preparation

1. Categorization - the data is originally divided into the 7 categories listed [above](#data-description). We trained the model architectures to categorize a photo by skin lesion type.

2. Binary - we determined which of the 7 categories were considered malignant and which were considered benign (sometimes labelled in the code as "cancer" and "nonCancer," respectively) and set up the models for binary classification between these two classes instead of all seven.

## Model Experiment Notebook Links

1. [densnet121, Categorization]()
2. [densenet121/resnet18 hybrid, Categorization]()
3. [densnet121, Binary]()
4. [densnet121/resnet18 hybrid, Binary]()

# Summary of Model Generation Instruction 
In a nutshell here's how to use this template, so **for example** assume you want to implement ResNet-18 to train mnist, so you should do the following:
- In `modeling`  folder create a python file named whatever you like, here we named it `example_model.py` . In `modeling/__init__.py` file, you can build a function named `build_model` to call your model.

```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
``` 

   
- In `engine`  folder create a model trainer function and inference function. In trainer function, you need to write the logic of the training process, you can use some third-party library to decrease the repeated stuff.

```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
```

- In `tools`  folder, you create the `train.py` .  In this file, you need to get the instances of the following objects "Model",  "DataLoader”, “Optimizer”, and config
```python
# create instance of the model you want
model = build_model(cfg)

# create your data generator
train_loader = make_data_loader(cfg, is_train=True)
val_loader = make_data_loader(cfg, is_train=False)

# create your model optimizer
optimizer = make_optimizer(cfg, model)
```

- Pass the all these objects to the function `do_train` , and start your training
```python
# here you train your model
do_train(cfg, model, train_loader, val_loader, optimizer, None, F.cross_entropy)
```

**You will find a template file and a simple example in the model and trainer folder that shows you how to try your first model simply.**


# Expected Data Structure after downloading from source
```
./archive
├── HAM10000_metadata.csv
├── hmnist_28_28_L.csv
├── hmnist_28_28_RGB.csv
├── hmnist_8_8_L.csv
├── hmnist_8_8_RGB.csv
└── lesion_images
    └── all_images [10015 entries]
```

# Results

# Future Work

# Contributing
Any kind of enhancement or contribution is welcomed.

# Project Hard Pivot - Background
We originally proposed as our course project to generate a deep learning model for basecalling raw nanopore sequencing data. We investigated and attempted to run the [SACall](https://github.com/huangnengCSU/SACall-basecaller) repository as a starting point, as they employ transformers, an element of deep learning all project members are interested in. As a backup, we attempted to run [bonito](https://github.com/nanoporetech/bonito), the standard Oxford Nanopore basecaller, for generating models. In both cases, we ran into a number of errors over the course of a week and never succeeded generating a deep learning model using raw data. As such, we requested approval for changing our project to what is listed above.

# Acknowledgments




