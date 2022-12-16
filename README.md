# Skin Cancer Classification using MNIST: HAM10000
Project for **CS6553** Deep Learning
Data:[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download)

# Project Member Team Names
Abdias Baldiviezoaguilar, Sam Himes, Caleb Cranney

**Initially our project was related to nanopore-basecalling, after many hours of work and debug, we reconsidered and shifted our focus to a another area: Skin Cancer Detection/Classification.**


# Requirements
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)

# Table Of Contents
-  [Project Background](#project-background)
-  [Data Description](#data-description)
-  [Model Experimentation Description](#model-experimentation-description)
-  [Expected Data Structure after downloading from source](#expected-data-structure-after-downloading-from-source)
-  [Results](#results)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Background: Project Hard Pivot](#background-project-hard-pivot)
-  [Acknowledgments](#acknowledgments)

# Project Background
Skin Cancer is an extremely prevalent form of cancer. In the US, about 9,500 people in the US are diagnosed with skin cancer every day. When detected early, patients with skin cancer have an extremely high survival rate. Steps should be taken to improve the accessibility of early detection. With an accurate deep learning model, patients could take pictures of their own skin abnormalities and detect cancer early. We aim to modify a deep learning model to more accurately classify cancer.

# Data Description
This dataset contains 10015 dermatoscopic images of pigmented lesions for patients in 7 diagnostic categories. For more than half of the subjects, the diagnosis was confirmed through histopathology and for the rest of the patience through follow-up examinations, expert consensus, or by in-vivo confocal microscopy. More information about the dataset and the diagnosis categories, features and patience conditions besides the links to download the dataset can be found on either [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) or on [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/home).

The categories include; Actinic keratoses and intraepithelial carcinoma / Bowen's disease (AKIEC), basal cell carcinoma (BCC), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, BKL), dermatofibroma (DF), melanoma (MEL), melanocytic nevi (NV) vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, VASC). Of these categories, 3 are cancerous (BCC, AKIEC, MEL) and 4 are non-cancerous (BKL, DF, VASC, NV).

![alt text](https://github.com/abadias618/nanopore-basecalling/blob/main/images/skin_ex.png)

# Model Experimentation Description
We approached our model generation from 2 different model architectures with 2 different data preparations, resulting in 4 different model runs.

## Model Architectures

1. densnet121 - We investigated how other programmers had addressed this problem in the past. We found [the following notebook](https://www.kaggle.com/code/mathewkouch/ham10000-skin-lesion-classifier-82-pytorch), which details using the [densenet121](https://pytorch.org/vision/master/models/generated/torchvision.models.densenet121.html) model for lesion type identification using this dataset. The model came with pretrained data, though the classifier step was altered to be trained specific to the dataset.

2. densenet121/resnet18 hybrid - We found [the following paper](https://ieeexplore.ieee.org/document/8683352) which outlines the hybridization of pre-trained models for identifying lesion types. Following this pattern, we combined the [densenet121](https://pytorch.org/vision/master/models/generated/torchvision.models.densenet121.html) and [resnet18](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18) pretrained models using the same classifier modification step as model architecture 1.

## Data Preparation

1. Categorization - the data is originally divided into the 7 categories listed [above](#data-description). We trained the model architectures to categorize a photo by skin lesion type.

2. Binary - we determined which of the 7 categories were considered malignant and which were considered benign (sometimes labelled in the code as "cancer" and "nonCancer," respectively) and set up the models for binary classification between these two classes instead of all seven.

## Model Notebook Links

1. [densnet121, Categorization](https://github.com/abadias618/nanopore-basecalling/blob/main/notebooks/Densenet_multiclassifier.ipynb)
2. [densenet121/resnet18 hybrid, Categorization](https://github.com/abadias618/nanopore-basecalling/blob/main/notebooks/Mix_multiclassifier.ipynb)
3. [densnet121, Binary](https://github.com/abadias618/nanopore-basecalling/blob/main/notebooks/Densenet_binary_classifier.ipynb)
4. [densnet121/resnet18 hybrid, Binary](https://github.com/abadias618/nanopore-basecalling/blob/main/notebooks/Mix_binary_classifier.ipynb)

# Expected Data Structure after downloading from source
```
./archive
├── HAM10000_metadata.csv
├── hmnist_28_28_L.csv
├── hmnist_28_28_RGB.csv
├── hmnist_28_28_RGB_binary.csv
├── hmnist_8_8_L.csv
├── hmnist_8_8_RGB.csv
└── lesion_images
    └── all_images [10015 entries]
```

# Results

These were the final accuracies for each of the runs.

DenseNet, multi class: 0.8248
Mix, multi class: 0.8208
DenseNet, binary: 0.8867
Mix, binary: 0.8877

It appears that changing the problem to a binary classification improves the accuracy of the classifier. This is encouraging, as the primary intent of our program is to reduce the likelihood for mistaking a non-cancerous lesion for a cancerous lesion generally. While identifying the specific subtype and implied danger levels is important, this may be best confirmed by medical professional who can recommend proper medical treatment. 

Furthermore, it appears that mixing ResNest and DenseNet pre-trained classifiers did not appear to help the accuracy for either binary or multi class.

## Multi Class DenseNet (Original)
![alt text](https://github.com/abadias618/nanopore-basecalling/blob/main/images/densenet_multi_accuracy.png)
![alt text](https://github.com/abadias618/nanopore-basecalling/blob/main/images/densenet_multi_confussion_matrix.png)

## Multi Class Mix
![alt text](https://github.com/abadias618/nanopore-basecalling/blob/main/images/mix_multi_accuracy.png)
![alt text](https://github.com/abadias618/nanopore-basecalling/blob/main/images/mix_multi_confusion_matrix.png)

## Binary DenseNet 
![alt text](https://github.com/abadias618/nanopore-basecalling/blob/main/images/densenet_binary_accuracy.png)
![alt text](https://github.com/abadias618/nanopore-basecalling/blob/main/images/densenet_binary_confussion_matrix.png)

## Binary Mix
![alt text](https://github.com/abadias618/nanopore-basecalling/blob/main/images/mix_binary_accuracy.png)
![alt text](https://github.com/abadias618/nanopore-basecalling/blob/main/images/mix_binary_confusion_matrix.png)






# Future Work
We were inspired by Mahbod et al. to used pertained models to used multiple pre-trained models to generate feature maps that would be fed into our classifier. However, in there work, they included an additional step of training an SVM classifier at the end of each pre-trained model. We, on the other hand, simply concatenated our feature maps and fed them into our fully connected classifier. It would have been interesting to train classifiers for each pre-trained model, and then combine them to get a prediction.

Additionally, it would be interesting to test a variety of combinations of pre-trained networks. Due to time constraints we could only combine DenseNet and ResNet, but it would be interesting to see if combinations of other pre-trained networks faired better or worse.

# Contributing
Any kind of enhancement or contribution is welcomed.

# Background: Project Hard Pivot
We originally proposed as our course project to generate a deep learning model for basecalling raw nanopore sequencing data. We investigated and attempted to run the [SACall](https://github.com/huangnengCSU/SACall-basecaller) repository as a starting point, as they employ transformers, an element of deep learning all project members are interested in. As a backup, we attempted to run [bonito](https://github.com/nanoporetech/bonito), the standard Oxford Nanopore basecaller, for generating models. In both cases, we ran into a number of errors over the course of a week and never succeeded generating a deep learning model using raw data. As such, we requested approval for changing our project to what is listed above.

# Acknowledgments
We thank Mathew Kouch, the author of the [original notebook](https://www.kaggle.com/code/mathewkouch/ham10000-skin-lesion-classifier-82-pytorch) that we started from. We used this notebook as a starting point, and, from there, made our modifications to the input data and pertained models.



