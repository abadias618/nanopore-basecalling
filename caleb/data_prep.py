#importing the libraries
import torch
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch import nn
from torch import optim
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import shutil

df = pd.read_csv('archive/HAM10000_metadata.csv')

df = df[['image_id','dx']]
types = set(df['dx'])
dataDir = 'data/'
trainDir = dataDir + 'train/'
testDir = dataDir + 'test/'
typeDirs = [f'{x}/' for x in sorted(types)]
if not os.path.exists(dataDir): os.mkdir(dataDir)
if not os.path.exists(trainDir): os.mkdir(trainDir)
if not os.path.exists(testDir): os.mkdir(testDir)
for typeDir in typeDirs:
    if not os.path.exists(trainDir + typeDir): os.mkdir(trainDir + typeDir)
    if not os.path.exists(testDir + typeDir): os.mkdir(testDir + typeDir)

allDataDir = 'archive/lesion_images/all_images/'
#random.shuffle()
gp = df.groupby('dx')
allImages = {x:list(gp.get_group(x)['image_id']) for x in types}

for type in types:
    random.shuffle(allImages[type])
    tempImgs = allImages[type]
    for y in tempImgs[:(len(tempImgs)//5)*4]:
        img = y + '.jpg'
        source = allDataDir + img
        destination = trainDir + type + '/' + img
        dest = shutil.copyfile(source, destination)
    for y in tempImgs[(len(tempImgs)//5)*4:]:
        img = y + '.jpg'
        source = allDataDir + img
        destination = testDir + type + '/' + img
        dest = shutil.copyfile(source, destination)



print(df.head())
