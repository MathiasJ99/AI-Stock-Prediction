import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import DataProcessing

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


#data
#data = DataProcessing.GetData()
data = pd.read_excel("MergedDF.xlsx")

#Randomisation
torch.manual_seed(99)

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cuda:0" if torch.cuda.is_available() else "cpu"
#print(device)

