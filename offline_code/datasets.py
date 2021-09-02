import os
import argparse
from string import digits
from string import punctuation

import numpy as np
import pandas as pd
from PIL import Image
import random

import torch
from torchvision import transforms as trn
from torch.utils.data.dataset import Dataset

class CleanDataset(Dataset):
    def __init__(self, args, transforms, classes):
        """
        Custom dataset, loads images from filepath given in codex,
        Splits reproducibly into train and validation sets
        
        Inputs:
         - datadir: path to datastore that contains codex and images
         - transforms: dict of torch.transforms.compose objects
        """
        self.args = args
        self.data_dir = args.data_dir
        # Transforms
        self.transforms = transforms
        self.classes = classes
        
        # Read the csv file of cleaned codex
        c_path = os.path.normpath("C:/Users/lukej/Documents/VieweetInternship/DataForLuke-20210816T212605Z-001/DataForLuke/full_wmeta.csv"),
        #self.data_dir+"..\\full_wmeta.csv"
        #c_path = self.data_dir+'/codices/clean_codices/{}.csv'.format(args.classifier)
        #if args.incl_metadata: 
        #c_path = c_path[:-4]+'_wmeta'+c_path[-4:]
        full_codex = pd.read_csv(c_path)
        self.codex = full_codex[full_codex['set']=='test']
 
        # Calculate length of full set
        self.full_data_len = len(self.codex.index)
        
        np.random.seed(1)
        indices = list(range(self.full_data_len))
        np.random.shuffle(indices)
        indices = indices[1000:] # replace 1000 with desired dataset size
        
        # Get image paths
        self.image_arr = np.asarray(self.codex.iloc[indices]['panorama_file'])
        # Get labels
        self.label_arr = np.asarray(self.codex.iloc[indices]['panorama_name'])
        
        if self.args.incl_metadata:
            # Get metadata
            self.meta_arr = np.array(self.codex.iloc[indices]['property_beds'])

    def __getitem__(self, index):
        # Get image path 
        img_path = os.path.join(self.data_dir, 'images', self.image_arr[index])
        # Open and transform image
        img_as_img = Image.open(img_path)        
        img = self.transforms(img_as_img)     
        
        # Get label(class) of the image based on the cropped pandas column
        img_label = self.label_arr[index]
        
        # **Refactor   return appropriate variables depending on args
        #if self.args.sample_conf and not self.args.incl_metadata:
        #    return (img, img_label, '', self.image_arr[index]) 
        if self.args.incl_metadata and not self.args.sample_conf:
            return (img, img_label, self.meta_arr[index], '')
        # elif self.args.sample_conf and self.args.incl_metadata:
        #     return (img, img_label, self.meta_arr[index])#, self.image_arr[index]) 
        # else:            
        #     return (img, img_label, '', '')  

    def __len__(self):
        return len(self.image_arr)
    
    def make_weights(self):   
        # for balanced class sampling
        counts = np.array([np.count_nonzero(self.label_arr==k) for k in [*self.classes]])
        class_weights = 1./counts
        sample_weights = [class_weights[c] for c in self.label_arr]
        print(class_weights)
        return sample_weights  


class TestDataset(Dataset):
    def __init__(self, args, transforms, classes):
        """
        Custom dataset, loads images from filepath given in codex,
        Splits reproducibly into train and validation sets
        
        Inputs:
         - datadir: path to datastore that contains codex and images
         - transforms: dict of torch.transforms.compose objects
        """
        self.args = args
        self.data_dir = args.data_dir
        # Transforms
        self.transforms = transforms
        self.classes = classes
        
        # Read the csv file of cleaned codex
        #csv_str = r'..\\full_wmeta.csv'
        c_path = os.path.normpath("C:/Users/lukej/Documents/VieweetInternship/DataForLuke-20210816T212605Z-001/DataForLuke/full_wmeta.csv")
        #self.data_dir+csv_str
        #/codices/clean_codices/full_wmeta.csv'#.format(args.classifier)
        #if args.incl_metadata: c_path = c_path[:-4]+'_wmeta'+c_path[-4:]
            
        full_codex = pd.read_csv(c_path)
        self.codex = full_codex[full_codex['set']=='test']
        self.codex = self.codex[self.codex['panorama_name'].isin(range(0,17))]
 
        # Calculate length of full set
        self.full_data_len = len(self.codex.index)
        print(self.full_data_len)
        
        np.random.seed(1)
        indices = list(range(self.full_data_len))
        np.random.shuffle(indices)
        
        # Get image paths
        self.image_arr = np.asarray(self.codex.iloc[indices]['panorama_file'])
        # Get labels
        self.label_arr = np.asarray(self.codex.iloc[indices]['panorama_name'])
        
        if self.args.incl_metadata:
            # Get metadata
            self.meta_arr = np.array(self.codex.iloc[indices]['property_beds'])

    def __getitem__(self, index):
        # Get image path 
        img_path = os.path.join(self.data_dir, 'images', self.image_arr[index])
        # Open and transform image
        img_as_img = Image.open(img_path)        
        img = self.transforms(img_as_img)     
        
        # Get label(class) of the image based on the cropped pandas column
        img_label = self.label_arr[index]
        if img_label in [0,1,2,3,4,5]:
            img_label = 0
        elif img_label in [6,7]:
            img_label = 1
        elif img_label in [8,9,10]:
            img_label = 2
        elif img_label in [11,12]:
            img_label = 3
        elif img_label in [13,14,15,16]:
            img_label = 4            
        
        if self.args.sample_conf and not self.args.incl_metadata:
            return (img, img_label, '', self.image_arr[index]) 
        elif self.args.incl_metadata and not self.args.sample_conf:
            return (img, img_label, self.meta_arr[index], '')
        elif self.args.sample_conf and self.args.incl_metadata:
            return (img, img_label, self.meta_arr[index])#, self.image_arr[index]) 
        else:            
            return (img, img_label, '', '')  

    def __len__(self):
        return len(self.image_arr)
    
    def make_weights(self):   
        # for balanced class sampling
        counts = np.array([np.count_nonzero(self.label_arr==k) for k in [*self.classes]])
        class_weights = 1./counts
        sample_weights = [class_weights[c] for c in self.label_arr]
        print(class_weights)
        return sample_weights   

class CleanNoiseDataset(Dataset):
    def __init__(self, args, transforms, classes):
        """
        Custom dataset, loads images from filepath given in codex,
        Splits reproducibly into train and validation sets
        
        Inputs:
         - datadir: path to datastore that contains codex and images
         - transforms: dict of torch.transforms.compose objects
         - classes: dict of possible labels (for cleaning)
         - phase: train or val split
        """
        self.datadirs = {'c' : args.c_datadir, 'im' : args.im_datadir}
        # Transforms
        self.transforms = transforms
        self.noise_frac = 1 - args.noise_frac #to work backwards from number of clean samples
        self.classes = classes
        
        # Read the csv file of cleaned codex
        fclean_codex = pd.read_csv(self.datadirs['c']+'/clean_codices/{}.csv'.format(args.classifier))
        self.clean_codex = fclean_codex[fclean_codex['set']=='test']
        self.clean_data_len = len(self.clean_codex.index)
        
        # Read the csv file of full codex
        full_codex = pd.read_csv(self.datadirs['c'] + '/codex.csv')
        full_codex = full_codex[full_codex['set']=='test']
        
        # Remove non-noise samples from full codex
        full_noise = full_codex[~(full_codex.panorama_file.isin(self.clean_codex.panorama_file))].copy()

        # Randomly select desired proportion of samples to be added as noise
        np.random.seed(2)
        noise_data_len = int(self.clean_data_len * ((1/self.noise_frac) - 1))
        indices = list(range(noise_data_len))
        np.random.shuffle(indices)
        noise_codex = full_noise.iloc[indices].copy()
        
        # Set class as other
        noise_codex['panorama_name'] = max([*self.classes])

        self.codex = self.clean_codex.append(noise_codex)
        self.full_data_len = len(self.codex.index)        
        
        # Shuffle
        np.random.seed(1)
        indices = list(range(self.full_data_len))
        np.random.shuffle(indices)

        # First column contains the image paths
        self.image_arr = np.asarray(self.codex.iloc[indices]['panorama_file'])
        # Second column contains the labels
        self.label_arr = np.asarray(self.codex.iloc[indices]['panorama_name'])
        
        del fclean_codex, full_codex, full_noise, noise_codex

    def __getitem__(self, index):
        # Get image path 
        img_path = os.path.join(self.datadirs['im'], self.image_arr[index])
        # Open and transform image
        img_as_img = Image.open(img_path) 
        img = self.transforms(img_as_img)     
        
        # Get label(class) of the image based on the cropped pandas column
        img_label = self.label_arr[index]
        return (img, img_label)  

    def __len__(self):
        return len(self.image_arr) 
