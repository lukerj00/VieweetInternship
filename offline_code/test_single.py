# +
# #!pip install torch==1.6.0

# +
"""
Load and test combinations of trained models
"""

import os
import argparse
import copy
import joblib
import json

import numpy as np
import pandas as pd

from argparse import Namespace 
from pathlib import Path


import torch
from torch.autograd import Variable as V
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms as trn
from torch.utils.data.sampler import WeightedRandomSampler

from azureml.core import Workspace, Datastore, Dataset, Experiment

import smallmodel
import utils
# import custom_trn

print(torch.__version__)

from azureml.core.run import Run
# get the Azure ML run object
run = Run.get_context()


# -

def load_data(args):
    """Load the test data."""

    # Data augmentation and normalization for training
    # Just normalization for validation
    #torch.manual_seed(1)
    data_transforms = trn.Compose([
            trn.Resize((64,128)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Simple selection of possible class labels
    classes = utils.clean_classes(args.classifier)

    # load the data (passed as an input dataset)
    print("Loading Data...")

    if args.noise_frac is not None:
        print('Introducing noise to clean dataset')        
        classes[max([*classes])+1] = ['other']

        image_dataset = utils.CleanNoiseDataset(args, data_transforms, classes)
        dataloader = torch.utils.data.DataLoader(image_dataset, args.batch_size, 
            num_workers=4, shuffle=True)   
    else:
        print('Original class distributions used (possibly imbalanced)')
        image_dataset = utils.TestDataset(args, data_transforms, classes)
        dataloader = torch.utils.data.DataLoader(image_dataset, args.batch_size, 
            num_workers=2, shuffle=True)

    dataset_size = len(image_dataset)

    return dataloader, dataset_size, [*classes]


def load_model(args):
    classifier = args.classifier
    base_path = os.path.join(args.model_dir,'rc')
    
    model = smallmodel.SmallModel(len(utils.clean_classes(classifier)))

    pretrained_dict = torch.load(os.path.join(base_path,'{}\model.pth'.format(classifier)),
                                 map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)
    model.to(device)
    for name, param in model.named_parameters():
            param.requires_grad = False 
            
    return model


if __name__ == "__main__":
     # get command-line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', default='room-images', help='directory of data')
    # parser.add_argument('--model_dir', help='directory of models')
    # parser.add_argument('--log_dir', type=str, default='logs', help='log directory')
    # parser.add_argument('--batch_size', type=int, default=100)
    # parser.add_argument('--arch', type=str, default='resnet50', help='Which model to use (resnet on places365, efficientnet on ImageNet)')
    # parser.add_argument('--classifier', type=str, default='full', help='Which classifier to train')
    # parser.add_argument('--dummy', dest='dummy', action='store_true', help='Dummy 5-batch run')
    # parser.add_argument('--noise_frac', type=float, default=None,help='fraction of input from previous classifier that is noise/ mislabelled')
    # parser.add_argument('--sample_conf', dest='sample_conf', action='store_true', help='save confidence of each sample')
    # parser.add_argument('--incl_metadata', dest='incl_metadata', action='store_true', help='Load metadata from codex')
    # args = parser.parse_args()

    args = Namespace(
    classifier='full',
    batch_size=1,
    dummy=False,
    data_dir = Path("C:/Users/lukej/Documents/VieweetInternship/DataForLuke-20210816T212605Z-001/DataForLuke"),
    model_dir = Path("C:/Users/lukej/Documents/VieweetInternship/DataForLuke-20210816T212605Z-001/DataForLuke/models"),
    c_path = Path("C:/Users/lukej/Documents/VieweetInternship/DataForLuke-20210816T212605Z-001/DataForLuke/full_wmeta.csv"),
    incl_metadata = True,
    sample_conf = False
    #arch = "test3",
    #noise_frac = "test4",
    #log_dir = "test5"
    )
        
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    model = load_model(args)
    dataloader, dataset_size, class_names = load_data(args)

    running_corrects = 0

    # for confusion matrix
    all_preds, all_labels = [],[]
    # for analysing confidence
    all_ids, all_probs = [],[]

    # Iterate over data.
    for batch_idx, (inputs, labels, _, _) in enumerate(dataloader): 
        if args.dummy and batch_idx > 5:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        probs = F.softmax(outputs,1)
        _, preds = torch.max(probs, 1)

        # statistics
        corrects = torch.sum(preds == labels.data).float()
        running_corrects += corrects

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())        

        # if args.sample_conf == True:
        #     all_ids.extend(img_ids)

        if batch_idx % 10000 == 0:
            print('Batch {}/{}'.format(batch_idx, dataset_size// args.batch_size))
            
    test_acc = running_corrects.double() / dataset_size    
    utils.metrics(run, test_acc, all_labels, all_preds, class_names)   

    # if args.sample_conf == True:
    #     confs = [[all_ids[i],all_labels[i],all_probs[i]] for i,_ in enumerate(all_ids)]

    print('Test Acc: {:.4f}'.format(test_acc))    
    
    os.makedirs('./outputs', exist_ok=True)
    # with open('outputs/args.json', 'w') as fp:
    #     json.dump(vars(args), fp)

    run.complete()
