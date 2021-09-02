# +
# #!pip install torch==1.6.0

# +
"""
Load and test combinations of trained models
"""

from argparse import Namespace
import os
import argparse
import copy
import joblib
import json

import numpy as np

import torch
from torch.autograd import Variable as V
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms as trn
from torch.utils.data.sampler import WeightedRandomSampler

from azureml.core import Workspace, Datastore, Dataset, Experiment

import smallmodel
import utils
import datasets

print(torch.__version__)

from azureml.core.run import Run
# get the Azure ML run object
run = Run.get_context()


# -

def load_data():
    """Load the test data."""

    args = Namespace(
    classifier='full',
    batch_size=1,
    dummy=True)

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
    print('Original class distributions used (possibly imbalanced)')
    image_dataset = datasets.CleanDataset(args, data_transforms, classes)
    dataloader = torch.utils.data.DataLoader(image_dataset, args.batch_size, 
        num_workers=2, shuffle=True)

    dataset_size = len(image_dataset)

    return dataloader, dataset_size, [*classes]


def load_models(args):
    base_path = os.path.join(args.model_dir,'rc')
    selections = [         
        ['int_ext',     1],   
        ['per_com',     1],
        ['int_main_5',  1],   
        ['bathroom',    1],   
        ['ensuite',     1],   
        ['bedroom',     1],  
        ['kitchen_3',   1], 
        ['living_2',    1], 
        ['aux_4',       1],
        ['ext_main',    1],
        ['ext_front',   1], 
        ['ext_rear',    1] 
    ]

    models={}
    for s in selections:
        if s[1] == 1:
            model = smallmodel.SmallModel(len(utils.clean_classes(s[0])))
            
            pretrained_dict = torch.load(os.path.join(base_path,'{}/model.pth'.format(s[0])),
                                         map_location=torch.device('cpu'))
            model.load_state_dict(pretrained_dict)
            model.to(device)
            for name, param in model.named_parameters():
                    param.requires_grad = False            
                    
            models[s[0]] = model
    
    return models


# +
# Define model order

def predict(classifier, inputs):
    #print('{} on cuda: {}'.format(classifier, next(models[classifier].parameters()).device))
    outputs = models[classifier](inputs)
    probs = F.softmax(outputs,1)
    _, pred = torch.max(probs, 1)
    return pred

def model_logic(models, inputs, metadata):
    try:
        tmp = predict('int_ext', inputs)
        if tmp == 0:
            tmp = predict('per_com',inputs)
            if tmp == 0:
                tmp = predict('int_main_52',inputs)
                if tmp == 0:
                    tmp = predict('bathroom',inputs)
                    if tmp == 0:
                        tmp = predict('ensuite',inputs)
                        if tmp  == 0:
                            output = 1
                        elif tmp ==1 : 
                            output = 0
                        else: 
                            print('Error: {}, {}'.format('ensuite01',tmp))
                            output = 22
                    elif tmp == 1:
                        tmp = predict('ensuite',inputs)
                        if tmp == 0:
                            output = 3
                        elif tmp == 1: 
                            output = 2
                        else: 
                            print('Error: {}, {}'.format('ensuite23',tmp))
                            output = 22
                    elif tmp == 2:
                        tmp = predict('ensuite',inputs)
                        if tmp == 0:
                            output = 5
                        elif tmp == 1:
                            output = 4
                        else: 
                            print('Error: {}, {}'.format('ensuite45',tmp))
                            output = 22
                    else: 
                        print('Error: {}, {}'.format('bathroom',tmp))
                        output = 22
                elif tmp == 1:
                    if metadata == 1.0:
                        output = 6
                    else: 
                        tmp = predict('bedroom',inputs)
                        if tmp == 0:
                            output = 6
                        elif tmp == 1:
                            output = 7
                        else: 
                            print('Error: {}, {}'.format('bedroom',tmp))
                            output = 22
                elif tmp == 2:
                    tmp = predict('kitchen_3',inputs)
                    if tmp == 0:
                        output = 8
                    elif tmp == 1:
                        output = 9
                    elif tmp == 2:
                        output = 10
                    else: 
                        print('Error: {}, {}'.format('kitchen',tmp))
                        output = 22
                elif tmp == 3:
                    tmp = predict('living_2',inputs)
                    if tmp == 0:
                        output = 11
                    elif tmp == 1:
                        output = 12
                    else: 
                        print('Error: {}, {}'.format('living',tmp))
                        output = 22
                elif tmp == 4:
                    tmp = predict('aux_4',inputs)
                    if tmp == 0:
                        output = 13 # output = tmp + 13?
                    elif tmp == 1:
                        output = 14
                    elif tmp == 2:
                        output = 15
                    elif tmp == 3:
                        output = 16
                    else: 
                        print('Error: {}, {}'.format('int_main',tmp))
                        output = 22
                else: 
                    print('Error: {}, {}'.format('int_main',tmp))
                    output = 22
            elif tmp == 1:
                output = 17 ####predict('communal')
            else: 
                print('Error: {}, {}'.format('per_com',tmp))
                output = 22
        elif tmp == 1:
            tmp = predict('ext_main',inputs)
            if tmp == 0:
                tmp = predict('ext_front',inputs)
                if tmp == 0:
                    output = 19
                elif tmp == 1:
                    output = 20
                elif tmp == 2:
                    output = 21
                else: 
                    print('Error: {}, {}'.format('ext_front',tmp))
                    output = 22
            elif tmp == 1:
                tmp = predict('ext_rear',inputs)
                if tmp == 0:
                    output = 22
                elif tmp == 1:
                    output = 23
                elif tmp == 2:
                    output = 24
                else: 
                    print('Error: {}, {}'.format('ext_rear',tmp))
                    output = 22
            else: 
                print('Error: {}, {}'.format('ext_main',tmp))
                output = 22
        else: 
            print('Error: {}, {}'.format('int_ext',tmp))
            output = 22
            
        return torch.Tensor([output]).to(device)
    except:
        print('Error, {}'.format(tmp))
        return torch.Tensor([0]).to(device)


# -


if __name__ == "__main__":
     # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='room-images', help='directory of data')
    parser.add_argument('--model_dir', help='directory of models')
    parser.add_argument('--log_dir', type=str, default='logs', help='log directory')
    parser.add_argument('--arch', type=str, default='resnet50', help='Which model to use (resnet on places365, efficientnet on ImageNet)')
    parser.add_argument('--classifier', type=str, default='full', help='Which classifier to train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dummy', dest='dummy', action='store_true', help='Dummy 5-batch run')
    parser.add_argument('--noise_frac', type=float, default=None,help='fraction of input from previous classifier that is noise/ mislabelled')
    parser.add_argument('--sample_conf', dest='sample_conf', action='store_true', help='save confidence of each sample')
    parser.add_argument('--incl_metadata', dest='incl_metadata', action='store_true', help='Load metadata from codex')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    models = load_models(args)
    dataloader, dataset_size, class_names = load_data(args)

    running_corrects = 0
    int_main_preds = []

    # for confusion matrix
    all_preds, all_labels = [],[]
    # for analysing confidence
    all_ids, all_probs = [],[]

    # Iterate over data.
    for batch_idx, (inputs, labels, metadata, _) in enumerate(dataloader): 
        if args.dummy and batch_idx > 5:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)
        preds = model_logic(models, inputs, metadata)

        # statistics
        corrects = torch.sum(preds == labels.data).float()
        running_corrects += corrects

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        

        #if args.sample_conf == True:
        #    all_ids.extend(img_ids)

        if batch_idx % 10000 == 0:
            print('Batch {}/{}'.format(batch_idx, dataset_size))
            print(int_main_preds)
            
    test_acc = running_corrects.double() / dataset_size    
    utils.metrics(run, test_acc, all_labels, all_preds, class_names)   

    if args.sample_conf == True:
        confs = [[all_ids[i],all_labels[i],all_probs[i]] for i,_ in enumerate(all_ids)]

    print('Test Acc: {:.4f}'.format(test_acc))    
    
    os.makedirs('./outputs', exist_ok=True)
    with open('outputs/args.json', 'w') as fp:
        json.dump(vars(args), fp)

    run.complete()
