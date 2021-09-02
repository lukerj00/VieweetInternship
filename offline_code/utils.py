# +
import os
import argparse
from string import digits
from string import punctuation
import json

import numpy as np
import pandas as pd
from PIL import Image
import random

import torch
from torchvision import transforms as trn
from torch.utils.data.dataset import Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
    
def metrics(run,epoch_acc,all_labels,all_preds,class_names):
        cr = classification_report(all_labels, all_preds, class_names, output_dict=True)     
        fscore_wavg = cr['weighted avg']['f1-score']
        fscore_mavg = cr['macro avg']['f1-score']
        prec_wavg = cr['weighted avg']['precision']
        prec_mavg = cr['macro avg']['precision']
        recall_wavg = cr['weighted avg']['recall']
        recall_mavg = cr['macro avg']['recall']
        
        run.log('test_acc', np.float(epoch_acc))
        run.log('test_wavg_fscore', np.float(fscore_wavg))
        run.log('test_wavg_precision', np.float(prec_wavg))
        run.log('test_wavg_recall', np.float(recall_wavg))

        run.log('test_mavg_fscore', np.float(fscore_mavg))
        run.log('test_mavg_precision', np.float(prec_mavg))
        run.log('test_mavg_recall', np.float(recall_mavg))  

        # Compute and log confusion matrix
        cmtx = confusion_matrix(all_labels, all_preds, [*class_names])        
        cmtx = {
            "schema_type": "confusion_matrix",
            "data": {"class_labels": [*class_names],
                 "matrix": [[int(y) for y in x] for x in cmtx]}
        }
        run.log_confusion_matrix('Confusion matrix',cmtx)
        
        with open('outputs/cr.json', 'w') as fp:
            json.dump(cr, fp)
        with open('outputs/cm.json', 'w') as fp:
            json.dump(cmtx, fp)
        
def label_check(label, classifier):
        label = label.translate(str.maketrans('', '', digits))
        label = label.translate(str.maketrans('', '', punctuation))
        label = label.lower().replace(' ','')
 #       for key in classes:     
 #           if label != '' and (label in classes[key]):# or 'communal' in label):
 #              return label 

def clean_classes(classifier):
    # Simple selection of possible class labels
    # Classifiers at top of tree must contain all classes of its children, even if not obvious 
    if classifier == 'int_ext':
        classes = {
            0 : ['bath','bathroom','wc','ensuite','toilet','showerroom','bed','bedroom','masterbedroom','kitchen', 'kitchenette','kitchenarea','kitchendiner','diner','breakfastroom','bfastroom','diningroom','sittingroom','sitting','lounge','receptionroom','reception','living','livingroom','study','office','entrancehall','porch','entrance','hall','hallway','upstairshallway','corridor','landing','topofstairs','stairs','store','cupboard','cellar','utility','lift','liftarea','communalcorridor','communalhall','communallanding','lobby', 'communal'],       
            1 : ['garden','backgarden','frontgarden','reargarden','patio','rearpatio','frontaspect','parkingarea','terrace','frontelevation','balcony','frontexternal','backexternal','external','exterior','streetview','propertyfront','frontdoor','frontview','backview','outside']
            }

    elif classifier == 'per_com':
        # add 'communal' to label filter
        classes = {
            0 : ['bath','bathroom','wc','ensuite','toilet','showerroom','bed','bedroom','masterbedroom','kitchen', 'kitchenette','kitchenarea','kitchendiner','diner','breakfastroom','bfastroom','diningroom','sittingroom','sitting','lounge','receptionroom','reception','living','livingroom','study','office','entrancehall','porch','entrance','hall','hallway','upstairshallway','corridor','landing','topofstairs','stairs','store','cupboard','cellar','utility'],
            1 : ['lift','liftarea','communalcorridor','communalhall','communallanding','lobby', 'communal'] # or 'communal' in label
        }

    elif classifier == 'int_main_5':
        classes = {
            0 : ['bath','bathroom','wc','ensuite','toilet','showerroom'],
            1 : ['bed','bedroom','masterbedroom'],
            2 : ['kitchen', 'kitchenette','kitchenarea','kitchendiner','diner','breakfastroom','bfastroom','diningroom'],
            3 : ['sittingroom','sitting','lounge','receptionroom','reception','living','livingroom','study','office'],
            4 : ['entrancehall','porch','entrance','hall','hallway','upstairshallway','corridor','landing','topofstairs','stairs','store','cupboard','cellar','utility']
        }
        
    elif classifier == 'int_main_52':
        classes = {
            0 : ['bath','bathroom','wc','ensuite','toilet','showerroom'],
            1 : ['bed','bedroom','masterbedroom'],
            2 : ['kitchen', 'kitchenette','kitchenarea','kitchendiner','diner','breakfastroom','bfastroom','diningroom'],
            3 : ['sittingroom','sitting','lounge','receptionroom','reception','living','livingroom','study','office'],
            4 : ['entrancehall','porch','entrance','hall','hallway','upstairshallway','corridor','landing','topofstairs','stairs','store','cupboard','cellar','utility']
    }

    elif classifier == 'bathroom':
        classes = {
            0 : ['bathroom','bath','ensuite','ensuitebathroom'],
            1 : ['wc','toilet'],
            2 : ['showerroom','ensuiteshower','ensuiteshowerroom']
        }
    
    elif classifier == 'ensuite':
        classes = {
            0 : ['not ensuite'],
            1 : ['ensuite']
        }

    elif classifier == 'bedroom':
        classes = {
            0 : ['masterbedroom','bedroom1','bed1'],
            1 : ['bedroom','bed','bedroom2','bed2','bedroom3','bed3','bedroom4','bed4','bedroom5','bed5']
        }
        
    elif classifier == 'kitchen':
        classes = {
            0 : ['kitchen','kitchenette'],
            1 : ['kitchenarea'],
            2 : ['kitchendiner','diner'],
            3 : ['breakfastroom','bfastroom'],
            4 : ['diningroom']
        }
        
    elif classifier == 'kitchen_2':
        classes = {
            0 : ['kitchen','kitchenette','kitchenarea'],
            1 : ['kitchendiner','diner','breakfastroom','bfastroom']
        }
        
    elif classifier == 'kitchen_3':
        classes = {
            0 : ['kitchen','kitchenette','kitchenarea'],
            1 : ['kitchendiner','diner','breakfastroom','bfastroom'],
            2 : ['diningroom']
        }
        
    elif classifier == 'living':
        classes = {
            0 : ['sittingroom','sitting'],
            1 : ['lounge'],
            2 : ['reception','receptionroom'],
            3 : ['living','livingroom'],
            4 : ['study','office']
        }
        
    elif classifier == 'living_2':
        classes = {
            0 : ['sittingroom','sitting','lounge','reception','receptionroom','living','livingroom'],
            1 : ['study','office']
        }
        
    elif classifier == 'reception':
        classes = {
            0 : ['diningroom'],
            1 : ['sittingroom','sitting','lounge','reception','receptionroom','living','livingroom','study','office']
        }
        
    elif classifier == 'aux':
        classes = {
            0 : ['entrancehall','entrance','porch'],
            1 : ['hall'],
            2 : ['hallway','upstairshallway'],
            3 : ['corridor'],
            4 : ['landing','topofstairs','stairs'],
            5 : ['store','cupboard','cellar','utility']
        }      
        
    elif classifier == 'aux_4':
        classes = {
            0 : ['entrancehall','entrance','porch'],
            1 : ['hall','hallway','upstairshallway','corridor'],
            2 : ['landing','topofstairs','stairs'],
            3 : ['store','cupboard','cellar','utility']
        }  
        
    elif classifier == 'ext_main':
        classes = {
            0 : ['front','frontelevation','frontaspect','frontexternal','streetview',
                 'propertyfront','exterior','external','frontview','frontgarden',
                 'parking','parkingarea','stoop','frontdoor'],
            1 : ['rear','backgarden','garden','backgarden','reargarden','backexternal',
                 'terrace','balcony','patio','decking']
        }
        
    elif classifier == 'ext_front':
        classes = {
            0 : ['front','frontelevation','frontaspect','frontexternal','streetview',
                 'propertyfront','exterior','external','frontview','frontgarden'],
            1 : ['parking','parkingarea'],
            2 : ['stoop','frontdoor']
        }
            
    elif classifier == 'ext_rear':
        classes = {
            0 : ['rear','backgarden','garden','backgarden','reargarden','backexternal'],
            1 : ['patio','decking','terrace'],
            2 : ['balcony'] 
        }       
           
    elif classifier == 'full':
        classes = {
            0 : 'Ensuite bathroom',
            1 : 'Non-ensuite bathroom',
            2 : 'Ensuite WC',
            3 : 'Non-ensuite WC',
            4 : 'Ensuite shower-room',
            5 : 'Non-ensuite shower-room',
            6 : 'Master bedroom',
            7 : 'Bedroom',
            8 : 'Kitchen/ Kitchen Area',
            9 : 'Kitchen-Diner/ Breakfast room',
            10 :'Dining Room',
            11 :'Sitting Room/ Lounge/ Living Room',
            12 : 'Study',
            13 : 'Private Entrance Hall/ Porch',
            14 : 'Private corridor/ Hallway',
            15 : 'Landing',
            16 : 'Store/ Utility',
            17 : 'Communal area',
            18 : 'Front Elevation/ Front Garden',
            19 : 'Parking Area',
            20 : 'Front Door',
            21 : 'Rear Elevation/ Back Garden',
            22 : 'Terrace/ Patio',
            23 : 'Balcony'         
        }

    return classes
