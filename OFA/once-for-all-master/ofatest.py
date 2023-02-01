import os
from symbol import star_expr
from turtle import Turtle, color
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy
from matplotlib import pyplot as plt

from ofa.model_zoo import ofa_net
from ofa.utils import download_url

from ofa.tutorial import AccuracyPredictor, FLOPsTable, LatencyTable, EvolutionFinder
from ofa.tutorial import evaluate_ofa_subnet, evaluate_ofa_specialized

#random_seed = 1
#random.seed(random_seed)
#np.random.seed(random_seed)
#torch.manual_seed(random_seed)
#print('Successfully imported all packages and configured random seed to %d!'%random_seed)

#Using GPU
cuda_available = torch.cuda.is_available
#if cuda_available:
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True
    #torch.cuda.manual_seed(random_seed)
    #print('GPU is in use.')
#else:
    #print('Using CPU.')

#Environment Configuration has been done. 
#Downloading OFA network
ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained = True)
print('OFA NEtwrok is ready')

#Building Imagenet Dataset and dataloader
#... Downloaded using terminal

#MANUALLY ADD IMAGENET DATASET
#ALREADY DOWNLOADED IT
#ADD IT TO PATH
if cuda_available:
    # print("Please input the path to the ImageNet Dataset. \n")
    # imagenet_data_path = input()
    imagenet_data_path = "/home/ssan/Desktop/OFA/once-for-all-master/imagenet_1k"

#Building dataloader for evaluation
if cuda_available:
    #First, we will build the transforms for the test
    def build_val_tranforms(size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size/0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_data_path,'val'),
            transform=build_val_tranforms(224)
        ),
        batch_size = 250,
        shuffle = True,
        num_workers =16,
        pin_memory = True,
        drop_last = False
    )
    print('The ImageNet Dataloader is ready.')

#if cuda_available:
    #net_id = evaluate_ofa_specialized(imagenet_data_path, data_loader)
    #print('Evaluation of the pre-trained sub-network has been finished: %s' % net_id) 

#Building the accuracy predictor
accuraacy_predictor = AccuracyPredictor(
    pretrained = True,
    device='cuda:0' if cuda_available else 'cpu'
)
print('The accuracy predictor is ready')
print(accuraacy_predictor.model)

flops_lookuptable=FLOPsTable(
    device='cuda:0' if cuda_available else 'cpu',
    batch_size=1)
  

print('FLOPS Lookup table is ready.')

#Hyper Parameters for the evolutionary search process
#latency_constraint = 25 #Suggested range is [15,33]ms
Pop = 100 #Size of population in each generation
noPop = 500 #No.of generations of populations to be searched
ratio = 0.25 #Ratio of netwroks that are used as parets fort next generation

params = {
    'contraint_type': 'flops', #FLOPs constrained search 
    'efficiency_constraint': 600,
    'mutation_prob': 0.1, #Probability of mutation in each evp;utionary search
    'mutation_ratio': 0.5, #Ratio of networks that are generated thru mutation in generation n>=2
    'efficiency_predictor': flops_lookuptable, #We will use a predefined efficiency predictor
    'accuracy_predictor': accuraacy_predictor, #We will use a predefined accuracy predictor
    'population_size': Pop,
    'max_time_budegt': noPop,
    'parent_ratio': ratio,
}

#Building the evolution finder
finder = EvolutionFinder(**params)

#Initiate the search
result_lis = []
for flops in [600,400,350]:
    st=time.time()
    finder.set_efficiency_constraint(flops)
    best_valids, best_info = finder.run_evolution_search()
    ed = time.time()
    result_lis.append(best_info)

plt.figure(figsize=(4,4))
plt.plot([x[-1] for x in result_lis], [x[0] * 100 for x in result_lis], 'x-', marker='*', color='darkred', linewidth=2, markersize=8, label='OFA')
plt.xlabel('FLOPs (M)', size=12)
plt.ylabel('Pred Holdout Top-1 Accuracy (%)', size=12)
plt.legend(['OFA'], loc='lower right')
plt.grid(True)
plt.show()


