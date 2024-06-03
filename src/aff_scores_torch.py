import torch
from torchvision.models import list_models
import torchvision.models as models
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from metrics import sparsity_calc, entropy_batch_calc
from aff_core import rho_aff
from cka_core import linear_CKA
from RandomDataset import RandomDataset
from sklearn.linear_model import LinearRegression

import argparse
import os 
import pickle 
import copy
from pathlib import Path

from utils import set_seeds, init_dict
from tqdm import tqdm

#torch.cuda.empty_cache()

def hook_fn(module, inputs):

    input = inputs[0]

    try:
        sparsity_before = sparsity_calc(input.cpu())
        module.dict['sparsity_before'].append(sparsity_before)
    except:
        sparsity_before = 0
        print("Error in sparsity before")

    # check if the tensor has only positive values
    if torch.all(input >= 0):
        print("The tensor has only positive values")

    # check if the tensor has only negative values
    if torch.all(input <= 0):
        print("The tensor has only negative values")

    if input.ndim == 4:
        input2d = torch.mean(input, dim=(-2,-1))
    elif input.ndim == 3:
        input2d = torch.mean(input, dim=(-1))
    else:
        input2d = copy.deepcopy(input)

    output = module.forward(input)

    if output.ndim == 4:
        output2d = torch.mean(output, dim=(-2,-1))
    elif output.ndim == 3:
        output2d = torch.mean(output, dim=(-1))
    else:
        output2d = copy.deepcopy(output)

    if save_feats:
        module.dict['input2d'].append(input2d.cpu().numpy())
        module.dict['output2d'].append(output2d.cpu().numpy())
    
    try:
        aff_true = rho_aff(input2d.T, output2d.T, device=device)
        module.dict['aff_score'].append(aff_true)
    except:
        aff_true = 0
        print("Error in aff_true")

    try:    
        aff_shrinked = rho_aff(input2d.T, output2d.T, correctCov=True, device=device)
        module.dict['aff_score_corrected'].append(aff_shrinked)
    except:
        aff_shrinked = 0
        print("Error in aff_shrinked")

    try:
        ckaRbf_score = linear_CKA(input2d, output2d, device=device)
        module.dict['cka_score'].append(ckaRbf_score)
    except:
        ckaRbf_score = 0
        print("Error in cka")

    try:
        ent_before = entropy_batch_calc(input2d)
        module.dict['ent_before'].append(ent_before)
    except:
        ent_before = 0
        print("Error in entropy before")

    try:
        ent_after = entropy_batch_calc(output2d)
        module.dict['ent_after'].append(ent_after)
    except:
        ent_after = 0
        print("Error in entropy after")

    try:
        sparsity_after = sparsity_calc(output.cpu())
        module.dict['sparsity_after'].append(sparsity_after)
    except:
        sparsity_after = 0    
        print("Error in sparsity after")

    try:
        norm_diff = torch.linalg.norm(input2d-output2d)
        module.dict['norm_diff'].append(norm_diff)
    except:
        norm_diff = 0
        print("Error in norm diff")

    try:
        r2 = LinearRegression().fit(input2d.cpu().numpy(), output2d.cpu().numpy()).score(input2d.cpu().numpy(), output2d.cpu().numpy())
        module.dict['r2_score'].append(r2)
    except:
        r2 = 0
        print("Error in R2")

    if verbose:
        print(f'\n{module._get_name()}:')
        print(f"Sparsity before: {sparsity_before}")
        print(f"Sparsity after: {sparsity_after}")
        print(f"Entropy before: {ent_before}")
        print(f"Entropy after: {ent_after}")
        print(f"Aff score: {aff_true}")
        print(f"Aff score corrected cov: {aff_shrinked}")
        print(f"CKA score: {ckaRbf_score}")
        print(f"Norm difference: {norm_diff}")
        print(f"R2 score: {r2}")
        print('\n')

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=987, type=int)
parser.add_argument('--path_data', default='./data/', type=str)
parser.add_argument('--path_save', default='./results/', type=str)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--batch_id', default=0, type=int)
parser.add_argument('--device', default='cuda')
parser.add_argument('--weights', default='DEFAULT')
parser.add_argument('--model_name', default='alexnet', type=str)
parser.add_argument('--val_dataset', nargs='+', default=[''], type=str)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--rerun', action='store_true', default=False)
parser.add_argument('--save_feats', action='store_true', default=False)
parser.add_argument('--eps', type=float, default=1e-6)
args = parser.parse_args()

set_seeds(args.seed)

print(args)

global device 
device = torch.device(args.device)

global verbose
verbose = args.verbose

global save_feats
save_feats = args.save_feats

path_data = Path(args.path_data)
path_save = Path(args.path_save)

if not os.path.isdir(path_save):
    os.makedirs(path_save)


# Get the list of the models
classification_models = list_models(module=models)

# get a list of all available activation functions in PyTorch
activation_functions = [name for name, obj in torch.nn.modules.activation.__dict__.items() 
                        if isinstance(obj, type) and issubclass(obj, torch.nn.Module)][3:] # cut couple of activation functions


# choose which metrics to save
metrics = ['acc@1', 'acc@5']


resultsAll = {}
metaAll = {}

for model_name in classification_models:
    exists = all([os.path.exists(path_save / f"{model_name}_{dataset}_{args.batch_size}_{args.batch_id}.pkl") for dataset in args.val_dataset])
    if exists and not args.rerun:
        print('Found saved results for {}, skipping it.'.format(model_name))
        continue

    if args.model_name is None or args.model_name in model_name: 
        print(model_name)

        resultsAll[model_name] = {}
        metaAll[model_name] = {}

        if args.weights == 'random':
            model = getattr(models, model_name)().to(device)
        elif args.weights == 'ImageNet-1K':
            ModelWeights = models.get_model_weights(model_name).IMAGENET1K_V1
            model = getattr(models, model_name)(weights=ModelWeights).to(device)
        elif args.weights == 'DEFAULT':
            ModelWeights = models.get_model_weights(model_name).DEFAULT
            model = getattr(models, model_name)(weights=ModelWeights).to(device)
        preprocess = ModelWeights.transforms(antialias=True)
        model.eval()
        print(f"Loaded {model_name} successfully.")
        print(model)
        
        for metric in metrics:
            metaAll[model_name][metric] = ModelWeights.meta['_metrics']['ImageNet-1K'][metric]

        metaAll[model_name]['total_params'] = sum(p.numel() for p in model.parameters())
        metaAll[model_name]['trainable_total_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metaAll[model_name]['depth'] = len(list(model.named_modules()))

        for name, module in model.named_modules():
            module_name = module._get_name()
            if 'MultiheadAttention' in module_name:
                continue

            for func in activation_functions: 
                if isinstance(module, getattr(torch.nn.modules.activation, func)) and 'attn' not in name.split('.'):
                    print(f"{name} uses the {func} activation function from the list")

                    resultsAll[model_name][name+'_'+func] = init_dict()
                    module.dict = resultsAll[model_name][name+'_'+func]
                    module.register_forward_pre_hook(hook_fn)

        for dataset in args.val_dataset:
            
            if os.path.exists(path_save / f"{model_name}_{dataset}_{args.batch_size}_{args.batch_id}.pkl") and not args.rerun:
                continue
            print(dataset)

            if dataset == 'cifar100':
                testset =  datasets.CIFAR100(root=path_data / 'cifar100', train=False,
                                            download=True, transform=preprocess)
                data_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                                            shuffle=False, drop_last=True)
            elif dataset == 'cifar10':
                testset =  datasets.CIFAR10(root=path_data / 'cifar10', train=False,
                                            download=True, transform=preprocess)
                data_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                                            shuffle=False, drop_last=True)
                
            elif dataset == 'fashionMNIST':
                testset = datasets.FashionMNIST(root=path_data / 'fmnist', train=False,
                                            download=True, transform=preprocess)
                data_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                                            shuffle=False, drop_last=True)
            elif dataset == 'random':
                data_loader = DataLoader(dataset=RandomDataset((3, 256, 256), 10000),
                                                                    batch_size=args.batch_size, 
                                                                    shuffle=False, drop_last=True)
                
            elif dataset == 'imagenet':
                valset = datasets.ImageFolder( path_data / 'imagenet', preprocess)
                data_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                                            shuffle=False, drop_last=True)
            
            else: # expecting to find a dataset with train folder
                train_dir = args.path_data / dataset / 'train'
                input_data = ImageFolder(train_dir, preprocess)
                data_loader = torch.utils.data.DataLoader(input_data, batch_size=args.batch_size, 
                                                                            shuffle=False, drop_last=True)
            
            with torch.no_grad():
                for batch_id, inputs in enumerate(tqdm(data_loader)):
                    if args.batch_id == -1:
                        if len(inputs)==2:
                            inputs, labels = inputs[0], inputs[1]
                        else:
                            inputs = preprocess(inputs) # case of random dataset where we do not generate labels
                            
                        model(inputs.to(device))
                    elif batch_id == args.batch_id:
                        if len(inputs)==2:
                            inputs, labels = inputs[0], inputs[1]
                        else:
                            inputs = preprocess(inputs) # case of random dataset where we do not generate labels
                            
                        model(inputs.to(device))
                        break
                    
                print("\n")

            with open( path_save / f"{model_name}_{dataset}_{args.batch_size}_{args.batch_id}.pkl", 'wb') as f:
                pickle.dump(resultsAll[model_name], f, protocol=pickle.HIGHEST_PROTOCOL)    
                init_dict(resultsAll[model_name])

with open(path_save / 'metaInfo.pkl', 'wb') as f:
    pickle.dump(metaAll, f, protocol=pickle.HIGHEST_PROTOCOL)   

print("Done !") 
