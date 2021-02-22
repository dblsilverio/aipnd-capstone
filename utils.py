import argparse
import json
import numpy as np
import os
import time
import torch

from PIL import Image
from torch import nn
from torchvision import datasets, transforms, models

CATEGORIES = None


def timer(fn):
    
    def wrapper(*a, **kwa):
        start = time.time()
        value = fn(*a, **kwa)
        finish = time.time()
        
        total = finish - start
        print('\'{}()\' execution time: {:.2f}ms'.format(fn.__name__, total))
        
        return value
    return wrapper

def train_args():
    ###Defines arguments for the training module###
    
    archs_available = ['alexnet', 'resnet18', 'vgg16']
    
    parser = argparse.ArgumentParser(
        description='This is the trainer module for the AI Programming With Python Nanodegree Project')
    
    parser.add_argument('data_directory', help='Directory with train/validation/test datasets')
    
    parser.add_argument('--arch', help='Choose the architecture for the network', choices=archs_available, default="vgg16")
    
    parser.add_argument('--save_dir', help='Directory to store trained models/checkpoints', default="./saved_models")
    
    parser.add_argument('--learning_rate', help='Learning rate to be applied', type=float, default=0.002)
    
    parser.add_argument('--hidden_units', help='Hidden units for the network ', default=4096)
    
    parser.add_argument('--epochs', help='Number of epochs for training the network', type=int, default=3)
    
    parser.add_argument('--dropout', help='Dropout applied to traning the network', type=float, default=0.1)
    
    parser.add_argument('--gpu', help='Pick the GPU device for training', action='store_true', default=False)
    
    return parser

def predict_args():
    ###Defines arguments for the predict module###
       
    parser = argparse.ArgumentParser(
        description='This is the predict module for the AI Programming With Python Nanodegree Project')
    
    parser.add_argument('input_image', help='The image for prediction')
    
    parser.add_argument('checkpoint', help='Checkpoint file to load')
    
    parser.add_argument('-t', '--top_k', help='Display top n categories', type=int, default=5)
    
    parser.add_argument('-c', '--category_names', help='Select category mapping file', default='./cat_to_name.json')
    
    parser.add_argument('-g', '--gpu', help='Pick the GPU device for training', action='store_true', default=False)
    
    return parser


def category_name(id):
    return CATEGORIES[f'{id}']

def model_categories(category_names_file, class_to_idx):
    
    if not CATEGORIES:
        load_categories(category_names_file)
    
    model_categories = {ix: (value, CATEGORIES[value]) for value, ix in class_to_idx}
    
    return model_categories

def load_categories(category_names_file):
    global CATEGORIES
    
    with open(category_names_file, 'r') as f:
        print(f'Reading categories file: {category_names_file}')
        CATEGORIES = json.load(f)

def process_image(image_path):

    mean = np.array([.485, .456, .406])
    stddev = np.array([.229, .224, .225])
    
    image = Image.open(image_path)
    
    old_w, old_h = image.size
    new_w, new_h = 0, 0
    
    if old_w <= old_h:
        new_w = 256
        new_h = int(256 * (old_h / old_w))
    else:
        new_h = 256
        new_w = int(256 * (old_w / old_h))
    
    print(f'Resized original image from ({old_w}, {old_h}) to ({new_w}, {new_h})')
    
    image.thumbnail((new_w, new_h), Image.ANTIALIAS)
    
    new_size_crop = 224
    
    n_left = (new_w - new_size_crop)/2
    n_top = (new_h - new_size_crop)/2
    n_right = (new_w + new_size_crop)/2
    n_bottom = (new_h + new_size_crop)/2

    image = image.crop((n_left, n_top, n_right, n_bottom))
    image_np = np.array(image)
    
    image_np = image_np / 255.0
    
    image_np = (image_np - mean) / stddev
    image_np = image_np.transpose((2, 0, 1))
    
    print(f'Image \'{image_path}\' processed')
    
    return image_np   

def pre_trained_model(name, hidden_units, dropout):
    """Select pre-trained model based on architecture argument provided"""
    
    models_available = {
        'alexnet': {
            'method': 'models.alexnet',
            'size': 'model.classifier[1].in_features'
        },
        'resnet18': {
            'method': 'models.resnet18',
            'size': 'model.fc.in_features'
        },
        'vgg16': {
            'method': 'models.vgg16',
            'size': 'model.classifier[0].in_features'
        }
    }
    
    model = eval(models_available[name]['method'])(pretrained=True)
    model_layer_size = eval(models_available[name]['size'])
    
    print(f'{name} pre-trained selected...')

    for param in model.parameters():
        param.requires_grad = False
        
    clf = build_classifier(model_layer_size, hidden_units, dropout)    
     
    if name in ('alexnet', 'vgg16'):
        model.classifier = clf
    else:
        model.fc = clf
       
    return (model, model_layer_size)

@timer
def build_classifier(model_layer_size, hidden_units, dropout):
    
    clf = nn.Sequential(
        nn.Linear(model_layer_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    print(f'Classifier built with {model_layer_size}x{hidden_units} and dropout of {dropout * 100.0}%')
    
    return clf

@timer
def save_checkpoint(model, architecture, save_dir, loaders, learning_rate, epochs, model_layer_size, hidden_units, dropout, checkpoint_name='checkpoint.pth'):
    model.class_to_idx = loaders['train_dataset'].class_to_idx
    
    if not os.path.exists(save_dir):
        print(f'Creating directory: {save_dir}')
        os.makedirs(save_dir)
    
    checkpoint = {
        'architecture': architecture,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'hidden_units': hidden_units,
        'model_layer_size': model_layer_size,
        'dropout': dropout
    }
    
    torch.save(checkpoint, f'{save_dir}/{checkpoint_name}')
    print(f'Checkpoint saved at {save_dir}/{checkpoint_name}')

@timer
def load_checkpoint(checkpoint_path, device, train=False):
    checkpoint = torch.load(checkpoint_path)
    
    architecture = checkpoint['architecture']
    hidden_units = checkpoint['hidden_units']
    model_layer_size = checkpoint['model_layer_size']
    dropout = checkpoint['dropout']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']
    
    (trained_model, model_layer_size) = pre_trained_model(architecture, hidden_units, dropout)

    trained_model.load_state_dict(state_dict)
    trained_model.class_to_idx = class_to_idx

    trained_model.to(device)
    
    if not train:
        trained_model.eval()
    else:
        trained_model.train()
    
    print(f'Checkpoint loaded from {checkpoint_path}')
    print(f"""Detected model params
    Architecture: {architecture}
    Dropout: {dropout * 100.0}%
    Hidden Units: {hidden_units}
    """)
    
    return trained_model
        
def device_type(gpu):
    """
    Selects the device based on availability of a GPU with CUDA enabled
    """
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    
    print(f'Running with {device} device...')
    
    return device
