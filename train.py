from utils import train_args, device_type, build_classifier, pre_trained_model, save_checkpoint

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

import time

import torch
import torch.nn.functional as F

from torch import nn
from torch import optim

from torchvision import datasets, transforms

from utils import timer

    
def start_training(args):
    arch = args.arch
    data_directory = args.data_directory
    dropout = args.dropout
    epochs = args.epochs
    gpu = args.gpu
    hidden_units = args.hidden_units
    learning_rate = args.learning_rate
    save_dir = args.save_dir
    
    print(f"""Preparing training with params
    Architecture: {arch}
    Data Directory: {data_directory}
    Dropout: {dropout * 100.0}%
    Epochs: {epochs}
    Cuda: {gpu}
    Hidden Units: {hidden_units}
    Learning Rate: {learning_rate}
    Save Directory: {save_dir}
    """)
    
    device = device_type(gpu)
    loaders = build_dataloaders(data_directory)
    
    (model, model_layer_size) = pre_trained_model(arch, hidden_units, dropout)
    model.to(device)
    
    train(arch, model, epochs, learning_rate, loaders, device)
    test(model, loaders, device)
    
    save_checkpoint(model, arch, save_dir, loaders, learning_rate, epochs, model_layer_size, hidden_units, dropout)

@timer
def train(arch, model, epochs, lr, loaders, device):
    print('Starting training')
    
    params = None
    
    if arch == 'resnet18':
        params = model.fc.parameters()
    else:
        params = model.classifier.parameters()
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params, lr=lr)
    
    train_losses = []
    valid_losses = []

    for e in range(epochs):
        model.train()
        train_loss = 0

        for ix, (inputs, labels) in enumerate(loaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        else:
            valid_loss = 0
            accuracy= 0

            with torch.no_grad():
                model.eval()

                for images, labels in loaders['valid']:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model.forward(images)
                    valid_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(train_loss/len(loaders['train']))
            valid_losses.append(valid_loss/len(loaders['valid']))
            
            if e == 0:
                print('Epoch\t\tTrain Loss\t\tValid Loss\t\tValid Accuracy')
            
            t_loss = train_loss / len(loaders['train'])
            v_loss = valid_loss / len(loaders['valid'])
            v_accu = accuracy / len(loaders['valid'])
            
            print(f'{e+1}/{epochs}\t\t{t_loss}\t\t{v_loss}\t\t{v_accu * 100.0}%')

            train_loss = 0
    
    print('Finished training')

@timer
def test(model, loaders, device):
    
    criterion = nn.NLLLoss()
    
    test_losses = []
    test_loss = 0

    with torch.no_grad():
        model.eval()
        accuracy = 0

        for images, labels in loaders['test']:
            images, labels = images.to(device), labels.to(device)
            log_ps = model.forward(images)
            test_loss += criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        test_losses.append(test_loss/len(loaders['test']))

        print(f"Test Loss    : {test_loss / len(loaders['test'])}")
        print(f"Test Accuracy: {(accuracy / len(loaders['test'])) * 100.0}%")

@timer
def build_dataloaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data_transforms = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    common_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    # TODO: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=common_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=common_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)
    
    print('Loaders built...')
    
    return {
        'train': train_dataloaders,
        'train_dataset': train_image_datasets,
        'test': test_dataloaders,
        'valid': valid_dataloaders
    }


if __name__ == '__main__':
    parser = train_args()
    args = parser.parse_args()
    
    start_training(args)