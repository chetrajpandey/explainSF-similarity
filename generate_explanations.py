import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor
import torch.nn.functional as F
import torch.nn as nn
from models import Custom_AlexNet, Custom_VGG16


from captum.attr import IntegratedGradients, DeepLiftShap, GuidedGradCam

from captum.attr import visualization as viz

from torch.utils.data import Dataset, DataLoader
import warnings
warnings.simplefilter("ignore", Warning)
device = torch.device('cuda')


class MyJP2Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, rgb=True):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.rgb = rgb

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        if self.rgb:  # Simplified the condition
            hmi = Image.open(img_path).convert('RGB')
        else:
            hmi = Image.open(img_path)   

        if self.transform:
            image = self.transform(hmi)
            
        y_prob = round(float((self.annotations.iloc[index, 1])), 2)
        y_label = str(self.annotations.iloc[index, 2])
        
        return (image, y_prob, y_label)

    def __len__(self):
        return len(self.annotations)


def load_model(model_class, weights_path, device='cuda'):
    """
    Load a PyTorch model from the given weights file.

    Args:
        model_class (torch.nn.Module): The class of the model to instantiate.
        weights_path (str): The path to the weights file.
        device (str): The device to load the model on (default is 'cuda').

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Instantiate the model
    model = model_class().to(device)
    
    # Load the weights
    weights = torch.load(weights_path)
    model.load_state_dict(weights['model_state_dict'])
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def attribute_image_features(test_model, algorithm, input, target, **kwargs):
    test_model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=target,
                                              **kwargs
                               )
    return tensor_attributions
    
def plot_attributions_guided_grad_cam(test_model, img, target, layer_index):
    inp = img.unsqueeze(0)
    inp.requires_grad = True
    guided_gc = GuidedGradCam(test_model, test_model.features[layer_index])
    grads = guided_gc.attribute(inp.to(device), target=target)
    grads = np.transpose(grads.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
#     print(grads.shape)
    x = np.abs(grads.squeeze(2))
    grads = ((x - np.min(x)) / ( np.max(x) - np.min(x) ))* 255
    grads = grads.astype(np.uint8)
    return grads
    
    
def plot_attributions_deepshap(test_model, img, i, target):
    inp = img.unsqueeze(0)
    inp.requires_grad = True
    saliency = DeepLiftShap(test_model)
    if i < 12:
        grads = saliency.attribute(inp.to(device), baselines= images[i:i+12].to(device), target=target)
    else:
        grads = saliency.attribute(inp.to(device), baselines= images[i-12:i].to(device), target=target)
    grads = np.transpose(grads.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    x = np.abs(grads.squeeze(2))
    grads = ((x - np.min(x)) / ( np.max(x) - np.min(x) ))* 255
    grads = grads.astype(np.uint8)
    return grads
    
    
def plot_attributions_intgrad(test_model, img, target):

    inp = img.unsqueeze(0)
    inp.requires_grad = True
    inp=inp.to(device)
    ig = IntegratedGradients(test_model)
    attr_ig, delta = attribute_image_features(ig, inp, target, baselines=inp * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    x = np.abs(attr_ig.squeeze(2))
    grads = ((x - np.min(x)) / ( np.max(x) - np.min(x) ))* 255
    grads = grads.astype(np.uint8)
    return grads


if __name__ == "__main__":
    # Load combined folds data
    combine_all_folds = pd.DataFrame()
    for x in range(1, 5):
        combine_all_folds = pd.concat([combine_all_folds, pd.read_csv(f'flare_labels_with_predictions/vgg/fold{x}_res.csv')], axis=0)
    combine_all_folds['timestamp'] = combine_all_folds['timestamp'].str.replace('/scratch/cpandey1/hmi_jpgs_512/', '')
    combine_all_folds = combine_all_folds[combine_all_folds.target == 1]
    combine_all_folds.to_csv(r'flare_labels_with_predictions/combined_res.csv', index=False)

    csv_file = 'flare_labels_with_predictions/combined_res.csv'
    # This datapath is based on downloaded magnetograms using the code inside download_mag
    data_path = '/data/hmi_jpgs_512/'
    data_transforms = Compose([ToTensor()])
    dataset1 = MyJP2Dataset(csv_file=csv_file, root_dir=data_path, transform=data_transforms, rgb=False)
    batch_size = len(combine_all_folds)
    loader = DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=False, num_workers=8)

    dataiter = iter(loader)
    images, probs, labels = next(dataiter)

    # Choose the model to be loaded
    vgg_model = load_model(Custom_VGG16, 'vgg16_trained.pth', device='cuda')
    
    ggcam = []
    deepshap = []
    ig = []
    for i in range(len(images)):
        img = images[i]
        ggcam.append(plot_attributions_guided_grad_cam(vgg_model, img, 1, 28)) # For AlexNet layer_index=10
        deepshap.append(plot_attributions_deepshap(vgg_model, img, i, 1))
        ig.append(plot_attributions_intgrad(vgg_model, img, 1))

    np.save("explanation_maps/vgg_ggcam.npy", ggcam)
    np.save("explanation_maps/vgg_deepshap.npy", deepshap)
    np.save("explanation_maps/vgg_ig.npy", ig)

