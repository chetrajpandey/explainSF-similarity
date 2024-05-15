import numpy as np

def load_data():
    alex_gc = np.load("explanation_maps/alex_ggcam.npy")
    alex_dp = np.load("explanation_maps/alex_deepshap.npy")
    alex_it = np.load("explanation_maps/alex_ig.npy")

    vgg_gc = np.load("explanation_maps/vgg_ggcam.npy")
    vgg_dp = np.load("explanation_maps/vgg_deepshap.npy")
    vgg_it = np.load("explanation_maps/vgg_ig.npy")

    return alex_gc, alex_dp, alex_it, vgg_gc, vgg_dp, vgg_it
