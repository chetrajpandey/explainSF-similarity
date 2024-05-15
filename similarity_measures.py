import numpy as np
from scipy.ndimage import gaussian_filter

def rmse(array1, array2):
    # Ensure arrays have the same shape
    assert array1.shape == array2.shape, "Arrays must have the same dimensions"

    # Compute squared difference
    squared_diff = (array1 - array2) ** 2

    # Compute mean squared error
    rmse = np.sqrt(np.mean(squared_diff))

    return rmse


def calculate_iou(binary1, binary2):
    # Convert images to binary

    
    # Calculate intersection
    intersection = np.logical_and(binary1, binary2).sum()
    
    # Calculate union
    union = np.logical_or(binary1, binary2).sum()
    iou = intersection / union if union!=0 else 0
    
    return iou


def cosine_similarity(array1, array2):
    # Normalize arrays if they are not all zeros
    if np.any(array1) and np.any(array2):
        array1_norm = array1 / np.linalg.norm(array1)
        array2_norm = array2 / np.linalg.norm(array2)
        # Compute dot product and cosine similarity
        dot_product = np.dot(array1_norm.flatten(), array2_norm.flatten())
        cosine_similarity = dot_product
    else:
        cosine_similarity = 0  # If arrays are all zeros, cosine similarity is 0
    
    return cosine_similarity