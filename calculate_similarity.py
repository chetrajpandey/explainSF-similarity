from similarity_measures import calculate_iou, cosine_similarity, rmse
from scipy.ndimage import gaussian_filter
from load_explanations import load_data
import numpy as np

alex_gc, alex_dp, alex_it, vgg_gc, vgg_dp, vgg_it = load_data()

#Functions for binarizing 
def binarize(array1):    
    # Apply median filtering
    filtered_image = gaussian_filter(array1, sigma=10)
    binary_image = (filtered_image > 0).astype(int)
    return binary_image.astype(int)

#Function for applying Gaussian blur
def filtering(array1):
    filtered_image = gaussian_filter(array1, sigma=10)
#     print(np.max(binary))
    return filtered_image.astype('uint8')


# Define a function to calculate a metric for a pair of arrays
def calculate_metric(metric_function, arr1, arr2):
    # Calculate the metric using the provided metric function
    return metric_function(arr1, arr2)

# Define your lists of arrays
arrays = {
    "alex_gc": alex_gc,
    "alex_it": alex_it,
    "alex_dp": alex_dp,
    "vgg_gc": vgg_gc,
    "vgg_it": vgg_it,
    "vgg_dp": vgg_dp
}

# Define a list of metric functions
metric_functions = [rmse, calculate_iou, cosine_similarity]

def calculate_sim():
    # Loop through each metric function
    for metric_function in metric_functions:
        print(f"Metric: {metric_function.__name__}")

        # Loop through each combination of array pairs
        for array1_name, array1 in arrays.items():
            for array2_name, array2 in arrays.items():
                # Ensure array1_name comes before array2_name lexicographically
                if array1_name < array2_name:
                    # Initialize a list to store metric values for each pair of arrays
                    metric_values = []

                    # Loop through each pair of arrays
                    for arr1, arr2 in zip(array1, array2):
                        # Compute metric for each pair of arrays and append to the list
                        metric_values.append(calculate_metric(metric_function, arr1, arr2))

                    # Compute the average metric value for this combination
                    average_metric = np.mean(metric_values)

                    # Print the average metric value along with array names
                    print(f"Arrays: {array1_name}_{array2_name}, Average: {average_metric}")

def calculate_sim_with_blur():
    # Loop through each metric function
    for metric_function in metric_functions:
        print(f"Metric: {metric_function.__name__}")
        if metric_function.__name__ == "rmse" or metric_function.__name__ == "cosine_similarity":
            preprocess = filtering
        else:
            preprocess = binarize

        # Loop through each combination of array pairs
        for array1_name, array1 in arrays.items():
            for array2_name, array2 in arrays.items():
                # Ensure array1_name comes before array2_name lexicographically
                if array1_name < array2_name:
                    # Initialize a list to store metric values for each pair of arrays
                    metric_values = []

                    # Loop through each pair of arrays
                    for arr1, arr2 in zip(array1, array2):
                        arr1, arr2 = preprocess(arr1), preprocess(arr2)
    #                     print(np.max(arr2))
                        # Compute metric for each pair of arrays and append to the list
                        metric_values.append(calculate_metric(metric_function, arr1, arr2))

                    # Compute the average metric value for this combination
                    average_metric = np.mean(metric_values)

                    # Print the average metric value along with array names
                    print(f"Arrays: {array1_name}_{array2_name}, Average: {average_metric}")


if __name__ == "__main__":
    print('Similarity Scores without blur\n')
    calculate_sim()
    print('Similarity Scores with blur\n')
    calculate_sim_with_blur()


