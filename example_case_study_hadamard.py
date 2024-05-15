import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from load_explanations import load_data

# Call the load_data function to load the data
alex_gc, alex_dp, alex_it, vgg_gc, vgg_dp, vgg_it = load_data()


def visualize_hadamard_products():

    # Selecting the 50th image from each of the arrays
    image1 = vgg_gc[50]
    image2 = vgg_it[50]
    image3 = vgg_dp[50]
    image4 = alex_gc[50]
    image5 = alex_it[50]
    image6 = alex_dp[50]

    # Apply Gaussian filter to each image
    sigma = 5
    image1_gaussian = gaussian_filter(image1, sigma=sigma)
    image2_gaussian = gaussian_filter(image2, sigma=sigma)
    image3_gaussian = gaussian_filter(image3, sigma=sigma)
    image4_gaussian = gaussian_filter(image4, sigma=sigma)
    image5_gaussian = gaussian_filter(image5, sigma=sigma)
    image6_gaussian = gaussian_filter(image6, sigma=sigma)

    # Calculate the Hadamard product for the first set of images (1, 2, 3)
    hadamard_product_123 = image1 * image2 * image3
    hadamard_product_123_gaussian = image1_gaussian * image2_gaussian * image3_gaussian

    # Calculate the Hadamard product for the second set of images (4, 5, 6)
    hadamard_product_456 = image4 * image5 * image6
    hadamard_product_456_gaussian = image4_gaussian * image5_gaussian * image6_gaussian

    # Calculate the Hadamard product for all six images (1, 2, 3, 4, 5, 6)
    hadamard_product_123456 = image1 * image2 * image3 * image4 * image5 * image6
    hadamard_product_123456_gaussian = image1_gaussian * image2_gaussian * image3_gaussian * image4_gaussian * image5_gaussian * image6_gaussian

    # Scale back to 0-255 range
    hadamard_product_123_scaled = hadamard_product_123 / np.max(hadamard_product_123) * 255
    hadamard_product_456_scaled = hadamard_product_456 / np.max(hadamard_product_456) * 255
    hadamard_product_123456_scaled = hadamard_product_123456 / np.max(hadamard_product_123456) * 255

    hadamard_product_123_gaussian_scaled = hadamard_product_123_gaussian / np.max(hadamard_product_123_gaussian) * 255
    hadamard_product_456_gaussian_scaled = hadamard_product_456_gaussian / np.max(hadamard_product_456_gaussian) * 255
    hadamard_product_123456_gaussian_scaled = hadamard_product_123456_gaussian / np.max(hadamard_product_123456_gaussian) * 255

    # Define font size for labels
    font_size = 12

    # Display the resulting images
    plt.figure(figsize=(18, 12))

    # Plotting Hadamard product before applying Gaussian filter
    plt.subplot(2, 3, 1)
    plt.imshow(hadamard_product_123_scaled, cmap='plasma')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(hadamard_product_456_scaled, cmap='plasma')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(hadamard_product_123456_scaled, cmap='plasma')
    plt.axis('off')

    # Plotting Hadamard product after applying Gaussian filter
    plt.subplot(2, 3, 4)
    plt.imshow(hadamard_product_123_gaussian_scaled, cmap='plasma')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(hadamard_product_456_gaussian_scaled, cmap='plasma')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(hadamard_product_123456_gaussian_scaled, cmap='plasma')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('newplots/hadamard.svg')
    plt.show()

# Call the function to execute the code
visualize_hadamard_products()

