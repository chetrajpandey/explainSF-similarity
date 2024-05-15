from load_explanations import load_data

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

# Call the load_data function to load the data
alex_gc, alex_dp, alex_it, vgg_gc, vgg_dp, vgg_it = load_data()


def plot_images_and_save(alex_gc, alex_it, alex_dp, vgg_gc, vgg_it, vgg_dp, apply_gaussian=True):
    # Create a GridSpec to define the layout
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.03])  # 2 rows, 4 columns, 4th column is for color bar

    # Function to apply Gaussian filter if required
    def apply_filter(image):
        if apply_gaussian:
            return gaussian_filter(image, sigma=10)
        else:
            return image

    # Plotting the first row of images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(apply_filter(alex_gc[50]), cmap='plasma')
    ax1.set_title('')  # Remove title
    ax1.axis('off')

    # Plotting the second image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(apply_filter(alex_it[50]), cmap='plasma')
    ax2.set_title('')  # Remove title
    ax2.axis('off')

    # Plotting the third image
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(apply_filter(alex_dp[50]), cmap='plasma')
    ax3.set_title('')  # Remove title
    ax3.axis('off')

    # Plotting the second row of images
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(apply_filter(vgg_gc[50]), cmap='plasma')
    ax4.set_title('')  # Remove title
    ax4.axis('off')

    # Plotting the second image
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(apply_filter(vgg_it[50]), cmap='plasma')
    ax5.set_title('')  # Remove title
    ax5.axis('off')

    # Plotting the third image
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(apply_filter(vgg_dp[50]), cmap='plasma')
    ax6.set_title('')  # Remove title
    ax6.axis('off')

    # Add color bar spanning two rows
    cax = fig.add_subplot(gs[:, 3])
    plt.colorbar(ax1.imshow(alex_gc[50], cmap='plasma'), cax=cax)
    cax.yaxis.set_label_position('left')
    cax.yaxis.set_ticks_position('left')

    # Ensure 'plot' directory exists
    if not os.path.exists('plot'):
        os.makedirs('plot')

    # Save each subplot individually as SVG and PDF files
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        array_name = ['alex_gc', 'alex_it', 'alex_dp', 'vgg_gc', 'vgg_it', 'vgg_dp'][i]  # Name of arrays
        filename_svg = f'plot/gauss_{array_name}.svg'
        filename_pdf = f'plot/gauss_{array_name}.pdf'
        fig = plt.figure(figsize=(3, 3))  # Create a new figure for each subplot
        ax_new = fig.add_subplot(111)
        ax_new.imshow(ax.get_images()[0].get_array(), cmap='plasma')
        ax_new.axis('off')
        plt.savefig(filename_svg, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.savefig(filename_pdf, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.close('all')  # Close all figures

    plt.tight_layout()
    plt.show()

# Call the function with the appropriate arguments
plot_images_and_save(alex_gc, alex_it, alex_dp, vgg_gc, vgg_it, vgg_dp, apply_gaussian=True)
plot_images_and_save(alex_gc, alex_it, alex_dp, vgg_gc, vgg_it, vgg_dp, apply_gaussian=False)
