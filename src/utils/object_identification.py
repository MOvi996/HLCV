from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from .image_filtering import rgb2gray
from .image_histograms import is_grayvalue_hist, get_dist_by_name, get_hist_by_name


def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = is_grayvalue_hist(hist_type)

    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

    D = np.zeros((len(model_images), len(query_images)))

    # Your code here
    for i, qh in enumerate(query_hists):
        for j, mh in enumerate(model_hists):
            D[j, i] = get_dist_by_name(qh, mh, dist_type)
            
    best_match = np.argmin(D, axis=0)
    return best_match, D


def compute_histograms(image_list, hist_type, hist_isgray, num_bins):

    image_hist = []

    # Compute hisgoram for each image and add it at the bottom of image_hist
    # Your code here
    
    for img in image_list:
        img_arr = np.array(Image.open(img))
        if hist_isgray:
            img_arr = rgb2gray(img_arr.astype('double'))
        else:
            img_arr = img_arr.astype('double')

        image_hist.append(get_hist_by_name(img_arr, num_bins, hist_type))

    return np.array(image_hist)


def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

    plt.figure()
    num_nearest = 5  # Show the top-5 neighbors
    
    # Your code here
    
    [_, D] = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    neighbors = np.argsort(D, axis=0)[:num_nearest]
    
    for i in range(len(query_images)):
        plt.figure()
        plt.subplot(1,num_nearest+1,1)
        plt.title('query image').set_fontsize(6)
        plt.imshow(np.array(Image.open(query_images[i])))

        for j in range(num_nearest):
            plt.subplot(1,num_nearest+1,1+j+1)
            plt.title(f'neighbor {j+1}').set_fontsize(6)
        
            plt.imshow(np.array(Image.open(model_images[neighbors[j][i]])))
            


