import math
import numpy as np
from scipy import signal


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def gauss(sigma):
    
    ### Your code here
    n = 3*int(sigma)
    x = np.linspace(-n, n, 2*n+1)
    Gx = np.array((1/(np.sqrt(2*math.pi)*sigma))*np.exp(-np.square(x)/(2 * sigma**2)))

    return Gx, x


def gaussianfilter(img, sigma):
    
    ### Your code here
    G, x = gauss(sigma)
    
    # # convolve on x - axis
    GG = G[np.newaxis, :]
    outimage = signal.convolve2d(img,GG,mode='same')

    # convolve on y - axis
    #outimage = signal.convolve2d(outimage,GG.T,mode='same')

    return outimage


def gaussdx(sigma):
    ### Your code here
    n = 3*int(sigma)
    x = np.linspace(-n, n, 2*n+1)
    D = np.array([-xx/(np.sqrt(2*math.pi)*sigma**3)*math.exp((-xx**2)/(2 * sigma**2)) for xx in x])
    return D, x

def gaussian_derivative(x, y, sigma):
    """
    Computes the derivative of a 2D Gaussian function with respect to x and y.
    """
    
    # exponential term of the Gaussian function
    exp_term = np.exp(-(np.square(x) + np.square(y))/(2*sigma**2)) / (math.pi*sigma**2)
    
    # Compute the derivatives of the Gaussian function with respect to x and y
    dx = -x/(sigma**2) * exp_term
    dy = -y/(sigma**2) * exp_term
    
    return dx, dy

def gaussderiv(img, sigma):
    """
    Convolves an image with the derivative of a 2D Gaussian function.
    """
    # Your code here
    n = int(3 * sigma)
    x = np.linspace(-n, n, 2*n+1)
    y = np.linspace(-n, n, 2*n+1)
    xx, yy = np.meshgrid(x, y)
    
    # derivative of the Gaussian function with respect to x and y
    dx, dy = gaussian_derivative(xx, yy, sigma)
    
    # Convolve the image with the derivative of the Gaussian function
    imgDx = signal.convolve2d(img, dx, mode='same')
    imgDy = signal.convolve2d(img, dy, mode='same')
    
    return imgDx, imgDy

