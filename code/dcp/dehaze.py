#!/usr/bin/env python
# coding: utf-8

# In[24]:


import cv2;
import math;
import numpy as np;
from matplotlib import pyplot as plt
# from config import config


# In[44]:


def extract_dark_channel(im,config):
    """
    extract depth map
    """
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(config['sz'],config['sz']))
    dark = cv2.erode(dc,kernel)
    return dark

def estimate_atmospheric(im,dark):
    """
    estimate atmospheric
    """
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def estimate_depth_map(im,A,config):
    """
    estimate depth map
    """
#     omega = 0.85;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - config['omega']*extract_dark_channel(im3,config);
    return transmission

def Guidedfilter(im,et,config):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(config['r'],config['r']));
    mean_p = cv2.boxFilter(et, cv2.CV_64F,(config['r'],config['r']));
    mean_Ip = cv2.boxFilter(im*et,cv2.CV_64F,(config['r'],config['r']));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(config['r'],config['r']));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + config['eps']);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(config['r'],config['r']));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(config['r'],config['r']));

    q = mean_a*im + mean_b;
    return q;

# def refine_depth_map(img_path,et, config):
def refine_depth_map(img, et, config):
#     img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    t = Guidedfilter(gray,et,config);

    return t;

def recover(im,t,A,config):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,config['tx']);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res
    


# In[45]:


def dehaze(I, config):
#     I = cv2.imread(img_path);
    I_norm = I.copy()
    I_norm = I_norm/255.
    dark = extract_dark_channel(I_norm,config)
    A = estimate_atmospheric(I_norm,dark)
    te = estimate_depth_map(I_norm,A,config)
    t = refine_depth_map(I, te, config)
    J = (recover(I_norm,t,A,config)*255).astype(np.uint8)
    return J


# In[47]:


# get_ipython().system('jupyter nbconvert --to script dehaze.ipynb')

