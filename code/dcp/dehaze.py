#!/usr/bin/env python
# coding: utf-8

# In[24]:


import cv2;
import math;
import numpy as np;
from matplotlib import pyplot as plt
import timeit
# import numpy as np
# import numpy.linalg
# import scipy.sparse
import scipy
from tqdm import tqdm


def extract_dark_channel(im,config):
    """
    extract depth map
    """
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
#     print('dc', dc.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(config['sz'],config['sz']))
    dark = cv2.erode(dc,kernel)
#     print('dark', dark.shape)
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

def refine_depth_map(img, et, config):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    t = Guidedfilter(gray,et,config);
    return t;

def recover(im,t,A,config):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,config['tx']);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res
    

def dehaze(I, config):
    """
    DCP
    """
    I_norm = I.copy()
    I_norm = I_norm/255.
    t1 = timeit.default_timer()
    dark = extract_dark_channel(I_norm,config)
    t2 = timeit.default_timer()

    atm = estimate_atmospheric(I_norm,dark)
    t3 = timeit.default_timer()
    te = estimate_depth_map(I_norm,atm,config)
    t4 = timeit.default_timer()
    t = refine_depth_map(I, te, config)
    t5 = timeit.default_timer()
    J = (recover(I_norm,t,atm,config)*255).astype(np.uint8)
    t6 = timeit.default_timer()
#     dict_time = {}
#     dict_time['ext_dm'] = t2 - t1
#     dict_time['est_atm'] = t3 - t2
#     dict_time['est_dm'] = t4 - t3
#     dict_time['ref_dm'] = t5 - t4
#     dict_time['rec_rad'] = t6 - t5
#     print(dict_time)

    avg_trans = np.mean(te)
    
    return J

def dehaze_video(I, config, g_atm, bov = False, mu = 0.4):
    """
    DCP using global atmospheric
    """
    I_norm = I.copy()
    I_norm = I_norm/255.
    t1 = timeit.default_timer()
    dark = extract_dark_channel(I_norm,config)
    t2 = timeit.default_timer()
    atm = estimate_atmospheric(I_norm,dark)
    if bov:
        g_atm = atm
    else:
#         print(f'g_atm: {g_atm}; atm: {atm}')
        g_atm = mu*g_atm + (1-mu)*atm
    t3 = timeit.default_timer()
    te = estimate_depth_map(I_norm, g_atm, config)
    t4 = timeit.default_timer()
    t = refine_depth_map(I, te, config)
    t5 = timeit.default_timer()
    J = (recover(I_norm,t, g_atm, config)*255).astype(np.uint8)
    t6 = timeit.default_timer()
    avg_trans = np.mean(te)
    
    return J, g_atm


if __name__ == 'main':
    img_path = '../../data/aerial.png'
    print(img_path)
    iimg = cv2.imread(img_path)
    dehaze(iimg, config)
#     plt.imshow(dehaze(iimg, config))

