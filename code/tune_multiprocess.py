#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from cap.dehaze import dehaze as cap_dehaze
from dcp.dehaze import dehaze as dcp_dehaze
from config import config
import os
from glob import glob
from matplotlib import pyplot as plt
import cv2
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
import multiprocessing as mp
from tqdm import tqdm
from sklearn.metrics import mean_squared_error as compute_mse
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

_RESIDE_PATH = '/mnt/d/data/reside/indoor_train'
_RESULT_PATH = '../result'


# In[3]:


hazy_fold_path = os.path.join(_RESIDE_PATH,'hazy')
clear_fold_path = os.path.join(_RESIDE_PATH,'clear')


def get_clear(hazy_img_path, clear_fold_path):
    hazy_img_path = hazy_img_path.replace('\\', '/')
    clear_file = hazy_img_path.split('/')[-1].split('_')[0] + '.png'
    clear_img_path = os.path.join(clear_fold_path, clear_file)
    return clear_img_path


# list_hazy_imgs = glob(os.path.join(hazy_fold_path, '*.png'))[:20]
list_hazy_imgs = glob(os.path.join(hazy_fold_path, '*_10_*.png'))[:50]


def pipeline_finetune(hazy):
    clear = get_clear(hazy, clear_fold_path)
    clear_img = cv2.imread(clear)
    dehaze_img = dcp_dehaze(hazy, config['dcp'])
    psnr = compute_psnr(clear_img, dehaze_img)
    ssim = compute_ssim(clear_img, dehaze_img, multichannel = True)
    return psnr, ssim


pool = mp.Pool(4)

list_omegas = list(np.arange(0.1, 1.0, 0.05))


list_psnr = []
list_ssim = []
for omg in list_omegas:
    print(f'------ Current omega: {omg} ------')
    config['dcp']['omega'] = omg
    psnrs = []
    ssims = []
    for (psnr, ssim) in tqdm(pool.imap_unordered(pipeline_finetune,list_hazy_imgs), total=len(list_hazy_imgs)):
        psnrs.append(psnr)
        ssims.append(ssim)
    list_psnr.append(np.array(psnrs))
    list_ssim.append(np.array(ssims))

file_psnr = open(os.path.join(_RESULT_PATH,f'psnr_{len(list_hazy_imgs)}.npy'), 'wb')
np.save(file_psnr, np.array(list_psnr))

file_ssim = open(os.path.join(_RESULT_PATH,f'ssim_{len(list_hazy_imgs)}.npy'), 'wb')
np.save(file_ssim, np.array(list_ssim))

