import cv2
import numpy as np
from cap.dehaze import dehaze as cap_dehaze_img
from dcp.dehaze import dehaze as dcp_dehaze_img
from cap.dehaze import dehaze_video as cap_dehaze_vid
from dcp.dehaze import dehaze_video as dcp_dehaze_vid
from config import config
from sklearn.metrics import mean_squared_error as compute_mse
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from matplotlib import pyplot as plt

def resize_aspect_ratio(img, size = 1280, interp=cv2.INTER_LINEAR):
    """
    resize image and keep aspect ratio
    """
    if len(img.shape) == 2:
        h,w = img.shape
    elif len(img.shape) == 3:
        h,w,_ = img.shape
    else:
        return None

    new_w = size
    new_h = h*new_w//w    
    img_rs = cv2.resize(img.copy(), (new_w, new_h), interpolation=interp)
    return img_rs

def show_compaire(hazy, is_save = False):
    """
    visualize the comparison of single image dehzing methods on RESIDE dataset
    param:
        hazy: is the path to hazy image
        is_save: boolean flag to save the comparison results
    return:
        
    """
    hazy_img = cv2.imread(hazy)
    img_id = hazy.split('\\')[-1].split('_')[0]
    clear = hazy.replace('hazy','clear')
    clear_ele = clear.split('\\')
    clear_ele[-1] = img_id + '.png'
    clear = '\\'.join(clear_ele)
    clear_img = cv2.imread(clear)
    
    dehaze_1 = dcp_dehaze_img(hazy, config['dcp'])
    dehaze_2 = cap_dehaze_img(hazy, config['cap'])
    
    psnr_1 = compute_psnr(clear_img, dehaze_1)
    psnr_2 = compute_psnr(clear_img, dehaze_2)
    
    ssim_1 = compute_ssim(clear_img, dehaze_1 , multichannel=True)
    ssim_2 = compute_ssim(clear_img, dehaze_2 , multichannel=True)
    
    mse_1 = compute_mse(clear_img.flatten(), dehaze_1.flatten())
    mse_2 = compute_mse(clear_img.flatten(), dehaze_2.flatten())
    
    fig = plt.figure(figsize=(30, 30), dpi=80)
    ax1 = fig.add_subplot(1,4,1)
    ax1.imshow(hazy_img)
    plt.axis('off')
    ax1.set_title('Hazy Image')
    ax2 = fig.add_subplot(1,4,2)
    ax2.imshow(dehaze_1)
    plt.axis('off')
    ax2.set_title(f'DCP - PSNR: {psnr_1:.3f} - SSIM: {ssim_1:.3f} - MSE: {mse_1:.3f}')

    ax3 = fig.add_subplot(1,4,3)
    ax3.imshow(dehaze_2)
    plt.axis('off')
    ax3.set_title(f'CAP - PSNR: {psnr_2:.3f} - SSIM: {ssim_2:.3f} - MSE: {mse_2:.3f}')

    ax4 = fig.add_subplot(1,4,4)
    ax4.imshow(clear_img)
    plt.axis('off')
    ax4.set_title('Clear Image')
    
    if is_save:  
        case_name = clear.replace('\\','/').split('/')[-1].replace('.png','')
        fig.savefig(os.path.join(_RESULT_PATH,case_name+'.png'))
        plt.close(fig) 
    plt.show()