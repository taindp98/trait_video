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


def closed_form_laplacian(image, epsilon=1e-7, r=1):
    h,w = image.shape[:2]
    window_area = (2*r + 1)**2
    n_vals = (w - 2*r)*(h - 2*r)*window_area**2
    k = 0
    # data for matting laplacian in coordinate form
    i = np.empty(n_vals, dtype=np.int32)
    j = np.empty(n_vals, dtype=np.int32)
    v = np.empty(n_vals, dtype=np.float64)

    # for each pixel of image
    for y in tqdm(range(r, h - r)):
        for x in range(r, w - r):

            # gather neighbors of current pixel in 3x3 window
            n = image[y-r:y+r+1, x-r:x+r+1]
#             print('n', n.shape)
            u = np.zeros(3)
            for p in range(3):
                u[p] = n[:, :, p].mean()
            c = n - u

            # calculate covariance matrix over color channels
            cov = np.zeros((3, 3))
            for p in range(3):
                for q in range(3):
                    cov[p, q] = np.mean(c[:, :, p]*c[:, :, q])

            # calculate inverse covariance of window
            inv_cov = np.linalg.inv(cov + epsilon/window_area * np.eye(3))

            # for each pair ((xi, yi), (xj, yj)) in a 3x3 window
            for dyi in range(2*r + 1):
                for dxi in range(2*r + 1):
                    for dyj in range(2*r + 1):
                        for dxj in range(2*r + 1):
                            i[k] = (x + dxi - r) + (y + dyi - r)*w
                            j[k] = (x + dxj - r) + (y + dyj - r)*w
                            temp = c[dyi, dxi].dot(inv_cov).dot(c[dyj, dxj])
                            v[k] = (1.0 if (i[k] == j[k]) else 0.0) - (1 + temp)/window_area
                            k += 1
#         print("generating matting laplacian", y - r + 1, "/", h - 2*r)

    return i, j, v

def make_system(L, trimap, constraint_factor=100.0):
    # split trimap into foreground, background, known and unknown masks
    is_fg = (trimap > 0.9).flatten()
    is_bg = (trimap < 0.1).flatten()
    is_known = is_fg | is_bg
    is_unknown = ~is_known

    # diagonal matrix to constrain known alpha values
    d = is_known.astype(np.float64)
    D = scipy.sparse.diags(d)

    # combine constraints and graph laplacian
    A = constraint_factor*D + L
    # constrained values of known alpha values
    b = constraint_factor*is_fg.astype(np.float64)

    return A, b

def soft_matting(image, trimap):
    # configure paths here
#     image_path  = "cat_image.png"
#     trimap_path = "cat_trimap.png"
#     alpha_path  = "cat_alpha.png"
#     cutout_path = "cat_cutout.png"

#     # load and convert to [0, 1] range
#     image  = np.array(Image.open( image_path).convert("RGB"))/255.0
#     trimap = np.array(Image.open(trimap_path).convert(  "L"))/255.0

    # make matting laplacian
    i,j,v = closed_form_laplacian(image)
    h,w = trimap.shape
    L = scipy.sparse.csr_matrix((v, (i, j)), shape=(w*h, w*h))

    # build linear system
    A, b = make_system(L, trimap)

    # solve sparse linear system
    print("solving linear system...")
    alpha = scipy.sparse.linalg.spsolve(A, b).reshape(h, w)

    # stack rgb and alpha
#     cutout = np.concatenate([image, alpha[:, :, np.newaxis]], axis=2)

    # clip and convert to uint8 for PIL
#     cutout = np.clip(cutout*255, 0, 255).astype(np.uint8)
#     alpha  = np.clip( alpha*255, 0, 255).astype(np.uint8)

#     # save and show
#     Image.fromarray(alpha ).save( alpha_path)
#     Image.fromarray(cutout).save(cutout_path)
#     Image.fromarray(alpha ).show()
#     Image.fromarray(cutout).show()
#     plt.imshow(alpha)
    return alpha


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
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
#     t = Guidedfilter(gray,et,config);
    img_norm = img.copy()
    img_norm = np.float64(img_norm)/255 
    t = soft_matting(img_norm, et)
    return t;

def recover(im,t,A,config):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,config['tx']);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res
    


# In[45]:


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

if __name__ == 'main':
    img_path = '../../data/aerial.png'
    iimg = cv2.imread(img_path)
    plt.imshow(dehaze(iimg, config))

