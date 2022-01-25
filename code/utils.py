import numpy as np

def divide_blocks(I, b_size = 16):
    """
    I is the image function
    """
    sizeX = I.shape[1]
    sizeY = I.shape[0]
    
    list_roi = []
    
    for i in range(0,int(sizeY/b_size)):
        for j in range(0, int(sizeX/b_size)):
            roi = I[int(i*b_size):int(i*b_size + b_size),
                    int(j*b_size):int(j*b_size + b_size):,]
            list_roi.append(roi)
    array_roi = np.array(list_roi)
    return array_roi