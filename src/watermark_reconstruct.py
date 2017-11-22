import numpy as np
import cv2
import os
import scipy
from scipy.sparse import *
from scipy.sparse import linalg
from estimate_watermark import *
from closed_form_matting import *
from numpy import nan, isnan

def get_cropped_images(foldername, num_images, start, end, shape):
    '''
    This is the part where we get all the images, extract their parts, and then add it to our matrix
    '''
    images_cropped = np.zeros((num_images,) + shape)
    # get images
    # Store all the watermarked images
    # start, and end are already stored
    # just crop and store image
    image_paths = []
    _s, _e = start, end
    index = 0

    # Iterate over all images
    for r, dirs, files in os.walk(foldername):

        for file in files:
            _img = cv2.imread(os.sep.join([r, file]))
            if _img is not None:
                # estimate the watermark part
                image_paths.append(os.sep.join([r, file]))
                _img = _img[_s[0]:(_s[0]+_e[0]), _s[1]:(_s[1]+_e[1]), :]
                # add to list images
                images_cropped[index, :, :, :] = _img
                index+=1
            else:
                print("%s not found."%(file))

    return (images_cropped, image_paths)


# get sobel coordinates for y
def _get_ysobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
        (i-1, j, k, -2), (i-1, j-1, k, -1), (i-1, j+1, k, -1),
        (i+1, j, k,  2), (i+1, j-1, k,  1), (i+1, j+1, k,  1)
    ]

# get sobel coordinates for x
def _get_xsobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
        (i, j-1, k, -2), (i-1, j-1, k, -1), (i-1, j+1, k, -1),
        (i, j+1, k,  2), (i+1, j-1, k,  1), (i+1, j+1, k,  1)
    ]

# filter
def _filter_list_item(coord, shape):
    i, j, k, v = coord
    m, n, p = shape
    if i>=0 and i<m and j>=0 and j<n:
        return True

# Change to ravel index
# also filters the wrong guys
def _change_to_ravel_index(li, shape):
    li = filter(lambda x: _filter_list_item(x, shape), li)
    i, j, k, v = zip(*li)
    return zip(np.ravel_multi_index((i, j, k), shape), v)

# TODO: Consider wrap around of indices to remove the edge at the end of sobel
# get Sobel sparse matrix for Y
def get_ySobel_matrix(m, n, p):
    size = m*n*p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_ysobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    return coo_matrix((vals, (i, j)), shape=(size, size))


# get Sobel sparse matrix for X
def get_xSobel_matrix(m, n, p):
    size = m*n*p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_xsobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    return coo_matrix((vals, (i, j)), shape=(size, size))

# get estimated normalized alpha matte
def estimate_normalized_alpha(J, W_m, num_images=30):
    _Wm = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
    ret, thr = cv2.threshold(_Wm, 140, 255, cv2.THRESH_BINARY)
    thr = np.stack([thr, thr, thr], axis=2)

    num, m, n, p = J.shape
    alpha = np.zeros((num_images, m, n))
    iterpatch = 900

    print("Estimating normalized alpha using %d images."%(num_images))
    # for all images, calculate alpha
    for idx in xrange(num_images):
        # imgcopy = J[idx].copy()
        # for i in xrange(iterpatch):
        #     r = np.random.randint(10)
        #     x = np.random.randint(m-r)
        #     y = np.random.randint(n-r)
        #     imgcopy[x:x+r, y:y+r, :] = thr[x:x+r, y:y+r, :]
        imgcopy = thr
        alph = closed_form_matte(J[idx], imgcopy)
        alpha[idx] = alph

    alpha = np.median(alpha, axis=0)
    return alpha

# estimate the blend factor C
def estimate_blend_factor(J, W_m, alph, threshold=0.01*255):
    alpha_n = alph
    S = J.copy()
    num_images = S.shape[0]
    # for i in xrange(num_images):
    #     S[i] = PlotImage(S[i])
    # R = (S<=threshold).astype(np.float64)
    R = (S>-1).astype(np.float64)

    est_Ik = S.copy()
    # est_Ik[S>threshold] = nan
    est_Ik = np.nanmedian(est_Ik, axis=0)
    est_Ik[isnan(est_Ik)] = 0

    alpha_k = R.copy()
    beta_k = R*J
    for i in xrange(num_images):
        alpha_k[i] *= alpha_n*est_Ik
        beta_k[i] -= R[i]*W_m
    beta_k = np.abs(beta_k)
    # we have alpha and beta, solve for c's now
    c = []
    for i in range(3):
        c_i = np.sum(beta_k[:,:,:,i]*alpha_k[:,:,:,i])/np.sum(np.square(alpha_k[:,:,:,i]))
        c.append(c_i)
    return c

    # TODO: Remove the blend factor formulation with something more suitable
    # Like taking the edge details instead of sum of squares of intensity
    # c=0.1
    #     ...: while c<1:
    #     ...:     img = (S[61]-c*alph*est_Ik)
    #     ...:     sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, 3)
    #     ...:     sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, 3)
    #     ...:     edge = np.sqrt(sx**2 + sy**2)
    #     ...:     edge = img
    #     ...:     plt.subplot(3,3,int(c*10)); plt.imshow(PlotImage(edge));
    #     ...:     c+=0.1
    #     ...:     print(np.mean(np.square(edge)))
