import warnings
import cv2
import os
from src import *

# the directory that containes the wartermarked images.
# please speficy the relative path from 'wdir'
wdir = './'
wartermkarked = 'wartermarked_images'

# the image size (pixels) to be used during processing.
width = 500
height = 500

forldername = os.path.join(wdir, wartermkarked)
resizeddir = os.path.join(wdir, wartermkarked + '_resized')
resultdir = os.path.join(wdir, wartermkarked + '_removed')

if not os.path.exists(forldername):
    warnings.warn("{} does not exist.".format(forldername), UserWarning)

if not os.path.exists(resizeddir):
    os.mkdir(resizeddir)

if not os.path.exists(resultdir):
    os.mkdir(resultdir)

original_shape = []  # (width, height, channel)
filenames = []
for r, dirs, files in os.walk(forldername):
    # Get all the images
    for file in files:
        img = cv2.imread(os.sep.join([r, file]))
        if img is not None:
            original_shape.append(img.shape)
            filenames.append(file)
            print('original shape: {}'.format(img.shape))
            img = cv2.resize(img, (width, height))
            print('resized shape:  {}'.format(img.shape))
            cv2.imwrite(os.path.join(resizeddir, file), img)
        else:
            print("%s not found." % (file))

gx, gy, gxlist, gylist = estimate_watermark(resizeddir)

# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape, dtype=np.float32)[:, :, 0])
cropped_gx, cropped_gy = crop_watermark(gx, gy, threshold=0.6)
W_m = poisson_reconstruct(cropped_gx, cropped_gy)

# random photo
im, start, end = watermark_detector(
    img, cropped_gx, cropped_gy, thresh_low=200/255, thresh_high=220/255)

plt.imshow(im)
plt.show()


# We are done with watermark estimation
# W_m is the cropped watermark
num_images = len(gxlist)

J, img_paths = get_cropped_images(
    resizeddir, num_images, start, end, cropped_gx.shape)

# get a random subset of J
idx = [389, 144, 147, 468, 423, 92, 3, 354, 196, 53, 470, 445, 314, 349, 105, 366, 56, 168, 351, 15, 465, 368, 90, 96, 202, 54, 295, 137, 17, 79, 214, 413, 454, 305, 187,
       4, 458, 330, 290, 73, 220, 118, 125, 180, 247, 243, 257, 194, 117, 320, 104, 252, 87, 95, 228, 324, 271, 398, 334, 148, 425, 190, 78, 151, 34, 310, 122, 376, 102, 260]
idx = idx[:25]
Wm = (255*PlotImage(W_m))
Wm = W_m - W_m.min()

# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(J, Wm, num_images=num_images)
alph = np.stack([alph_est, alph_est, alph_est], axis=2).astype(np.float32)
C, est_Ik = estimate_blend_factor(J, Wm, alph)

alpha = alph.copy()
for i in range(3):
    alpha[:, :, i] = C[i]*alpha[:, :, i]

Wm = Wm + alpha*est_Ik

W = Wm.copy()
for i in range(3):
    W[:, :, i] /= C[i]

Jt = J[:25]
# now we have the values of alpha, Wm, J
# Solve for all images
Wk, Ik, W, alpha1 = solve_images(
    Jt, W_m, alpha, W, gamma=1, beta=0.005, lambda_i=0.5, lambda_a=0.01, lambda_w=0.005)

print(Ik.shape)
cv2.imshow("test", Ik[0])
cv2.waitKey(0)

# W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)
