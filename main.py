import warnings
import cv2
import os
from src import *

# the directory that containes the watermarked images.
# please speficy the relative path from 'wdir'
wdir = './'
watermkarked = 'watermarked_images'

forldername = os.path.join(wdir, watermkarked)
resultdir = os.path.join(wdir, watermkarked + '_removed')

if not os.path.exists(forldername):
    warnings.warn("{} does not exist.".format(forldername), UserWarning)

if not os.path.exists(resultdir):
    os.mkdir(resultdir)

gx, gy, gxlist, gylist = estimate_watermark(forldername)

print("median gx shape: {}".format(gx.shape))

# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape, dtype=np.float32)[:, :, 0])
cropped_gx, cropped_gy = crop_watermark(gx, gy, threshold=0.6)
W_m = poisson_reconstruct(cropped_gx, cropped_gy, h=0.3, num_iters=100)
# W_m = poisson_reconstruct2(cropped_gx, cropped_gy)
plt.imshow(W_m)

# random photo
img = cv2.imread(os.path.join(forldername, os.listdir(forldername)[0]))
plt.imshow(img)

im, start, end = watermark_detector(
    img, cropped_gx, cropped_gy, thresh_low=230/255, thresh_high=250/255)

###### PLEASE CHECK and ADJUST THRESHOLD ######
plt.imshow(im)
plt.show()

# We are done with watermark estimation
# W_m is the cropped watermark
num_images = len(gxlist)

J, img_paths = get_cropped_images(
    forldername, num_images, start, end, cropped_gx.shape)

# get a random subset of J
# idx = [389, 144, 147, 468, 423, 92, 3, 354, 196, 53, 470, 445, 314, 349, 105, 366, 56, 168, 351, 15, 465, 368, 90, 96, 202, 54, 295, 137, 17, 79, 214, 413, 454, 305, 187,
#        4, 458, 330, 290, 73, 220, 118, 125, 180, 247, 243, 257, 194, 117, 320, 104, 252, 87, 95, 228, 324, 271, 398, 334, 148, 425, 190, 78, 151, 34, 310, 122, 376, 102, 260]
# idx = idx[:25]
# Wm = (255*PlotImage(W_m))

print("W_m min: {}, W_m max: {}".format(W_m.min(), W_m.max()))

Wm = W_m - W_m.min()
plt.imshow(Wm)
plt.show()

# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(
    J, Wm, num_images=num_images, threshold=170/255, invert=False)
alph = np.stack([alph_est, alph_est, alph_est], axis=2).astype(np.float32)
print("alph_est shape: {}".format(alph_est.shape))
plt.imshow(PlotImage(alph))
plt.show()

C, est_Ik = estimate_blend_factor(J, Wm, alph)

alpha = alph.copy()
for i in range(3):
    alpha[:, :, i] = C[i] * alpha[:, :, i]

print("aplpha shape: {}".format(alpha.shape))
print("alpha min: {}, alpha max: {}".format(alpha.min(), alpha.max()))

Wm = Wm + alpha*est_Ik

W = Wm.copy()
for i in range(3):
    W[:, :, i] /= C[i]

Jt = J[: 25]
# now we have the values of alpha, Wm, J
# Solve for all images
Wk, Ik, W, alpha1 = solve_images(
    J=Jt, W_m=W_m, alpha=alpha, W_init=W, gamma=1, beta=1, lambda_w=0.005, lambda_i=1, lambda_a=0.01, iters=1)

W = PlotImage(W)

cv2.imshow("alpha1", PlotImage(alpha1))
cv2.waitKey(0)

cv2.imshow("W", PlotImage(W))
cv2.waitKey(0)

cv2.imshow("Wk", PlotImage(Wk[0]))
cv2.waitKey(0)

cv2.imshow("Ik", PlotImage(Ik[0]))
cv2.waitKey(0)

I = PlotImage((J - alpha1 * W) / (1 - alpha1))
II = PlotImage((J - alpha1 * Wk) / (1 - alpha1))

cv2.imshow("I[0]", I[0])
cv2.waitKey(0)

cv2.imshow("II[0]", II[0])
cv2.waitKey(0)


for i in range(num_images):
    filename = os.listdir(forldername)[i]
    im = cv2.imread(os.path.join(forldername, filename))
    im[start[0]:(start[0] + end[0]), start[1]:(start[1] + end[1]),
       :] = (PlotImage(Ik[i])*255).astype(np.uint8)
    cv2.imwrite(os.path.join(resultdir, filename), im)
    # W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
    # ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)
