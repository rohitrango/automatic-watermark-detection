from src import *

gx, gy, gxlist, gylist = estimate_watermark('images/fotolia_processed')

# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
cropped_gx, cropped_gy = crop_watermark(gx, gy)

# random photo
img = cv2.imread('images/fotolia_processed/fotolia_137840668.jpg')
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)

# # Save result of watermark detection
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(im)
# plt.savefig('images/results/watermark_detect_result.png')
# We are done with watermark estimation

# W_m is the cropped watermark
W_m = poisson_reconstruct(cropped_gx, cropped_gy)

# Get the number of images in the folder
num_images = len(gxlist)

J, img_paths = get_cropped_images(
    'images/fotolia_processed', num_images, start, end, cropped_gx.shape)

# get a random subset of J

# get threshold of W_m for alpha matte estimate
Wm = W_m - W_m.min()

# estimate alpha matte
alph_est = estimate_normalized_alpha(J, Wm, num_images=len(J))
alph = np.stack([alph_est, alph_est, alph_est], axis=2)
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
Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
