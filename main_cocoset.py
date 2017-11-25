'''
This main file is for the Microsoft Coco dataset
'''
from src import *

IMAGE_FOLDER = "/media/rohitrango/2EC8DBB2C8DB7715/"
IMG_LOC = "coco_dataset"
IMG_PROCESSED_LOC = "coco_dataset_processed"

def get_alpha_matte(watermark, threshold=128):
	w = np.average(watermark, axis=2)
	_, w = cv2.threshold(w, threshold, 255, cv2.THRESH_BINARY_INV)
	return PlotImage(w)

def P(img,e=None):
    if e is None:
        plt.imshow(PlotImage(img)); plt.show()
    else:
        plt.imshow(PlotImage(img),'gray'); plt.show()

def bgr2rgb(img):
	return img[:,:,[2, 1, 0]]
'''
Ground Truth values
alpha -> coco_dataset/alpha.png
copyright -> coco_dataset/copyright.png
c = .45

Experiments: Threshold for estimating initial alpha -> 153, and then subtract 1 from alpha
'''
if __name__ == "__main__":
	# watermark = cv2.imread('coco_dataset/watermark.png')
	# alpha = get_alpha_matte(watermark)
	foldername = os.path.join(IMAGE_FOLDER, IMG_PROCESSED_LOC)
	gx, gy, gxlist, gylist = estimate_watermark(foldername)

	# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
	cropped_gx, cropped_gy = crop_watermark(gx, gy)
	W_m = poisson_reconstruct(cropped_gx, cropped_gy, num_iters=5000)

	# random photo
	img = cv2.imread(os.path.join(foldername, '000000051008.jpg'))
	im, start, end = watermark_detector(img, cropped_gx, cropped_gy)
	num_images = len(gxlist)

	J, img_paths = get_cropped_images(foldername, num_images, start, end, cropped_gx.shape)
	# get a random subset of J
	idx = [389, 144, 147, 468, 423, 92, 3, 354, 196, 53, 470, 445, 314, 349, 105, 366, 56, 168, 351, 15, 465, 368, 90, 96, 202, 54, 295, 137, 17, 79, 214, 413, 454, 305, 187, 4, 458, 330, 290, 73, 220, 118, 125, 180, 247, 243, 257, 194, 117, 320, 104, 252, 87, 95, 228, 324, 271, 398, 334, 148, 425, 190, 78, 151, 34, 310, 122, 376, 102, 260]
	idx = idx[:25]
	# Wm = (255*PlotImage(W_m))
	Wm = W_m - W_m.min()

	# get threshold of W_m for alpha matte estimate
	alph_est = estimate_normalized_alpha(J, Wm, num_images=15, threshold=125, invert=False, adaptive=False)
	alph = np.stack([alph_est, alph_est, alph_est], axis=2)
	C, est_Ik = estimate_blend_factor(J, Wm, alph)

	alpha = alph.copy()
	for i in xrange(3):
		alpha[:,:,i] = C[i]*alpha[:,:,i]

	# Wm = Wm + alpha*est_Ik

	W = Wm.copy()
	for i in xrange(3):
		W[:,:,i]/=C[i]

	Jt = J[idx]
	# now we have the values of alpha, Wm, J
	# Solve for all images
	Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
	# W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
	# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)  


	