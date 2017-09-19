import sys, os
import cv2
import numpy as np
import warnings
from matplotlib import pyplot as plt

# Variables
KERNEL_SIZE = 3

def estimate_watermark(foldername):
	'''
	Given a folder, estimate the watermark (grad(W) = median(grad(J)))
	'''
	if not os.path.exists(foldername):
		warnings.warn("Folder does not exist.", UserWarning)
		return None

	images = []
	for r, dirs, files in os.walk(foldername):
		# Get all the images
		for file in files:
			img = cv2.imread(os.sep.join([r, file]))
			if img is not None:
				images.append(img)
			else:
				print("%s not found."%(file))

		# Compute gradients
		print("Computing gradients.")
		gradx = map(lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE), images)
		grady = map(lambda x: cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE), images)

		# Compute median of grads
		print("Computing median gradients.")
		Wm_x = np.median(np.array(gradx), axis=0) 					
		Wm_y = np.median(np.array(grady), axis=0)

		# print(Wm_x.shape)
		# print(Wm_y.shape)

		# for i in xrange(3):
		# 	plt.figure()
		# 	plt.imshow(Wm_x[:,:,i])
		# 	plt.title("Wm_x channel %d"%(i))

		# 	plt.figure()
		# 	plt.imshow(Wm_y[:,:,i])
		# 	plt.title("Wm_y channel %d"%(i))
		grad_mod = np.sqrt(np.square(Wm_x) + np.square(Wm_y))
		for i in xrange(3):
			plt.figure()
			plt.imshow(grad_mod[:,:,i])

		plt.show()
		return (Wm_x, Wm_y)


if __name__ == "__main__":
	x,y = estimate_watermark("images/fotolia_processed")