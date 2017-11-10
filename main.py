from src import *

gx, gy, gxlist, gylist = estimate_watermark('images/fotolia_processed')

est, loss = poisson_reconstruct(gx, gy)
cropped_gx, cropped_gy = crop_watermark(gx, gy)
W_m, _ = poisson_reconstruct(cropped_gx, cropped_gy)

# random photo
img = cv2.imread('images/fotolia_processed/fotolia_137840645.jpg')
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)

# plt.imshow(im)
# plt.show()
# We are done with watermark estimation
# W_m is the cropped watermark
num_images = len(gxlist)

J, img_paths = get_cropped_images('images/fotolia_processed', num_images, start, end, cropped_gx.shape)

num_images, m, n, chan = J.shape
model = image_watermark_decompose_model(num_images, m, n, chan)
model2 = matte_update_model(num_images, m, n, chan)

# define the variables
# plt.imshow(PlotImage(est2))
# I = np.random.randn(num_images, m, n, chan)
# alpha = np.random.rand(m, n, chan)
# W_median = W_m.copy()
# # W = np.stack([W_m for _ in xrange(num_images)])			# list of W_k
# saver = tf.train.Saver()
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	for __ in xrange(10):

# 		# step 1 preprocess
# 		alpha_ = sess.run(model2['alpha'])
# 		# step 1
# 		for i in xrange(5):
# 			_, loss = sess.run([model['step'], model['loss']], feed_dict={
# 				model['J']: J,
# 				model['alpha']: alpha_,
# 				model['W_m']: W_m,
# 				model['W_median']: W_median,
# 			})
# 		print("Step 1: %f"%loss)

# 		# step 2
# 		W = sess.run(model['W'])
# 		W_median = np.median(W, axis=0)
# 		# print("Step 2")

# 		# step 3 preprocess
# 		I = sess.run(model['I'])
# 		# step 3
# 		for i in xrange(5):
# 			_, loss = sess.run([model2['step'], model2['loss']], feed_dict={
# 				model2['J']: J, 
# 				model2['W_m']: W_m, 
# 				model2['W_median']: W_median,
# 				model2['I']: I,
# 				model2['W']: W,
# 			})
# 		print("Step 3: %f"%(loss))
# 		print("---------------------------------------")

# 		plt.imshow(PlotImage(W_median))
# 		plt.show()