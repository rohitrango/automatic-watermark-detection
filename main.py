from src import *

gx, gy, gxlist, gylist = estimate_watermark('images/fotolia_processed')

est, loss = poisson_reconstruct(gx, gy)
cropped_gx, cropped_gy = crop_watermark(gx, gy)
est2, _ = poisson_reconstruct(cropped_gx, cropped_gy)

# random photo
img = cv2.imread('images/fotolia/fotolia_137840645.jpg')
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)

plt.imshow(im)
plt.show()
