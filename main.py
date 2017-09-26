from src import *

gx, gy, gxlist, gylist = estimate_watermark('images/fotolia_processed')
est, loss = poisson_reconstruct(gx, gy)
cropped_gx, cropped_gy = crop_watermark(gx, gy)

plt.imshow(PlotImage(est))

est2, _ = poisson_reconstruct(cropped_gx, cropped_gy)
plt.figure()
plt.imshow(PlotImage(est2))
plt.show()
