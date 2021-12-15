import numpy as np
from scipy.ndimage import convolve
from skimage.draw import disk
from skimage.io import imsave
import cv2
import matplotlib.pyplot as plt
import sys
import os

def DiskKernel(dim):
	if (dim < 3):
	    return None
	else:
	    kernelwidth = dim
	    kernel = np.zeros((kernelwidth+1, kernelwidth+1), dtype=np.float32)

	    circleCenterCoord = dim // 2
	    circleRadius = circleCenterCoord + 1
	    rr, cc = disk((circleCenterCoord, circleCenterCoord), circleRadius)
	    kernel[rr,cc] = 1

	    if(dim == 3 or dim == 5):
		    kernel[0,0] = 0
		    kernel[0,kernelwidth-1] = 0
		    kernel[kernelwidth-1,0] = 0
		    kernel[kernelwidth-1, kernelwidth-1] = 0

	    normalizationFactor = np.count_nonzero(kernel)
	    kernel = kernel / normalizationFactor
	    kernel = np.pad(kernel, ((circleCenterCoord//4, circleCenterCoord//4),), 'constant')
	    kernel = cv2.GaussianBlur(kernel, (0,0), sigmaX=0.5, sigmaY=0.5)
	    return kernel


class BlurImage:
	def __init__(self, disparities_smoothed, og, z, m, t, numLayers=10):
		self.OG_img = np.dstack((og, og, og, np.ones_like(og)))
		self.focusDist = z
		self.focusDisp = np.max(disparities_smoothed) + t # will see later -- sets disp = 0 to focus
		# maps greater values to "front", which is what we want
		self.dispMap = np.max(disparities_smoothed) - disparities_smoothed
		self.m = m
		self.cutoff = 0.05
		self.numLayers = numLayers

	def renderBlur(self):
		I_n = np.zeros_like(self.OG_img)
		I_d = np.zeros_like(self.dispMap)

		d_steps = np.linspace(np.min(self.dispMap)-0.3, np.max(self.dispMap), self.numLayers)
		step_size = d_steps[1] - d_steps[0]
		m_inv = 1.0 / (0.25 * step_size)

		for j in range(1, len(d_steps)):
			d_j = d_steps[j]
			d_jl = d_steps[j-1]

			print("Trying to blur for disp in between {} and {}".format(d_j, d_jl))

			Alpha_tent = 1.0 + m_inv * np.minimum(self.dispMap - d_jl, d_j - self.dispMap)
			Alpha = np.clip(Alpha_tent, 0, 1)

			Vals_ig = np.zeros_like(self.OG_img)
			Vals_ig[:,:,0] = self.OG_img[:,:,0] * Alpha
			Vals_ig[:,:,1] = self.OG_img[:,:,1] * Alpha
			Vals_ig[:,:,2] = self.OG_img[:,:,2] * Alpha
			Vals_ig[:,:,3] = self.OG_img[:,:,3] * Alpha
			blur_radius = self.m * np.clip((0.33 * self.focusDist + 0.17), 1, 3.5)
			blur_radius = blur_radius * np.maximum(0.0, (abs(d_j - self.focusDisp) - self.cutoff))
			blur_radius = int(blur_radius)
			blur_kernel = DiskKernel(blur_radius)
			print(blur_radius)
			if blur_kernel is not None:
				Vals_ig[:,:,0] = convolve(Vals_ig[:,:,0], blur_kernel, mode='constant', cval=0.0)
				Vals_ig[:,:,1] = convolve(Vals_ig[:,:,1], blur_kernel, mode='constant', cval=0.0)
				Vals_ig[:,:,2] = convolve(Vals_ig[:,:,2], blur_kernel, mode='constant', cval=0.0)
				Vals_ig[:,:,3] = convolve(Vals_ig[:,:,3], blur_kernel, mode='constant', cval=0.0)
				Alpha = convolve(Alpha, blur_kernel, mode='constant', cval=0.0)

			I_n[:,:,0] = I_n[:,:,0] * (1.0 - Alpha)
			I_n[:,:,1] = I_n[:,:,1] * (1.0 - Alpha)
			I_n[:,:,2] = I_n[:,:,2] * (1.0 - Alpha)
			I_n[:,:,3] = I_n[:,:,3] * (1.0 - Alpha)
			I_n = I_n + Vals_ig
			I_d = I_d * (1.0 - Alpha) + Alpha

		I_final = np.zeros_like(self.OG_img)
		valid_idx = np.where(I_d != 0.0)
		I_final[:,:,0][valid_idx] = I_n[:,:,0][valid_idx] / I_d[valid_idx]
		I_final[:,:,1][valid_idx] = I_n[:,:,1][valid_idx] / I_d[valid_idx]
		I_final[:,:,2][valid_idx] = I_n[:,:,2][valid_idx] / I_d[valid_idx]
		I_final[:,:,3][valid_idx] = I_n[:,:,3][valid_idx] / I_d[valid_idx]

		return I_final

if __name__ == "__main__":
	fname_load = "{}_smoothed.npz".format(sys.argv[1])
	fname = os.path.join(os.path.realpath("../"), "data", fname_load)
	X_dict = np.load(fname)
	blurrer = BlurImage(X_dict["disp"], X_dict["I0"], 0.01, 25, 1.2, numLayers=20)
	blurred = blurrer.renderBlur()
	fig, axs = plt.subplots(1, 2)
	axs[0].imshow(blurred)
	axs[1].imshow(blurrer.OG_img)

	final_img = "{}_blurred.png".format(sys.argv[1])
	imsave(final_img, np.clip(blurred*255, 0, 1))
	plt.show()
