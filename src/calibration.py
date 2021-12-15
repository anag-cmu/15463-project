import numpy as np
import matplotlib.pyplot as plt
import os
import sys

"""
Script that loads in disparity data for all calibration and fits it to line
Did not let me control for different focal distances, unfortunately
"""
def load_data(npz_file):
	print("Loading {}....".format(npz_file))
	fname = os.path.join(os.path.realpath("../"), "data", npz_file)
	with np.load(fname) as Y:
		I0, I1, pixelDisp, coarseDisp, pixelConf, coarseConf = [Y[i] for i in
		('I0', 'I1', 'pixel_disp', 'disp', 'pixel_conf', 'conf')]

	return pixelDisp

def save_data(slopes, intercepts):
	fname = "calibration.npz"
	ext_out = {"S":slopes, "I":intercepts}
	np.savez(os.path.join(os.path.realpath("../"), "data", fname), **ext_out)

def fitLine(img_channels, target_dists, save=True):
	fig, axs = plt.subplots(4, 6)
	slopes = np.zeros((img_channels.shape[0], img_channels.shape[1]), dtype=np.float64)
	interps = np.zeros((img_channels.shape[0], img_channels.shape[1]), dtype=np.float64)

	for i in range(img_channels.shape[0]):
		for j in range(img_channels.shape[1]):
			x = target_dists
			y = img_channels[i][j]
			m, b = np.polyfit(x, y, 1)

			slopes[i][j] = m
			interps[i][j] = b

			if (((i % 400) == 0) and ((j % 400) == 0)):
				print(i, j)
				ind_i = i // 400
				ind_j = j // 400
				ynew = m * x + b
				axs[ind_i][ind_j].scatter(x, y, c="red")
				axs[ind_i][ind_j].plot(x, ynew, c="blue")
				axs[ind_i][ind_j].set_title("x={}, y={}".format(j, i))

	fig1, axs1 = plt.subplots()
	im1 = axs1.imshow(slopes)
	plt.title("Slopes")
	fig1.colorbar(im1)

	fig2, axs2 = plt.subplots()
	im2 = axs2.imshow(interps)
	plt.title("Intercepts")
	fig2.colorbar(im2)

	if (save):
		save_data(slopes, interps)
	return slopes, interps

def make_img_channels(fnames):
	NUM_DISTS = len(fnames)
	img_channels = []
	for i in range(NUM_DISTS):
		fname = "{}_data.npz".format(fnames[i])
		pixelDisp_i = load_data(fname)
		img_channels.append(pixelDisp_i)

	img_channels_np = np.dstack(img_channels)
	return img_channels_np

def init_wrapper():
	"""
		Called initially and only once to make calibration
	"""
	fnames = ["11_33_3.375", "11_35_5.75", "11_36_4.25", "11_39_1.75", "11_44_7.25", "11_47_6.875"]
	target_dists = np.array([3.375, 5.75, 4.25, 1.75, 7.25, 6.875])
	#fnames = ["11_33_3.375", "11_35_5.75", "11_36_4.25", "11_44_7.25", "11_47_6.875"]
	#target_dists = np.array([3.375, 5.75, 4.25, 7.25, 6.875])
	target_dists = target_dists / 39.3701	# conversion to meters
	inv_dists = 1.0 / target_dists
	img_channels = make_img_channels(fnames)
	S, I = fitLine(img_channels, inv_dists)
	return S, I, img_channels

def correct_disps(fname, save=False):
	fname_aug = "{}_data.npz".format(fname)
	calib_fname = os.path.join(os.path.realpath("../"), "data", "calibration.npz")
	disps_fname = os.path.join(os.path.realpath("../"), "data", fname_aug)
	disps = load_data(disps_fname)

	with np.load(calib_fname) as Y:
		S, I = [Y[i] for i in ("S", "I")]

	H = disps.shape[0]
	W = disps.shape[1]
	S_center = S[(H // 2)][(W // 2)]
	I_center = I[(H // 2)][(W // 2)]

	disps_corrected = I_center + ((S_center * (disps - I)) / S)

	if (save):
		ext_out = {"D_corrected":disps_corrected}
		fname_out = "{}_disparities.npz"
		np.savez(os.path.join(os.path.realpath("../"), "data", fname), **ext_out)

	plt.figure()
	plt.imshow(disps)
	plt.title("{} Original Disparities".format(fname))
	plt.figure()
	plt.imshow(disps_corrected)
	plt.title("{} Corrected Disparities".format(fname))
	plt.figure()
	plt.imshow(S)
	plt.figure()
	plt.imshow(I)

	return disps_corrected

if __name__ == "__main__":
	#S, I, disps = init_wrapper()
	#plt.show()
	correct_disps(sys.argv[1])
	plt.show()
