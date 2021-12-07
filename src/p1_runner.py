import numpy as np
import cv2
from skimage.io import imread, imsave
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import consts
import sys
import os
import glob

def makePath(prefix, left=True):
	if (left):
		prefix += "_left.pgm"
	else:
		prefix += "_right.pgm"

	path = os.path.join(os.path.realpath("../"), "data", prefix)

	return path

def makeFolder(folder):
	path = os.path.join(os.path.realpath("../"), "data", folder)
	return path

def avgList(lsFiles, mat_avg):
	numFiles = len(lsFiles)
	for fname in lsFiles:
		mat_img = np.transpose(np.array(imread(fname), dtype=np.uint16))
		mat_img = mat_img.astype(np.float64)
		mat_img = mat_img / consts.MAX_VAL
		mat_avg += mat_img

	mat_avg /= numFiles
	return mat_avg

def upsample(h, w, mat_img):
	"""
		Linearly Upsampling mat_img
		Look at slides: https://www.cs.toronto.edu/~guerzhoy/320/lec/upsampling.pdf
	"""
	y = np.arange(h)
	x = np.arange(w)

	f = interp2d(x, y, mat_img, kind='linear')

	ynew = np.arange(h)
	xnew = np.arange(0, w, 0.5)
	znew = f(xnew, ynew)

	return znew

def normalize_img(mat_img, r, eps, denoised):
	"""
		First, we locally normalize each image by subtracting the local mean and
		dividing by local standard deviation

		For each 8x8 tile
		I' = (I - box(I, r)) / (eps^2 + box((I - box(I, r))^2, r))
	"""
	if (denoised):
		mat_img_denoised = cv2.GaussianBlur(mat_img, (0,0), sigmaX=consts.SIGMA_X,sigmaY=consts.SIGMA_Y)
	else:
		mat_img_denoised = mat_img
	mat_box = cv2.boxFilter(mat_img_denoised, -1, (r,r))
	mat_diff = mat_img_denoised - mat_box
	mat_diff_2 = np.square(mat_diff)
	mat_norm = mat_diff / np.sqrt(eps * eps + cv2.boxFilter(mat_diff_2, -1, (r,r)))

	return mat_norm

def ssd_tiles(img1, img2):
	return np.einsum('ijkl->ij', (img1 - img2)**2)

class Burst:
	"""
		Class that takes in a folder and assumes images are labeled 1-10 or something
	"""
	def __init__(self, folder):
		self.FolderName = folder

		print("Listing img names....")
		folderPath = makeFolder(self.FolderName)
		strGlobLeft = "{}/*_left.pgm".format(folderPath)
		strGlobRight = "{}/*_right.pgm".format(folderPath)

		leftFiles = glob.glob(strGlobLeft)
		rightFiles = glob.glob(strGlobRight)
		numFiles = len(leftFiles)

		one_img = np.transpose(np.array(imread(leftFiles[0])))
		avgLeft = np.zeros_like(one_img, dtype=np.float64)
		avgRight = np.zeros_like(one_img, dtype=np.float64)

		print("Averaging images....")
		avgLeft = avgList(leftFiles, avgLeft)
		avgRight = avgList(rightFiles, avgRight)

		self.Left = avgLeft
		self.Right = avgRight

	def getLeft(self):
		return self.Left

	def getRight(self):
		return self.Right

class Image:
	def __init__(self, burst, left=None, right=None, prefix=None, denoise=True):
		if (burst):
			# came from averaging a burst
			self.LeftImage = left
			self.RightImage = right
		else:
			self.prefix = prefix;

			print("Reading in imgs....")
			left_img = np.transpose(np.array(imread(makePath(prefix, True)), dtype=np.uint16))
			right_img = np.transpose(np.array(imread(makePath(prefix, False)), dtype=np.uint16))

			left_img = left_img.astype(np.float64)
			right_img = right_img.astype(np.float64)

			print("Normalizing imgs to 0-1....")
			self.RightImage = right_img / consts.MAX_VAL
			self.LeftImage = left_img / consts.MAX_VAL


		self.H = (self.RightImage).shape[0]
		self.W = (self.RightImage).shape[1]
		print("Upsampling imgs....")

		self.I0 = np.transpose(upsample(self.H, self.W, self.LeftImage))
		self.I1 = np.transpose(upsample(self.H, self.W, self.RightImage))

		self.H = (self.I0).shape[0]
		self.W = (self.I0).shape[1]
		self.Denoise = denoise

		print("Subtracting local means and dividing by variance imgs....")
		self.I0_norm = normalize_img(self.I0, consts.BOX_RADIUS, consts.EPS, denoised=denoise)
		self.I1_norm = normalize_img(self.I1, consts.BOX_RADIUS, consts.EPS, denoised=denoise)

	@classmethod
	def from_burst(cls, folderName):
		burstNew = Burst(folderName)
		burstLeft = burstNew.getLeft()
		burstRight = burstNew.getRight()
		return cls(True, left=burstLeft, right=burstRight, denoise=True)


	def block_disparities(self, reverse):
		"""
			Computes argmin of offsets for every non-overlapping 8x8 tile in I_0
			with respect to I_1
		"""
		if (reverse):
			REF_IMG = self.I1_norm	# the one to match against
			CMP_IMG = self.I0_norm  # the one we search
		else:
			REF_IMG = self.I0_norm	# the one to match against
			CMP_IMG = self.I1_norm  # the one we search

		# First we create a set of non-overlapping 8x8 tiles for I_0
		h = REF_IMG.shape[0]
		w = REF_IMG.shape[1]
		numRows = h // consts.TILE_H
		numCols_sep = w // consts.TILE_W # num cols for non-overlapping blocks
		numCols_overlap = w - consts.TILE_W + 1

		# Create number of shapes and strides
		print("Creating strides...")
		sz = (REF_IMG).itemsize
		shape_sep = np.array([numRows, numCols_sep, consts.TILE_H, consts.TILE_W], dtype="int")
		shape_overlap = np.array([numRows, numCols_overlap, consts.TILE_H, consts.TILE_W], dtype="int")

		strides_sep = sz * np.array([w * consts.TILE_H, consts.TILE_W, w, 1], dtype="int")
		strides_overlap = sz * np.array([w * consts.TILE_H, 1, self.W, 1], dtype="int")

		# now we create blocks of nonoverlapping blocks for ref and overlapping ones for cmp
		print("Creating strided views...")
		REF_BLOCKS = np.lib.stride_tricks.as_strided(REF_IMG, shape=shape_sep, strides=strides_sep)
		CMP_BLOCKS = np.lib.stride_tricks.as_strided(CMP_IMG, shape=shape_overlap, strides=strides_overlap)

		# now we create an array of argmin results for offsets
		# we say -4 since it's not an offset we are considering
		print("Initializing min_offset and min_ssds...")
		SSDS = np.full((numRows, numCols_sep, 9), np.inf)

		for d in range(-4, 5): # brute force search
			print("d = {}".format(d))
			if (d < 0):
				# if offset is less than 0, we don't consider the first column
				REF_VIEW = REF_BLOCKS[:,1:]
				d_new = consts.TILE_W + d
				CMP_VIEW = CMP_BLOCKS[:,d_new::consts.STRIDE_LENGTH]

			elif (d > 0):
				# if offset is more than 0, we don't consider the last column
				REF_VIEW = REF_BLOCKS[:,:-1]
				CMP_VIEW = CMP_BLOCKS[:,d::consts.STRIDE_LENGTH]
			else:
				REF_VIEW = REF_BLOCKS
				CMP_VIEW = CMP_BLOCKS[:,0::consts.STRIDE_LENGTH]

			# get the ssd
			d_ssd = ssd_tiles(REF_VIEW, CMP_VIEW)

			# put data into SSDS
			ssd_ind = d + 4

			if (d < 0):
				SSDS[:,1:,ssd_ind] = d_ssd
			elif (d > 0):
				SSDS[:,:-1,ssd_ind] = d_ssd
			else:
				SSDS[:,:,ssd_ind] = d_ssd

		return SSDS

	def subpixel_estimates(self, SSDS):
		# Now we try to extract subpixel estimates for each
		min_args = np.argmin((SSDS)[:,:,1:-1], axis=2)
		min_args = min_args - 3

		min_subpixel = np.zeros_like(min_args, dtype="double")
		confidence = np.zeros_like(min_args, dtype="double")

		# I am lazy, so we're gonna do a brute force for loop for fitting quadratic
		for i in range((SSDS).shape[0]):
			for j in range(1,(SSDS).shape[1]-1):
				int_x = min_args[i, j]
				x_2 = float(int_x)
				x_1 = x_2 - 1.0
				x_3 = x_2 + 1.0
				ind_x2 = int_x + 3
				y_2 = SSDS[i, j, ind_x2]
				y_1 = SSDS[i, j, ind_x2-1]
				y_3 = SSDS[i, j, ind_x2+1]

				x = np.array([x_1, x_2, x_3])
				y = np.array([y_1, y_2, y_3])

				a, b, c = np.polyfit(x, y, 2)

				if (abs(a) < consts.EPS_SMALL):
					#print("small a")
					if (b > 0):
						subpixel_est = (x_1+x_2) * 0.5
					else:
						subpixel_est = (x_3+x_2)*0.5
				else:
					x_min = (-0.5 * b) / a
					subpixel_est = np.clip(x_min, x_1,x_3)

				subpixel_est = np.clip(subpixel_est,-3,3)
				min_subpixel[i, j] = subpixel_est
				if (np.isnan(subpixel_est)):
					print('i:{}, j:{}'.format(i, j))
					print('small a: {}'.format(a))
					print("y: {}".format(y))

				confidence[i, j] = np.exp((np.log(abs(a)) / consts.SIGMA_A) - (c / (consts.SIGMA_c**2)))
				if (np.isnan(confidence[i, j])):
					confidence[i,j] = 0.0

		return min_args, min_subpixel, confidence

	def disparity_wrapper(self, reverse=False):
		ssds = self.block_disparities(reverse)
		rough, sub, conf = self.subpixel_estimates(ssds)
		self.coarse = rough.T
		self.subpix = sub.T
		self.conf = (conf -  np.min(conf)) / (np.max(conf) - np.min(conf))
		self.conf = self.conf.T

	def plot_og_imgs(self):
		fig, axs = plt.subplots(1, 2)
		axs[0].imshow(self.LeftImage, cmap="gray")
		axs[1].imshow(self.RightImage, cmap="gray")

	def plot_upsampled(self):
		fig, axs = plt.subplots(1, 3)
		axs[0].imshow(self.I0, cmap="gray")
		axs[1].imshow(self.I1, cmap="gray")
		axs[2].imshow(np.square(self.I0 - self.I1), cmap="jet")
		plt.suptitle("Upsampled")

	def plot_normalized(self):
		fig, axs = plt.subplots()
		#axs[0].imshow(self.I0_norm.T, cmap="gray")
		#axs[1].imshow(self.I1_norm.T, cmap="gray")
		axs.imshow(np.square((self.I0_norm).T - (self.I1_norm).T), cmap="jet")
		plt.title("Normalized")

	def plot_diffs(self):
		self.disparity_wrapper()
		coarse_offsets = self.coarse
		subpix_offsets = self.subpix
		confidences = self.conf

		fig1, axs1 = plt.subplots()
		im1 = axs1.imshow(coarse_offsets, cmap="jet", interpolation=None)
		axs1.set_title("Coarse")
		fig1.colorbar(im1)
		fig2, axs2 = plt.subplots()

		im2 = axs2.imshow(subpix_offsets, cmap="jet", interpolation=None)
		axs2.set_title("Subpix")
		fig2.colorbar(im2)

		fig3, axs3 = plt.subplots()
		im3 = axs3.imshow(confidences, cmap="gray", interpolation=None)
		axs3.set_title("Confidence")
		fig3.colorbar(im3)


if __name__ == "__main__":
	# first set of trials
	"""
	trial_denoised = Image(sys.argv[1], True, True)
	#trial_denoised.plot_og_imgs()
	#trial_denoised.plot_normalized()
	print("Getting coarse diffs from normalized, denoised and transposed...")
	trial_denoised.plot_diffs(True)

	trial_noisy = Image(sys.argv[1], False, True)
	#trial_noisy.plot_og_imgs()
	#trial_noisy.plot_normalized()
	print("Getting coarse diffs from normalized, noisy and transposed...")
	trial_noisy.plot_diffs(True)
	plt.show()
	"""

	# trial with image bursts
	trial1_burst = Image.from_burst(sys.argv[1])
	trial1_burst.plot_og_imgs()
	trial1_burst.plot_normalized()
	print("Getting coarse diffs from normalized, denoised and transposed...")
	trial1_burst.plot_diffs()
	plt.show()
