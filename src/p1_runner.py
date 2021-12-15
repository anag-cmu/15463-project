import numpy as np
import cv2
from skimage.io import imread, imsave
from skimage.draw import disk
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import multiprocess as mp
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

def denoise_img(mat_img):
	if (consts.DENOISE_MET == "OPEN"):
		kernel = np.ones((consts.SIGMA_X, consts.SIGMA_X), dtype=np.uint16)
		opening = cv2.morphologyEx(mat_img.astype(np.float32), cv2.MORPH_OPEN, kernel)
		return opening
	elif (consts.DENOISE_MET == "MEDIAN"):
		median = cv2.medianBlur(mat_img.astype(np.float32), consts.SIGMA_X)
		return median
	elif (consts.DENOISE_MET == "GAUSSIAN"):
		gaussian = cv2.GaussianBlur(mat_img, (0,0), sigmaX=consts.SIGMA_X,sigmaY=consts.SIGMA_Y)
		return gaussian
	elif (consts.DENOISE_MET == "BILATERAL"):
		bilat = cv2.bilateralFilter(mat_img.astype(np.float32),9,consts.SIGMA_X*5,consts.SIGMA_Y*5)
		return bilat
	else:
		boxFiltered = cv2.boxFilter(mat_img, -1, (consts.SIGMA_X, consts.SIGMA_X))
		return boxFiltered

def normalize_img(mat_img, r, eps, denoised):
	"""
		First, we locally normalize each image by subtracting the local mean and
		dividing by local standard deviation

		For each 8x8 tile
		I' = (I - box(I, r)) / (eps^2 + box((I - box(I, r))^2, r))
	"""
	if (denoised):
		mat_img_denoised = denoise_img(mat_img)
	else:
		mat_img_denoised = mat_img
	mat_box = cv2.boxFilter(mat_img_denoised, -1, (r,r))
	mat_diff = mat_img_denoised - mat_box
	mat_diff_2 = np.square(mat_diff)
	mat_norm = mat_diff / np.sqrt(eps * eps + cv2.boxFilter(mat_diff_2, -1, (r,r)))

	return mat_norm

def ssd_tiles(img1, img2):
	return np.einsum('ijkl->ij', (img1 - img2)**2)

def get_quadratic(int_x, SSDS, i, j):
	int_x = int_x - 3
	x_2 = float(int_x)
	x_1 = x_2 - 1.0
	x_3 = x_2 + 1.0
	ind_x2 = int_x + 4
	y_2 = SSDS[i, j, ind_x2]
	y_1 = SSDS[i, j, ind_x2-1]
	y_3 = SSDS[i, j, ind_x2+1]

	x = np.array([x_1, x_2, x_3])
	y = np.array([y_1, y_2, y_3])

	a, b, c = np.polyfit(x, y, 2)
	return a, b, c, x

def get_mins(a, b, c, x):
	if (abs(a) < consts.EPS_SMALL):
		#print("small a")
		if (b > 0):
			xmin = (x[0]+x[1]) * 0.5
		else:
			xmin = (x[2]+x[1])*0.5
		ymin = b * xmin + c
	else:
		xmin = (-0.5 * b) / a
		xmin = np.clip(xmin, x[0], x[2])
		ymin = a * (xmin * xmin) + b * xmin + c

	return xmin, ymin


def repeated_texture(d1, d2):
	"""
		d1 - value of Di at first minimum
		d2 - value of Di at second minimum
		Heuristic that helps update confidence of a tile based on presence of
		second minimum. We need the ratio between them to be kind of high
		for this to take effect.
	"""
	#print(d1 / d2)
	diff_y = np.maximum(consts.e_d, d1) - (d2 * consts.r_0)
	diff_y = diff_y * diff_y
	denom = d2 * (consts.r_1 - consts.r_0)
	denom = denom * denom
	ratio = diff_y / denom
	term = -1.0 * consts.w_r * np.clip(ratio, 0.0, 1.0)
	return np.exp(term)


def upsample_helper(f_x, f_y, motion, I1_interp, window_0):
	xnew_1 = np.array([f_x-1.0+motion, f_x+motion, f_x+1.0+motion])
	ynew_1 = np.array([f_y-1.0, f_y, f_y+1.0])
	window_1 = I1_interp(xnew_1, ynew_1)
	residual = np.sum(np.abs(window_0 - window_1))
	return residual


# HELP this is slow af but idc
def get_pixel_est(x, y, subpix, conf, I0_f, I1_f):
	#print(x, y)
	tile_i = y // consts.TILE_H
	tile_j = x // consts.TILE_W

	f_x = float(x)
	f_y = float(y)
	xnew_0 = np.array([f_x-1.0, f_x, f_x+1.0])
	ynew_0 = np.array([f_y-1.0, f_y, f_y+1.0])
	window_0 = I0_f(xnew_0, ynew_0)

	currMin = np.inf
	currPix = -5
	currConf = 1.0

	for x_offset in (-1, 1):
		# gets rid of any indexing errors
		tile_j_off = tile_j + x_offset

		if (tile_j_off < 0 or tile_j_off >= subpix.shape[1]):
			continue
		else:
			motion = subpix[tile_i, tile_j_off]
			res = upsample_helper(f_x, f_y, motion, I1_f, window_0)

		if (res < currMin):
			currMin = res
			currPix = motion
			currConf = conf[tile_i, tile_j_off]

	for y_offset in (-1, 1):
		# gets rid of any indexing errors
		tile_i_off = tile_i + y_offset

		if (tile_i_off < 0 or tile_i_off >= subpix.shape[0]):
			continue
		else:
			motion = subpix[tile_i_off, tile_j]
			res = upsample_helper(f_x, f_y, motion, I1_f, window_0)

		if (res < currMin):
			currMin = res
			currPix = motion
			currConf = conf[tile_i_off, tile_j]


	resFin = np.abs(I0_f(x,y) - I1_f(x+currPix,y)) - 0.2
	attenuation = np.exp(-1.0 * (np.maximum(0.0, resFin) / 0.5))
	currConf *= attenuation

	if ((x % 200 == 0) and (y % 200 == 0)):
		print(x, y, currPix, currConf)

	return currPix, currConf

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
		self.I0_norm = normalize_img(self.I0, consts.BOX_RADIUS, consts.EPS, denoised=True)
		self.I1_norm = normalize_img(self.I1, consts.BOX_RADIUS, consts.EPS, denoised=True)

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
		sorted_args = np.argsort((SSDS)[:,:,1:8], axis=2)
		min_args = sorted_args[:,:,0]
		second_args = sorted_args[:,:,1]

		min_subpixel = np.zeros_like(min_args, dtype="double")
		confidence = np.zeros_like(min_args, dtype="double")

		# I am lazy, so we're gonna do a brute force for loop for fitting quadratic
		for i in range((SSDS).shape[0]):
			for j in range(1,(SSDS).shape[1]-1):
				int_x1 = min_args[i, j]
				a1, b1, c1, x1 = get_quadratic(int_x1, SSDS, i, j)
				xmin1, ymin1 = get_mins(a1, b1, c1, x1)

				int_x2 = second_args[i, j]
				a2, b2, c2, x2 = get_quadratic(int_x2, SSDS, i, j)
				xmin2, ymin2 = get_mins(a2, b2, c2, x2)

				min_subpixel[i, j] = xmin1

				confidence[i, j] = np.exp((np.log(abs(a1)) / consts.SIGMA_A) - (c1 / (consts.SIGMA_c**2)))
				if (np.isnan(confidence[i, j])):
					confidence[i,j] = 0.0

				# APPLY THE HEURISTIC
				repeat_term = repeated_texture(ymin1, ymin2)
				confidence[i, j] = confidence[i, j] * repeat_term

		return min_args, min_subpixel, confidence



	def disparity_wrapper(self, reverse=False):
		ssds = self.block_disparities(reverse)
		rough, sub, conf = self.subpixel_estimates(ssds)
		self.coarse = rough
		self.subpix = sub
		self.conf = (conf - np.min(conf)) / (np.max(conf) - np.min(conf))


	def get_horizontal_grads(self):
		# First we create a set of non-overlapping 8x8 tiles for I_0
		if (self.Denoise):
			I0_denoised = denoise_img(self.I0)
		hor_grad = np.gradient(I0_denoised, axis=1) ** 2
		h = hor_grad.shape[0]
		w = hor_grad.shape[1]

		numRows = h // consts.TILE_H
		numCols_sep = w // consts.TILE_W # num cols for non-overlapping blocks
		numCols_overlap = w - consts.TILE_W + 1

		# Create number of shapes and strides
		print("Getting horizontal gradients...")
		sz = (hor_grad).itemsize
		shape = np.array([numRows, numCols_sep, consts.TILE_H, consts.TILE_W], dtype="int")
		strides = sz * np.array([w * consts.TILE_H, consts.TILE_W, w, 1], dtype="int")

		# now we create blocks of nonoverlapping blocks for ref and overlapping ones for cmp
		blocks = np.lib.stride_tricks.as_strided(hor_grad, shape=shape, strides=strides)
		mags = np.sqrt(np.einsum('ijkl->ij', blocks))

		# update confidences
		self.conf = self.conf * (np.exp(-1.0 * np.maximum(0.0, ((consts.w_v / mags) - consts.THRESH_PERCENT))))

	def get_outlier_tiles(self):
		"""
		For each offset, gets the neighborly flows
		"""
		outlier_weight = np.full_like(self.conf, np.inf)
		for i in range(self.subpix.shape[0]):
			for j in range(self.subpix.shape[1]):
				for x_off in (-1, 1):
					j_off = j + x_off
					if (j_off < 0 or j_off >= self.subpix.shape[1]):
						continue
					diff_neighbor = (self.subpix[i, j] - self.subpix[i, j_off]) ** 2
					diff_neighbor /= consts.VAR_FLOW
					outlier_weight[i, j] = np.minimum(outlier_weight[i, j], diff_neighbor)

				for y_off in (-1, 1):
					i_off = i + y_off
					if (i_off < 0 or i_off >= self.subpix.shape[0]):
						continue
					diff_neighbor = (self.subpix[i, j] - self.subpix[i_off, j]) ** 2
					diff_neighbor /= consts.VAR_FLOW
					outlier_weight[i, j] = np.minimum(outlier_weight[i, j], diff_neighbor)

		# update confidences
		self.conf = self.conf * np.exp(-1.0 * outlier_weight)

	# denoise confidences a little
	def smooth_conf(self, conf_map):
		kernel = np.ones((2, 2), dtype=np.uint16)
		return cv2.morphologyEx(conf_map.astype(np.float32), cv2.MORPH_OPEN, kernel)

	def disp_upsampler(self):
		y = np.arange(self.H)
		x = np.arange(self.W)
		print("interpolating I0_norm and I1_norm...")
		f0 = interp2d(x, y, self.I0_norm, kind='linear')
		f1 = interp2d(x, y, self.I1_norm, kind='linear')

		print("Nearest neighbor pixel disparities...")
		pixel_disp = np.zeros_like(self.I0)
		pixel_conf = np.zeros_like(self.I0)

		p = mp.Pool()
		arg_pairs = [(i,j) for i in range(0,self.H) for j in range(0,self.W)]

		def helper(i):
			#print(i[0], i[1])
			u, c = get_pixel_est(i[1], i[0], self.subpix, self.conf, f0, f1)
			return (i[0], i[1], u, c)

		results = p.map(helper, arg_pairs)

		for i, j, u, c in results:
			pixel_disp[i, j] = u
			pixel_conf[i, j] = c

		return pixel_disp, pixel_conf

	# PLOTTING FUNCTIONS
	def plot_og_imgs(self):
		fig, axs = plt.subplots(1, 2)
		axs[0].imshow(self.LeftImage, cmap="gray")
		axs[1].imshow(self.RightImage, cmap="gray")

	def plot_upsampled(self):
		fig, axs = plt.subplots(1, 2)
		axs[0].imshow(self.I0.T, cmap="gray")
		axs[1].imshow(self.I1.T, cmap="gray")
		plt.suptitle("Upsampled {}".format(consts.DENOISE_MET))

	def plot_normalized(self):
		fig, axs = plt.subplots(1, 3)
		axs[0].imshow(self.I0_norm.T, cmap="gray")
		axs[1].imshow(self.I1_norm.T, cmap="gray")
		axs[2].imshow(np.square((self.I0_norm).T - (self.I1_norm).T), cmap="jet")
		plt.title("Normalized {}".format(consts.DENOISE_MET))

	def plot_disparities(self):
		self.disparity_wrapper()
		coarse_offsets = self.coarse
		subpix_offsets = self.subpix

		fig1, axs1 = plt.subplots(1,2)
		im1 = axs1[0].imshow(coarse_offsets.T, cmap="jet", interpolation=None)
		im2 = axs1[1].imshow(subpix_offsets.T, cmap="jet", interpolation=None)
		axs1[0].set_title("Coarse {}".format(consts.DENOISE_MET))
		axs1[1].set_title("Subpix {}".format(consts.DENOISE_MET))
		fig1.colorbar(im2)

	def plot_pixel_disparities(self):
		pixel_disp, pixel_conf = self.disp_upsampler()
		self.pixDisp = pixel_disp
		self.pixConf = pixel_conf
		fig, axs = plt.subplots()
		im1 = axs.imshow(pixel_disp.T, cmap="jet", interpolation=None)
		fig.colorbar(im1)

	def plot_confidences(self):
		confidences = self.conf.T
		fig3, axs3 = plt.subplots()
		im3 = axs3.imshow(confidences, cmap="gray", interpolation=None)
		axs3.set_title("Confidence {}".format(consts.DENOISE_MET))
		fig3.colorbar(im3)

	def plot_hsv_pixelwise(self, denoise):
		H = np.zeros_like(self.pixDisp.T)
		S = np.abs(self.pixDisp.T) / 4.5 # normalized depth
		V = np.copy(self.pixConf.T) # map value to confidence
		if (denoise):
			V = self.smooth_conf(V)
		neg = np.where(self.pixDisp.T < 0.0)
		pos = np.where(self.pixDisp.T >= 0.0)
		H[neg] = 90.0 / 180.0
		H[pos] = 0.0
		HSV = np.dstack((H, S, V))
		RGB = hsv_to_rgb(HSV)
		plt.figure()
		plt.imshow(RGB)
		plt_title = "Pixel Disparities, Denoise={}".format(denoise)
		plt.title(plt_title)

	def plot_hsv_tiles(self, title):
		H = np.zeros_like(self.subpix.T)
		S = np.abs(self.subpix.T) / 4.5 # normalized depth
		V = np.copy(self.conf.T) # map value to confidence
		neg = np.where(self.subpix.T < 0.0)
		print(neg)
		pos = np.where(self.subpix.T >= 0.0)
		H[neg] = 90.0 / 180.0
		H[pos] = 0.0
		HSV = np.dstack((H, S, V))
		RGB = hsv_to_rgb(HSV)
		plt.figure()
		plt.imshow(RGB)
		plt.title(title)

	def save_info(self, name):
		fname = "{}_data.npz".format(name)
		ext_out = {"I0":self.I0, "disp":self.subpix, "pixel_disp":self.pixDisp,
		"conf":self.conf, "pixel_conf":self.pixConf, "I1":self.I1}
		np.savez(os.path.join(os.path.realpath("../"), "data", fname), **ext_out)

if __name__ == "__main__":
	# first set of trials
	# trial with image bursts
	trial1_burst = Image.from_burst(sys.argv[1])
	print("Getting coarse diffs from normalized, denoised and transposed...")
	trial1_burst.plot_upsampled()
	trial1_burst.plot_disparities()
	trial1_burst.plot_hsv_tiles("After Repeated Texture")
	trial1_burst.get_horizontal_grads()
	trial1_burst.plot_hsv_tiles("After Horizontal Grads")
	trial1_burst.get_outlier_tiles()
	trial1_burst.plot_hsv_tiles("After Outlier Tiles")
	trial1_burst.smooth_conf(trial1_burst.conf)
	trial1_burst.plot_confidences()
	if ((len(sys.argv) > 2) and sys.argv[2] == "pixel"):
		trial1_burst.plot_pixel_disparities()
		trial1_burst.plot_hsv_pixelwise(True)
		trial1_burst.save_info(sys.argv[1])
	plt.show()
