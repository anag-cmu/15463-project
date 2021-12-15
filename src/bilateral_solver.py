import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg
from skimage.io import imread, imsave
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import consts
import os
import sys

# ADAPTED FROM the implementation at https://github.com/poolio/bilateral_solver
# to work with this data
hashing_vec = np.array([255 ** i for i in range(3)])

class BilateralGrid:
	"""
		Constructs factorization of Weight/Affinity matrix based on the given Img
	"""
	def __init__(self, ref_img, sigmaSpatial, numSpaces):
		# first we want to get coordinates in the 3d voxel grid
		step_size = (np.max(ref_img) - np.min(ref_img)) / (numSpaces - 1.0)

    	# converts image intensities into coordinates in the j-spaced tick marks
		luma_coords = (ref_img - np.min(ref_img)) / (step_size)
		luma_coords = luma_coords.astype(int)

    	# also gets all the coordinates
		x_coords, y_coords = np.meshgrid(np.arange(ref_img.shape[1]),
		np.arange(ref_img.shape[0]))
		x_coords = x_coords / sigmaSpatial
		y_coords = y_coords / sigmaSpatial
		x_coords = x_coords.astype('int')
		y_coords = y_coords.astype('int')
		all_coords = np.dstack((x_coords, y_coords, luma_coords)).astype('int')
		coords_flat = all_coords.reshape(-1, 3)
		self.npixels = coords_flat.shape[0]
		self.get_matrices(coords_flat)

	def get_matrices(self, coords_flat):
		# map each coordinate to 1 * y + 255 * x + 65025 * luma
		coords_hashed = coords_flat.dot(hashing_vec)

		# map the hashed coordinates to unique vertices in the voxel grid
		unique_coords = {}

		# row indices of vertices for each pixel column
		splat_rows = np.zeros(self.npixels)

		num_unique = 0 # keeps track of number of unique that we've encountered so far
		for pix_idx in range(self.npixels):
			if (coords_hashed[pix_idx] in unique_coords):
				splat_rows[pix_idx] = unique_coords[coords_hashed[pix_idx]]
			else:
				unique_coords[coords_hashed[pix_idx]] = num_unique
				splat_rows[pix_idx] = num_unique
				num_unique += 1


		# Construct sparse splat matrix that takes set of pixels as input and "splats"
		# them to some vertex
		self.nvertices = len(unique_coords.keys())
		self.S = csr_matrix((np.ones(self.npixels), (splat_rows, np.arange(self.npixels))),
		shape=(self.nvertices, self.npixels))

		# Now we construct blur matrices for all 3 blur matrices
		# We get "1 0 1" type of matrices in bilateral space
		blurs = []
		unique_hashcoords = np.array(list(unique_coords.keys()))

		for d in range(3):
			blur = 0.0
			for offset in (-1, 1):
				#print(offset)
				# get the offset_hashcoords
				offset_hashcoords = (hashing_vec[d] * offset)
				neighbor_hashcoords = unique_hashcoords + offset_hashcoords
				#print(neighbor_hashcoords)
				# where a unique vertex exists for the neighbor
				mask = np.array([(x in unique_coords) for x in neighbor_hashcoords])
				#print(np.where(~mask)[0])
				valid_center_hc = unique_hashcoords[mask]
				valid_neighbor_hc = neighbor_hashcoords[mask]
				#print(valid_center_hc.shape)

				valid_centers = np.array([unique_coords[i] for i in valid_center_hc])
				valid_neighbors = np.array([unique_coords[i] for i in valid_neighbor_hc])
				num_valid = np.size(valid_center_hc)

				# now we construct blur matrix from this
				blur = blur + csr_matrix((np.ones(num_valid),
				(valid_centers, valid_neighbors)),
				shape=(self.nvertices, self.nvertices))

			blurs.append(blur)

		self.B0 = blurs[0]
		self.B1 = blurs[1]
		self.B2 = blurs[2]

	def splat(self, x):
		return self.S.dot(x)

	def slice(self, x):
		return self.S.T.dot(x)

	def blur(self, x):
		out = 2 * 3 * x # add weight to center vertex
		out = out + self.B0.dot(x) + self.B1.dot(x) + self.B2.dot(x)
		return out


class BilateralSolver:
	"""
		Uses Affinity matrix factorized into BilateralGrid, target, and confidence
		to "solve" for the x which minimizes 0.5 * y.T @ A @ y - b.T @ y + c
	"""
	def __init__(self, W_grid, conf, target):
		self.grid = W_grid
		self.c = conf.reshape(-1, 1) # hopefully num_pixels by 1
		self.t = target.reshape(-1, 1) # hopefully num_pixels by 1
		sol = self.solve()
		self.xhat = sol

	def bistocasticize(self):
		"""
			Recover D_n and D_m such that S.T D_m^-1 D_n B D_n D_m^-1 S
			and SS.T = D_m
			We want rows and cols to both sum up to 1
		"""
		m = self.grid.splat(np.ones(self.grid.npixels)) # "mass vector of vertices"
		n = np.ones(self.grid.nvertices)

		for i in range(consts.BIL_BIS_MAXITERS):
			numerator = n * m
			denom = self.grid.blur(n)
			n = np.sqrt(numerator / denom)

		m = n * self.grid.blur(n)
		self.Dn = diags(n, 0)
		self.Dm = diags(m, 0)

	def build_A(self):
		A_smooth = consts.BIL_LAMBDA * (self.Dm - self.Dn.dot(self.grid.blur(self.Dn)))
		A_c = diags(self.grid.splat(self.c)[:,0], 0)
		self.A = A_smooth + A_c

	def build_b(self):
		tc_prod = self.t * self.c
		self.b = self.grid.splat(tc_prod)

	def preconditioner(self):
		"""
			Jacobi precondition
		"""
		A_diag = np.maximum(self.A.diagonal(), consts.EPS_SMALL)
		self.M = diags(1 / A_diag, 0)

	def init_y(self):
		s_ct = self.grid.splat(self.c * self.t)
		s_c = self.grid.splat(self.c)
		self.y0 = s_ct / s_c

	def solve(self):
		print("Bistochastization S and B...")
		self.bistocasticize()

		print("Calculating A and b...")
		self.build_A()
		self.build_b()

		print("Preconditioning and initializing...")
		self.preconditioner()
		self.init_y()

		print("Solving...")
		yhat, info = cg(self.A, self.b, x0=self.y0, M=self.M, maxiter=consts.BIL_CG_MAXITERS,
		tol=consts.EPS_CG)
		xhat = self.grid.slice(yhat)
		return xhat


class BilateralWrapper:
	def __init__(self, npz_file, name):
		# first we must normalize the depth map because we assume target is normalized
		fname = os.path.join(os.path.realpath("../"), "data", npz_file)
		with np.load(fname) as Y:
			I0, I1, pixelDisp, coarseDisp, pixelConf, coarseConf = [Y[i] for i in
			('I0', 'I1', 'pixel_disp', 'disp', 'pixel_conf', 'conf')]

		GridPixel = BilateralGrid(I0, 8, 100)
		minDisp = np.min(pixelDisp)
		diffDisp = np.max(pixelDisp) - np.min(pixelDisp)
		pixelDisp = (pixelDisp - minDisp) / (diffDisp)
		pixelConf = pixelConf + 1e-21
		solPixel = BilateralSolver(GridPixel, pixelConf, pixelDisp)
		self.solPixel = solPixel.xhat.reshape(pixelDisp.shape).T
		self.solPixel = ((self.solPixel) * (diffDisp)) + minDisp
		self.refSol = I0.T
		self.pixelConf = pixelConf
		self.pixelDisp = (pixelDisp * diffDisp) + minDisp
		self.Name = name

	def hsv_pixelwise(self, disp):
		H = np.zeros_like(disp)
		S = np.abs(disp) / 4 # normalized depth
		V = np.copy(self.pixelConf.T) # map value to confidence
		neg = np.where(disp < 0.0)
		pos = np.where(disp >= 0.0)
		H[neg] = 90.0 / 180.0
		H[pos] = 0.0
		HSV = np.dstack((H, S, V))
		RGB = hsv_to_rgb(HSV)
		return RGB

	def plot_sol(self):
		plt_title = "Pixel Disparities {}".format(self.Name)
		smooth_title = plt_title + " Smoothed"
		fig1, axs1 = plt.subplots(1, 2)
		im1 = axs1[0].imshow(self.solPixel, cmap="jet")
		axs1[1].imshow(self.hsv_pixelwise(self.solPixel))
		fig1.colorbar(im1)
		plt.suptitle(smooth_title)

		fig2, axs2 = plt.subplots(1, 2)
		im2 = axs2[0].imshow(self.pixelDisp.T, cmap="jet")
		axs2[1].imshow(self.hsv_pixelwise(self.pixelDisp.T))
		fig2.colorbar(im2)
		plt.suptitle(plt_title)

		plt.figure()
		plt.imshow(self.pixelConf.T, cmap="gray")
		plt.title("Confidence")

	def save_info(self, name):
		fname = "{}_smoothed.npz".format(name)
		ext_out = {"I0":self.refSol, "disp":self.solPixel}
		np.savez(os.path.join(os.path.realpath("../"), "data", fname), **ext_out)


if __name__ == "__main__":
	fname_load = "{}_data.npz".format(sys.argv[1])
	Wrapper_1 = BilateralWrapper(fname_load, sys.argv[1])
	Wrapper_1.plot_sol()
	#Wrapper_1.save_info(sys.argv[1])
	plt.show()
