# preprocessing stage constants
BOX_RADIUS = 8
MAX_VAL = 65535.0
EPS = 0.0001
SIGMA_X = 8
SIGMA_Y = 0.5

DENOISE_MET = "BILATERAL"

# coarse estimate constants
TILE_H = 8 # should be some divisor of 2016
TILE_W = 8 # should be some divisor of 1512
STRIDE_LENGTH = 8

# subpixel estimates
EPS_SMALL = 0.0000001
SIGMA_A = 5
SIGMA_c = 256

# heuristic parameters
THRESH_PERCENT = 1200
w_v = 8.0

e_d = 70.0 # small value that ensures the "repeated texture" still has an effect
# as d approaches 0

# range of ratios over which "repeated textures" transitions from no effect to full
r_0 = 0.6
r_1 = 0.8

# weight of "repeated texture"
w_r = 1.0

# expected variance of flow
VAR_FLOW = 0.5

# bilateral solving parameters
BIL_LAMBDA = 120
BIL_BIS_MAXITERS = 20
BIL_CG_MAXITERS = 25
EPS_CG = 1e-20
