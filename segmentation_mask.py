import sys
import cv2 as cv
import numpy as np

# get paths from arguments
maskless_filename = sys.argv[1]
masked_filename = sys.argv[2]
output_filename = sys.argv[3]

# create image matrices
maskless = cv.imread(maskless_filename);
masked = cv.imread(masked_filename)
mask = maskless.copy()

# create mask from pixel differences
cv.absdiff(maskless, masked, mask)
mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
_, mask = cv.threshold(mask, 10, 255, cv.THRESH_BINARY)

# close holes in mask
kernel1 = np.ones((5, 5), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel1)

cv.imwrite(output_filename, mask)