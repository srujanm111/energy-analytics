import sys
import cv2 as cv

filename = sys.argv[1]
image = cv.imread(filename)  
cropped = image[5:213, 1:177]  
cv.imwrite(filename, cropped)