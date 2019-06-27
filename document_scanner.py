import imutils
import cv2 as cv
import numpy as np
from argparse import ArgumentParser
from transform import four_point_transform
from skimage.filters import threshold_local

#initialise argument parser and add arguments
ap = ArgumentParser()
ap.add_argument("--image", type=str, default="receipt.jpg", 
				help="path of image to be used")
args = vars(ap.parse_args())

#load the image and resize it
image = cv.imread(args["image"])
ratio = image.shape[0] / 500.0
image = imutils.resize(image, height=500)

#convert image to grayscale, blur it and find edges
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurr = cv.GaussianBlur(gray, (5,5), 0)
edged = cv.Canny(blurr, 75, 200)

#find contour
cnts = cv.findContours(edged, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#find the contour with largest area, it is assumed that page will have 
#largest contour
cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

for c in cnts:
	#draw contour
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02*peri, True)

	#if contout has 4 points, it is assumed to be a paper
	if len(approx) == 4:
		screenCnt = approx
		break

#display contour
contour = cv.drawContours(image.copy(), [screenCnt], -1, (0,255,0), 2)

#find four point transform of image
copy = image.copy()
warped = four_point_transform(copy, screenCnt.reshape(4, 2))


#convert image to gray then threshod it
warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped>T).astype("uint8")*255

#display original and transformed image
cv.imshow("Image", copy)
cv.imshow("Scanned", imutils.resize(warped,height=500))
cv.waitKey(0)
cv.destroyAllWindows()