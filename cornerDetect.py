import numpy as np
import cv2

imgg = cv2.imread('Perfect.png')
img = cv2.imread('Perfect.png',0)
kernel = np.ones((5,5),np.uint8)
img_dilation = cv2.dilate(img, kernel, iterations=1)
gray = np.float32(img_dilation)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
imgg[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',imgg)
cv2.imshow('dilation',img_dilation)
cv2.imshow('Original',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()