import numpy as np
import cv2

imgg = cv2.imread('Perfect.png')
img = cv2.imread('Perfect.png',0)
kernel = np.ones((5,5),np.uint8)
img_dilation = cv2.dilate(img, kernel, iterations=1)
gray = np.float32(img_dilation)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
for i in range(1, len(corners)):
    print(corners[i])
imgg[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('Final', imgg)
cv2.imshow('Original',img)
cv2.imshow('Intermediate',img_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows