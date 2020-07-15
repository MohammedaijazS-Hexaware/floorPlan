import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


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
j=0

round_off_values = np.around(corners, decimals = 0)

for i in range(1, len(round_off_values)):
    if(i<len(round_off_values)-1):
        if(round_off_values[i][1]==round_off_values[i+1][1]):
            if((round_off_values[i+1][0] - round_off_values[i][0]) < 10):
                value = (round_off_values[i][0] + round_off_values[i+1][0])/2
                round_off_values[i][0]=value
                round_off_values[i+1][0]=value

round_off_values = np.unique(round_off_values, axis=0)

for i in range(1,len(round_off_values)):
    if(i<len(round_off_values)-1):
        if(round_off_values[i][0]==round_off_values[i+1][0]):
            if((round_off_values[i+1][1]-round_off_values[i][1])<30):
                value = (round_off_values[i][1]+round_off_values[i+1][1])/2
                round_off_values[i][1]=value
                round_off_values[i][1]=value

round_off_values = np.unique(round_off_values, axis=0)


numpyData = {"coordinates": round_off_values}
encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
print("Writing coordinates into json file")
with open("coordinates.json", "w") as write_file:
    json.dump(numpyData, write_file, cls=NumpyArrayEncoder)

#for i in range(1, len(round_off_values)):
    #j+=1
    #print(round_off_values[i])
#print(j)
l=[]
pixels = []
dictio = dict()
inter_res = []
result = []
l1 = round_off_values.tolist()
print(l1)
im = Image.open(r"C:\Users\Home G\PycharmProjects\coronaSocialDistancing\IntermediateRes1.jpg")
print()
for x in range(0,len(round_off_values)):
     for y in range(0,len(round_off_values)):
         # print(round_off_values[x])
         # print(round_off_values[y])
         # print()
         if((l1[x][0]==l1[y][0] and l1[x][1]!=l1[y][1])):
             a=l1[x][0]
             b=l1[x][1]
             c=l1[y][1]
             if(b<c):
                 u=b+1
                 while(u<c):
                     coords = g, h = a, u
                     pixels.append(im.getpixel(coords));
                     u+=1
             else:
                 u = c + 1
                 while(u<b):
                     coords = g, h = a, u
                     pixels.append(im.getpixel(coords));
                     u+=1
             if 255 not in pixels:
                 inter_res.append('Source')
                 inter_res.append(l1[x])
                 inter_res.append('Destination')
                 inter_res.append(l1[y])
                 result.append(inter_res)
                 inter_res = []

             print(round_off_values[x])
             print(round_off_values[y])
             print(pixels)
             print()
             pixels=[]
         elif((l1[x][0]!=l1[y][0] and l1[x][1]==l1[y][1])):
             a = l1[x][1]
             b = l1[x][0]
             c = l1[y][0]
             if (b < c):
                 u = b + 1
                 while (u < c):
                     coords = g, h = u, a
                     pixels.append(im.getpixel(coords));
                     u += 1
             else:
                 u = c + 1
                 while (u < b):
                     coords = g, h = u, a
                     pixels.append(im.getpixel(coords));
                     u += 1
             if 255 not in pixels:
                 inter_res.append('Source')
                 inter_res.append(l1[x])
                 inter_res.append('Destination')
                 inter_res.append(l1[y])
                 result.append(inter_res)
                 inter_res = []
             print(round_off_values[x])
             print(round_off_values[y])
             print(pixels)
             print()
             pixels=[]
print(result)
         #     p = round_off_values[x]
         #     d = round_off_values[y] - round_off_values[x]
         #     N = np.max(np.abs(d))
         #     s = d / N
         #     #print(np.rint(p).astype('int'))
         #     for ii in range(0, np.rint(N).astype('int')):
         #         p = p + s;
         #         l.append(np.rint(p).astype('int').tolist())
         #     for j in l:
         #         coord = x, y = j[0], j[1]
         #         pixels.append(im.getpixel(coord));
         #     print(l)
         #     print()
         #     print(pixels)
         #     print()
         #     print()
         #     l=[]
         #     pixels=[]
         # else:
         #     continue

im = plt.imread('Perfect.png')
implot = plt.imshow(im)
for i in range(1, len(round_off_values)):
    x_cord = round_off_values[i][0]
    y_cord = round_off_values[i][1]
    plt.scatter([x_cord], [y_cord])
plt.show()


imgg[dst>0.01*dst.max()]=[0,0,255]



cv2.imshow('Final', imgg)
cv2.imshow('Original',img)
cv2.imshow('Intermediate',img_dilation)


edges = cv2.Canny(img_dilation, 150, 170)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=0)
for line in lines:
   x1, y1, x2, y2 = line[0]
   cv2.line(img_dilation, (x1, y1), (x2, y2), (0, 0, 128), 1)
cv2.imshow("linesEdges", edges)
cv2.imshow("linesDetected", img_dilation)



cv2.waitKey(0)
cv2.destroyAllWindows