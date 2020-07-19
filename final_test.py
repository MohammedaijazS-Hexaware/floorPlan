import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import json
from json import JSONEncoder
import imutils


# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)


imgg = cv2.imread('Perfect.png')
img = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
# imgg = imutils.resize(imgg,400,400)
# img=imutils.resize(img,400,400)

kernel = np.ones((5,5),np.uint8)
img_dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imwrite('Intermediate.jpg',img_dilation)
gray = np.float32(img_dilation)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

sure_round = np.around(corners, decimals = 0)

# numpyData = {"coordinates": round_off_values}
# encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
# print("Writing coordinates into json file")
# with open("coordinates.json", "w") as write_file:
#     json.dump(numpyData, write_file, cls=NumpyArrayEncoder)

l=[]
pixels = []
dictio = dict()
inter_res = []
result = []
l1 = sure_round.tolist()
l2=dict()
id=1
for g in l1:
    l2[id]=g
    id+=1

print(l1)
print()

im = Image.open(r"C:\Users\Home G\PycharmProjects\coronaSocialDistancing\Intermediate.jpg")

for x in range(0,len(corners)):
     for y in range(0,len(corners)):
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
             flag = 0
             for _ in range(240,256):
                 if _ in pixels:
                     flag = 1
             if flag == 0:
                inter_res.append('Source')
                for key,value in l2.items():
                    if(l1[x][0]==value[0] and l1[x][1]==value[1]):
                        inter_res.append(key)
                inter_res.append(l1[x])
                inter_res.append('Destination')
                for key,value in l2.items():
                    if(l1[y][0]==value[0] and l1[y][1]==value[1]):
                        inter_res.append(key)
                inter_res.append(l1[y])
                result.append(inter_res)
                inter_res = []

             print(l1[x])
             print(l1[y])
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
             flag = 0
             for _ in range(240, 256):
                 if _ in pixels:
                     flag = 1
             if flag == 0:
                 inter_res.append('Source')
                 for key, value in l2.items():
                     if (l1[x][0] == value[0] and l1[x][1] == value[1]):
                         inter_res.append(key)
                 inter_res.append(l1[x])
                 inter_res.append('Destination')
                 for key, value in l2.items():
                     if (l1[y][0] == value[0] and l1[y][1] == value[1]):
                         inter_res.append(key)
                 inter_res.append(l1[y])
                 result.append(inter_res)
                 inter_res = []
             print(l1[x])
             print(l1[y])
             print(pixels)
             print()
             pixels=[]

print(result)

for t in result:
    o1 = t[2]
    o2 = t[5]
    cv2.line(imgg,(int(o1[0]),int(o1[1])),(int(o2[0]),int(o2[1])),(0,0,255),2)

dimen = imgg.shape[:2]
final_dict = dict()
walls = dict()
doors = dict()
windows = dict()
coordinates = []
metadata = {'url':'','dimensions':dimen}
final_dict['metadata']=metadata
ids = []
ids2 = []
for _ in result:
    for i2 in result:
        if (_[1] == i2[1]):
            ids.append(i2[4])
    for i3 in coordinates:
        ids2.append(i3["id"])
    if _[1] not in ids2:
        coordinates.append({"id": _[1], "xy": _[2], "Connections": ids})
    ids = []
walls["corners"]=coordinates
final_dict["walls"]=walls
final_dict["Doors"]=doors
final_dict["Windows"]=windows
final_json=json.dumps(final_dict)
print(final_json)

with open('result.json', 'w') as fp:
    json.dump(final_json, fp)

im = plt.imread('Perfect.png')
implot = plt.imshow(im)
for i in range(1, len(sure_round)):
    x_cord = sure_round[i][0]
    y_cord = sure_round[i][1]
    plt.scatter([x_cord], [y_cord])
plt.show()

imgg[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('Final_F', imgg)
cv2.imshow('Original',img)
cv2.imshow('Intermediate',img_dilation)

cv2.waitKey(0)
cv2.destroyAllWindows