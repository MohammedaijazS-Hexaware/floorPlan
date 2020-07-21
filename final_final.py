from flask import Flask,jsonify, request
app = Flask(__name__)
import numpy as np
import cv2
from PIL import Image
import json
import base64

@app.route("/",methods = ['POST'])
def floor_plan():
    base_64_json = request.get_json()
    base_64=base_64_json["base__64"]
    imgdata = base64.b64decode(base_64)
    filename = 'input.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    imgg = cv2.imread('input.jpg')
    img = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)

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

    im = Image.open(r"C:\Users\Home G\PycharmProjects\coronaSocialDistancing\Intermediate.jpg")

    for x in range(0,len(corners)):
         for y in range(0,len(corners)):
             xx1 = int(l1[x][0])
             yy1 = int(l1[x][1])
             xx2 = int(l1[y][0])
             yy2 = int(l1[y][1])
             if (xx1==xx2 and yy1==yy2):
                 continue
             x1 = int(l1[x][0])
             y1 = int(l1[x][1])
             x2 = int(l1[y][0])
             y2 = int(l1[y][1])
             points = []
             issteep = abs(y2 - y1) > abs(x2 - x1)
             if issteep:
                 x1, y1 = y1, x1
                 x2, y2 = y2, x2
             rev = False
             if x1 > x2:
                 x1, x2 = x2, x1
                 y1, y2 = y2, y1
                 rev = True
             deltax = x2 - x1
             deltay = abs(y2 - y1)
             error = int(deltax / 2)
             yy = y1
             ystep = None
             if y1 < y2:
                 ystep = 1
             else:
                 ystep = -1
             for xx in range(x1, x2 + 1):
                 if issteep:
                     points.append((yy, xx))
                 else:
                     points.append((xx, yy))
                 error -= deltay
                 if error < 0:
                     yy += ystep
                     error += deltax
             # Reverse the list if the coordinates were reversed
             if rev:
                 points.reverse()
             if len(points)==1:
                 del points[0]
             elif len(points)>1:
                 del points[0]
                 del points[len(points)-1]
             for j in points:
                 coords = g, h = j[0], j[1]
                 pixels.append(im.getpixel(coords));
             flag = 0
             for _ in range(240, 256):
                 if _ in pixels:
                     flag = 1

             if flag == 0:
                 inter_res.append('Source')
                 for key, value in l2.items():
                     if (xx1 == int(value[0]) and yy1 == int(value[1])):
                         inter_res.append(key)
                 inter_res.append([xx1, yy1])
                 inter_res.append('Destination')
                 for key, value in l2.items():
                     if (xx2 == int(value[0]) and yy2 == int(value[1])):
                         inter_res.append(key)
                 inter_res.append([xx2, yy2])
                 result.append(inter_res)
                 inter_res = []
             pixels = []

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
            coordinates.append({"id": _[1], "xy": _[2], "connections": ids})
        ids = []
    walls["corners"]=coordinates
    final_dict["walls"]=walls
    final_dict["doors"]=doors
    final_dict["windows"]=windows
    final_json=json.dumps(final_dict)

    with open('result.json', 'w') as fp:
        json.dump(final_dict, fp)
    return jsonify(final_dict)

if __name__ == '__main__':
    app.run(debug=True)