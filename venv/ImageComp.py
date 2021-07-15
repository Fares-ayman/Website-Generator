import cv2
import numpy as np
import glob

from UI import final_pic

class component:

    # init method or constructor
    def __init__(self,id, name,x,y,w,h,lable):
        self.id = id
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h=h
        self.lable=lable


x_com=[]
y_com=[]
w_com=[]
h_com=[]
detected=[]
global components

components = []

def detect_btn_comb(obj):
    Button_list = []
    DropDown_list = []
    # detected.append("Button Combo")
    for f in glob.iglob("images\Button\*"):
        image = cv2.imread(f)
        Button_list.append(image)
    for f in glob.iglob("images\DropDown\*"):
        image = cv2.imread(f)
        DropDown_list.append(image)
    Button_dataset = Button_list[::-1]
    DropDown_dataset = DropDown_list[::-1]

    max = 0
    title = ""
    obj = cv2.resize(obj, (300, 220), interpolation=cv2.INTER_CUBIC)
    # keypoints_1, descriptors_1 = surf.detectAndCompute(obj, None)
    for button in Button_dataset:
        button = cv2.resize(button, (300, 220), interpolation=cv2.INTER_CUBIC)

        surf = cv2.xfeatures2d.SURF_create()

        keypoints_1, descriptors_1 = surf.detectAndCompute(obj, None)
        keypoints_2, descriptors_2 = surf.detectAndCompute(button, None)

        # feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_points = []
        for m in matches:
            if m.distance < 0.7:
                good_points.append(m)
        if len(good_points) > max:
            max = len(good_points)
            title = "Button"

    for dropdown in DropDown_dataset:
        dropdown = cv2.resize(dropdown, (300, 220), interpolation=cv2.INTER_CUBIC)

        surf = cv2.xfeatures2d.SURF_create()

        keypoints_1, descriptors_1 = surf.detectAndCompute(obj, None)
        keypoints_2, descriptors_2 = surf.detectAndCompute(dropdown, None)

        # feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_points = []
        for m in matches:
            if m.distance < 0.7:
                good_points.append(m)
        if len(good_points) > max:
            max = len(good_points)
            title = "DropDown"

    return title

def identifyElement(objects, contour,detected):
    for i in range(len(objects)):
        # print(contour[i])
        if contour[i] >= 10:
            detected.append("Image Icon")
        elif (objects[i].size / (objects[i].shape[1] - objects[i].shape[0])) >= 100:
            _,contours, _ = cv2.findContours(objects[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) <= 40:
                detected.append("Text Area")
            else:
                detected.append("Paragraph")
        else:
            _,contours, hierarchy = cv2.findContours(objects[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            rect = 0
            combo = 0
            for cnt in contours:
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 20 and h > 20:
                    # print(len(approx))
                    if len(approx) != 4:
                        combo += 1
                        rect = 0
                        break
                    else:
                        rect += 1
            if (rect == 1 or rect == 2) and not (combo > 0):
                detected.append("Text Input")
            else:
                res=detect_btn_comb(objects[i])
                detected.append(res)


def getcontours(img, imgContour, objects, imgOriginal, cnt,x_com,y_com,w_com,h_com):
    _,contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w > 20 and h > 20 and hierarchy[0][i][3] == -1:
            # print(hierarchy[0][i])
            approx = cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 0, 255), 0)
            imgCropped = img[y - 5:y + h + 5, x - 5:x + w + 5]
            epsilon = 0.01 * cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            cnt.append(len(approx))
            objects.append(imgCropped)
            x_com.append(x)
            y_com.append(y)
            w_com.append(w)
            h_com.append(h)

img = cv2.imread(UI.final_pic)
img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
imgContour = img.copy()
imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(imgGray, 40, 100)
kernel = np.ones((2, 2))
imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

objects = []
cnt = []
getcontours(imgDil, imgContour, objects, img, cnt,x_com,y_com,w_com,h_com)
elements = objects[::-1]
cnt = cnt[::-1]
identifyElement(elements, cnt,detected)
x_com=x_com[::-1]
y_com=y_com[::-1]
w_com=w_com[::-1]
h_com=h_com[::-1]

for i in range(len(detected)):
    c=component(detected[i]+str(i),detected[i],x_com[i],y_com[i],w_com[i],h_com[i],"")
    components.append(c)

cv2.imshow("Result", imgContour)
cv2.waitKey(0)