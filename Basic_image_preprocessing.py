import numpy as np
import cv2 as cv

def rotatePt(pt, centerPt, angle):
    """
    :param pt: point to be rotate, tuple
    :param centerPt: anchor point
    :param angle: in degree
    :return: coordinates after the rotating
    """
    angle = angle * np.pi/180
    M = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).astype('float')
    X = np.array([pt[0] - centerPt[0], pt[1] - centerPt[1]]).astype('float')
    X_new = np.add(M.dot(X), centerPt)
    return tuple(X_new.astype('int'))

def getUserChoice(options):
    """
    :param options: list of options
    :return: a number indicate which option is chosen
    """
    for i in range(len(options)):
        print("Function",i + 1, ":", options[i])
    return int(input("Please choose (1.." + str(len(options)) + "):"))

def drawRectangle(rec, img):
    """
    :param rec:  cvRectangle: instance to be drawn
    :param img: canvas
    :return: img with new rectangle
    """

    #Green color
    color = (0, 0, 255)
    topLPt = rotatePt((rec.cx - rec.w / 2, rec.cy - rec.h / 2), (rec.cx, rec.cy), rec.angle)
    topRPt = rotatePt((rec.cx + rec.w / 2, rec.cy - rec.h / 2), (rec.cx, rec.cy), rec.angle)
    botRPt = rotatePt((rec.cx + rec.w / 2, rec.cy + rec.h / 2), (rec.cx, rec.cy), rec.angle)
    botLPt = rotatePt((rec.cx - rec.w / 2, rec.cy + rec.h / 2), (rec.cx, rec.cy), rec.angle)
    #img = cv.line(img, topLPt, topRPt, color, 2)
    #img = cv.line(img, topRPt, botRPt, color, 2)
    #img = cv.line(img, botRPt, botLPt, color, 2)
    #img = cv.line(img, botLPt, topLPt, color, 2)
    pts = np.array([topLPt, topRPt, botRPt, botLPt], np.int32)
    img = cv.fillPoly(img, pts = [pts], color = color)
    return img.astype('int')

class cvRectangle:
    def __init__(self):
        self.cx = 0
        self.cy = 0
        self.w = 0
        self.h = 0
        self.angle = 0
    def rotate(self, angle):
        self.angle = angle
    def translate(self, offset):
        self.cx += offset[0]
        self.cy += offset[1]
    def scale(self, scales):
        self.w *= scales[0]
        self.h *= scales[1]


stat = False
drawing = False
ix, iy = -1, -1
img = np.ones((200, 200, 3), np.uint8) * 255
rec = cvRectangle()

# Define mouse callback function
def draw_rectangle(event, x, y, flags, params):
    global drawing, ix, iy, img, rec, stat
    if event == cv.EVENT_LBUTTONDOWN:
        #print("Button down")
        drawing = True
        ix = x
        iy = y
    elif event == cv.EVENT_MOUSEMOVE:
        #print("Mouse move")
        if drawing == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        #print("Button up")
        drawing = False
        cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
        rec.w = abs(x - ix)
        rec.h = abs(y - iy)
        rec.cx = (x + ix) / 2
        rec.cy = (y + iy) / 2
        rec.angle = 0
        stat = True
        cv.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)

def main():
    global ix, iy, img, rec, stat
    cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
    cv.imshow("image", img)
    cv.setMouseCallback("image", draw_rectangle, [img,rec])

    options = ("White background",
               "Draw rectangle",
               "Translation",
               "Rotation",
               "Scaling",
               "Exit")

    while(True):
        #img = setColorforImg(img)
        userChoice = getUserChoice(options)
        if userChoice == 1:
            img = np.ones((1024, 1024, 3), np.uint8) * 255
        elif userChoice == 2:
            #Draw rectangle here
            ix, iy = -1, -1
            while(stat == False):
                cv.imshow("image", img)
                if cv.waitKey(10) == 27:
                    break
            cv.setMouseCallback("image", lambda *args: None)
            stat = False


        elif userChoice == 3:
            dx = int(input("x: "))
            dy = int(input("y: "))
            rec.translate((dx,dy))
            img = np.ones((1024, 1024, 3), np.uint8) * 255
            drawRectangle(rec, img)
        elif userChoice == 4:
            rec.rotate(int(input("Rotation angle = ")))
            img = np.ones((1024, 1024, 3), np.uint8) * 255
            drawRectangle(rec, img)
        elif userChoice == 5:
            sx = float(input("x: "))
            sy = float(input("y: "))
            rec.scale((sx, sy))
            img = np.ones((1024, 1024, 3), np.uint8) * 255
            drawRectangle(rec, img)
        elif userChoice == 6: break
        cv.imshow("image", img)
        cv.waitKey(20)

main()
