import cv2
from PIL import Image


fourcc = cv2.VideoWriter_fourcc(*"MJPG")
fps = 25
im = Image.open("time0.png")
vw = cv2.VideoWriter("video" + '.avi', fourcc, fps, im.size)


for i in range(1000):
    frame = cv2.imread("time" + str(i) + '.png')
    vw.write(frame)
    print(i)

vw.release()

