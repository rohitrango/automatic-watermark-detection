import os,sys
from cv2 import cv2
import glob
from PIL import Image

os.mkdir('Resized')
path = "wildehm/"
dirs = os.listdir( path )

def resize():
    i=0
    for item in dirs:
        if os.path.isfile(path+item):
            image = cv2.imread(path+item)
            imgResized = cv2.resize(image, (800,500))
            cv2.imwrite("Resized/image%04i.jpg" %i, imgResized)
            i +=1
            cv2.imshow('image', imgResized)
            cv2.waitKey(30)
    cv2.destroyAllWindows()

resize()

