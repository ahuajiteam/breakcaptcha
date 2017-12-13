# reader.py
# author: loriex
# time: 19:49 2017-12-13
# guide:
# use empty() to check if all png is geted
# use init(path) to intialize this system, path is the folder path of imgs.
# use get() to get a object T
# object T:
# T.plaintext: the captcha's text
# T.width: the arrows of matrice
# T.height: the lines of matrice
# T.img: the image (object Image)
# T.width == -1 means that this matrice is unavailable
#               (means that all pics was geted)

import matplotlib.pyplot as plt
from PIL import Image
import os
import string

# image_binarize: input a Image
# then binarize the Image and return a RGB Image
# I calc the average grey-level
# then divide the Image into two group, (grep-level higher|lower)
# then choose the small-size group as white, another group as black
# then convert the result into RGB Image
def image_binarize(img):
    # convert to grep image
    img = img.convert('L')
    width = img.size[0]
    height = img.size[1]
    # calc the average grey-level
    sum = 0
    for i in range(0, width):
        for j in range(0, height):
            sum += img.getpixel((i, j))
    sum /= width * height

    print("average grey-level: ")
    print(sum)
    # count the size of grep-level-higher
    count = 0
    for i in range(0, width):
        for j in range(0, height):
            if img.getpixel((i, j)) > sum:
                count = count + 1
    # decide which group is white
    choice = ''
    if count * 2 < width * height:
        choice = 'B' # bigger
    else:
        choice = 'S' # smaller

    # built a color-table
    table = []
    if choice == 'B':
        for i in range(256):
            if i > sum:
                table.append(1)
            else:
                table.append(0)
    else:
        for i in range(256):
            if i < sum:
                table.append(1)
            else:
                table.append(0)
    img = img.point(table, '1')

    img = img.convert('RGB')
    return img

def image_denoising(img):
    return img

class PngMat:
    width = -1
    height = -1
    img = 0
    plaintext= ""
    def __init__(self, width = -1, height = -1):
        self.width = width
        self.height = height


class PngReader():
    filepath = ""
    filelist = []
    total = 0
    index = 0
    def __init__(self, path):
        if path[len(path)-1] == '/':
            self.filepath = path
        else:
            self.filepath = path + '/'
        self.filelist.clear()
        self.total = 0
        self.index = 0

        dirs = os.listdir(path)
        for allDir in dirs:
            if allDir.endswith(".png"):
                self.total = self.total + 1
                self.filelist.append(allDir)
                #child = os.path.join('%s%s' % (self.filepath, allDir))
                #print(child)

        if self.total == 0:
            print("no png found...\n")

    def empty(self):
        return self.index == self.total

    def get(self):
        T = PngMat()
        if self.index == self.total:
            return T
        path = self.filepath + self.filelist[self.index]
        print(path)
        img = Image.open(path)
        img = image_binarize(img)
        img = image_denoising(img)
        T.width = img.size[0]
        T.height = img.size[1]
        T.img = img
        T.plaintext = self.filelist[self.index][0:4]
        self.index = self.index + 1
        return T

# usage example:
if __name__ == '__main__':
    mreader = PngReader("./hua/") #(path)
    T = mreader.get()
    mreader.empty()
    print(T.width, T.height)
    print(T.plaintext)
    plt.imshow(T.img)
    plt.show()
#usage over
