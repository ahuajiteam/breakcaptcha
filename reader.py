# reader.py
# It's the middle layer of the generator and the trainer
# some Interfaces for trainer

import matplotlib.pyplot as plt
from PIL import Image
import os
import generator
import ClearNoise


# image_binarize: input a Image
# then binarize the Image and return a RGB Image
# I calc the average grey-level
# then divide the Image into two group, (grep-level higher|lower)
# then choose the small-size group as white, the other group as black
# then return the img
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

    if __name__ == '__main__':
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

    #img = img.convert('RGB')
    return img

# imagesolve: input img, width, height
# return a (width, height) img which is binarized
def imagesolve(img, width, height, flags = "NO_CLEAR_NOISE"):
    if flags == "CLEAR_NOISE":
        img = ClearNoise.pre_image(img)
    img = img.resize((width, height))
    img = image_binarize(img)
    return img

# imagetovec: input a binarized (width, height) img 
# return a x-vector which is needed by identify.py
def imagetovec(img, width, height):
    x = [0 for i in range(width * height)]
    for i in range(height):
        for j in range(width):
            x[i*width+j] = img.getpixel((j,i))
    return x

# class dongzj: a class for other partners to use
class dongzj:
    mwidth = 0
    mheight = 0
    table = {} # a map, map characters to numbers
    # determine the width & height, build the table
    def __init__(self, width, height):
        self.mwidth = width
        self.mheight = height
        # make a offset table
        cnt = 0
        for i in range(ord('0'), ord('9')+1):
            self.table[i] = cnt
            cnt = cnt + 1
        for i in range(ord('A'), ord('Z')+1):
            self.table[i] = cnt
            cnt = cnt + 1
        for i in range(ord('a'), ord('z')+1):
            self.table[i] = cnt
            cnt = cnt + 1
    # get: input how many imgs. flags="FULL" means all characters(number&letter), "ONLY_NUMBERS" means only numbers.
    # a function that return x-vector & y-vector 
    def get(self, size, flags="FULL"):
        img, codes = generator.getimgs(size)
        x = []
        y = []
        for i in range(size):
            x.append(imagetovec(imagesolve(img[i], self.mwidth, self.mheight), self.mwidth, self.mheight))
            size = 0
            if flags == "FULL":
                size = 10+26*2
            if flags == "ONLY_NUMBERS":
                size = 10
            sy = [0 for t in range(4*size)]
            for t in range(4):
                charact = codes[i][t]
                offset = self.table[ord(charact)]
                sy[t * size + offset] = 1
            y.append(sy)

        return x, y

# showImg: a function for debug, input a x-vector & y-vector & flags_info & width & height, 
# show the image & plaintext accordingly
def showImg(imglist, codelist, flags, width = 64, height = 40):
    img = Image.new('RGB', (width, height))
    for i in range(height):
        for j in range(width):
            img.putpixel((j,i), (imglist[i*width+j]*255,imglist[i*width+j]*255,imglist[i*width+j]*255))
    plaintext = ""
    if flags == "FULL":
        for tt in range(4):
            for i in range(10+26*2):
                if codelist[i+tt*(10+26*2)]:
                    if i <= 10:
                        plaintext = plaintext + str(i)
                    if i > 10 and i <= 10+26:
                        plaintext = plaintext + chr(i-10-1+ord('A'))
                    if i > 10*26:
                        plaintext = plaintext + chr(i-10-26-1+ord('a'))

    else:
        for tt in range(4):
            for i in range(10):
                if codelist[i+tt*10]:
                    plaintext = plaintext + str(i)

    plt.title(plaintext)
    plt.imshow(img)
    plt.show()
# a simple encapsulation
def ReadAll(width, height):
    res = dongzj(width, height)
    return res
# a function that get x-vector from file.
def GetFromFile(path, width, height):
    img = Image.open(path)
    img.resize((120, 80))
    img = imagesolve(img, width, height)
    img = imagetovec(img, width, height)
    return img

# usage example:
if __name__ == '__main__':
    x = GetFromFile("./Data/0yQPp.png", 120, 80)
    y = [ 0 for i in range(66)]
    showImg(x, y, "ONLY_NUMBERS", 120, 80)
    """
    width = 120
    height = 80
    input = ReadAll(width, height)
    x, y = input.get(10, "FULL")
    print(y[0])
    for i in range(4):
        showImg(x[i], y[i], "FULL", width, height)
    """
#usage over
