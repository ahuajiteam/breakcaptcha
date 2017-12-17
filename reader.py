# reader.py
# author: loriex
# time: 13:45 2017-12-17

#
#input = reader.ReadAll(path)
#	读取path下面所有的图片，返回一个类，支持以下方法：
#	input.total_numbers
#		图片总数
#	x, y = input.getone(i)
#		第i张图片 （从0开始）
#		x是一个[120*80]的一维列表。
#		y是一个[4*(10+26*2)]的列表，对应四个答案。这是一个只有四个1的0/1向量
#
#	X_batch, Y_batch = getbatch(i, BATCH_SIZE, width, height, flags)
# flags = "ONLY_NUMBERS" ==> Y only care numbers
#		i从0开始
#		读入第[i * BATCHSIZE, (i+1)*BATCHSIZE）的图片
#		X_batch是一个[BATCHSIZE, 120*80]的二维列表
#		Y_batch同理
#		i有可能很大。超过了就取个模，反正保证能够拿出BATCHSIZE个图片
#		我现在BATCHSIZE设的是100

import matplotlib.pyplot as plt
from PIL import Image
import os
import generator
import ClearNoise


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


def imagesolve(img, width, height):
    img = img.resize((width, height))
    img = image_binarize(img)
    return img

def imagetovec(img, width, height):
    x = [0 for i in range(width * height)]
    for i in range(height):
        for j in range(width):
            x[i*width+j] = img.getpixel((j,i))
    return x

class dongzj:
    mwidth = 0
    mheight = 0
    table = {}
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

def ReadAll(width, height):
    res = dongzj(width, height)
    return res
# usage example:
if __name__ == '__main__':
    width = 120
    height = 80
    input = ReadAll(width, height)
    x, y = input.get(10, "FULL")
    print(y[0])
    for i in range(4):
        showImg(x[i], y[i], "FULL", width, height)
#usage over
