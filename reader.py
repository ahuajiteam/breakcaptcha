# reader.py
# author: loriex
# time: 23:02 2017-12-15

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

    def get(self, i, width, height):
        if i >= self.total:
            i = i % self.total
        T = PngMat()
        path = self.filepath + self.filelist[i]
        #print(path)
        img = Image.open(path)
        img = ClearNoise.pre_image(img) #image_denoising
        img = img.resize((width, height))
        img = image_binarize(img)
        T.width = img.size[0]
        T.height = img.size[1]
        T.img = img
        T.plaintext = self.filelist[i][0:4]
        return T


class dongzj:
    total_numbers = 0
    mreader = 0
    table = {}
    def __init__(self, path):
        self.mreader = PngReader(path)
        self.total_numbers = self.mreader.total
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
    def getone(self, idx, width = 64, height = 40, flags="FULL"):
        if idx >= self.total_numbers:
            id = idx % self.total_numbers
        T = self.mreader.get(idx, width, height)

        x = [0 for i in range(width * height)]
        for i in range(height):
            for j in range(width):
                x[i*width+j] = T.img.getpixel((j,i))

        size = 0
        if flags == "FULL":
            size = 10+26*2
        if flags == "ONLY_NUMBERS":
            size = 10
        y = [0 for i in range(4*size)]
        for i in range(4):
            charact = T.plaintext[i]
            offset = self.table[ord(charact)]
            y[i * size + offset] = 1

        return x, y

    def getbatch(self, id, BATCH_SIZE, width = 64, height = 40, flags="FULL"):
        x = [[] for i in range(BATCH_SIZE)]
        y = [[] for i in range(BATCH_SIZE)]
        id = id * BATCH_SIZE
        for i in range(0, BATCH_SIZE):
            id = id % self.total_numbers
            x[i], y[i] = self.getone(id, width, height, flags)
            id = id + 1
        return x, y

def showImg(imglist, width = 64, height = 40):
    img = Image.new('RGB', (width, height))
    for i in range(height):
        for j in range(width):
            img.putpixel((j,i), (imglist[i*width+j]*255,imglist[i*width+j]*255,imglist[i*width+j]*255))

    plt.title("oh my god")
    plt.imshow(img)
    plt.show()

def ReadAll(path):
    res = dongzj(path)
    return res
# usage example:
if __name__ == '__main__':
    width = 64
    height = 40
    input = ReadAll("./ndata")
    x, y = input.getbatch(1, 1, width, height, "FULL")
    print(input.total_numbers)
    print(y[0])
    for i in range(1):
        showImg(x[i], width, height)
#usage over
