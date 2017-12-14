#pre_image()返回一个降噪完成的image
from PIL import Image,ImageDraw,ImageChops
import os

# 验证码预处理降噪
# 预处理结束后返回0/255二值图像
def pre_image(image):
    # 将图片转换成灰度图片
    image = image.convert("L")

    # 二值化,得到0/255二值图片
    # 阀值threshold = 180
    image = iamge2imbw(image,200)

    # 对二值图片进行降噪
    # N = 4
    clear_noise(image,2)
    return image

# 灰度图像二值化,返回0/255二值图像
def iamge2imbw(image,threshold):
    # 设置二值化阀值
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    # 像素值变为0,1
    image = image.point(table,'1')

    # 像素值变为0,255
    image = image.convert('L')
    return image

# 根据一个点A的灰度值(0/255值),与周围的8个点的值比较
# 降噪率N: N=1,2,3,4,5,6,7
# 当A的值与周围8个点的相等数小于N时,此点为噪点
# 如果确认是噪声,用该点的上面一个点的值进行替换
def get_near_pixel1(image,x,y,N):
    pix = image.getpixel((x,y))
    k = 3
    near_dots = 0
    if pix == image.getpixel((x - 3,y - 3)):
        near_dots += 1
    if pix == image.getpixel((x - 3,y)):
        near_dots += 1
    if pix == image.getpixel((x - 3,y + 3)):
        near_dots += 1
    if pix == image.getpixel((x,y - 3)):
        near_dots += 1
    if pix == image.getpixel((x,y + 3)):
        near_dots += 1
    if pix == image.getpixel((x + 3,y - 3)):
        near_dots += 1
    if pix == image.getpixel((x + 3,y)):
        near_dots += 1
    if pix == image.getpixel((x + 3,y + 3)):
        near_dots += 1

    if near_dots < N:
        # 确定是噪声,用上面一个点的值代替
        return image.getpixel((x,max(0,y-3)))
    else:
        return None
def get_near_pixel2(image,x,y,N):
    pix = image.getpixel((x,y))
    near_dots = 0
    if pix == image.getpixel((x - 1,y - 1)):
        near_dots += 1
    if pix == image.getpixel((x - 1,y)):
        near_dots += 1
    if pix == image.getpixel((x - 1,y + 1)):
        near_dots += 1
    if pix == image.getpixel((x,y - 1)):
        near_dots += 1
    if pix == image.getpixel((x,y + 1)):
        near_dots += 1
    if pix == image.getpixel((x + 1,y - 1)):
        near_dots += 1
    if pix == image.getpixel((x + 1,y)):
        near_dots += 1
    if pix == image.getpixel((x + 1,y + 1)):
        near_dots += 1

    if near_dots < N:
        # 确定是噪声,用上面一个点的值代替
        return image.getpixel((x,max(0,y-3)))
    else:
        return None

# 降噪处理
def clear_noise(image,N):
    draw = ImageDraw.Draw(image)

    # 外面一圈变白色
    Width,Height=image.size
    for x in range(Width):
        draw.point((x,0),255)
        draw.point((x,Height-1),255)
    for y in range(Height):
        draw.point((0,y),255)
        draw.point((Width-1,y),255)

    # 内部降噪
    for x in range(3,Width - 3):
        for y in range(3,Height - 3):
            color = get_near_pixel1(image,x,y,N)
            if color != None:
                draw.point((x,y),color)
    for x in range(3,Width - 3):
        for y in range(3,Height - 3):
            color = get_near_pixel2(image,x,y,4)
            if color != None:
                draw.point((x,y),color)



def main():
    for cur_num in range(0,10):

        image = Image.open("./"+str(cur_num)+'.png')


        image=pre_image(image)

        image.save("./"+'result'+str(cur_num)+'.png',format='png')


if __name__ == '__main__':
    main()
