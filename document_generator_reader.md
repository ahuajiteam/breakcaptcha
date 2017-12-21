## generator.py
我们使用了captcha库来生成验证码。该库的generate_image函数输入一个字符串即返回一个验证码。

用随机的方法来确定字符串。

为了对各个字符出现的概率进行有效控制，先把各个字符的概率存到了probability列表中，然后求了前缀和并除以总和，得到一个不降的序列。这样可以方便的通过随机一个$[0,1]$的浮点数的方式来随机字符。

为了对可选字符集合进行控制。用characters字符串来存储可选字符，与probability列表同时控制随机结果。

封装了getimgs函数，输入数目$n$，则随机生成$n$张验证码，返回一个两个列表，第一个列表是验证码图片列表，第二个列表是验证码字符串列表，长度均为输入数目。

由于该库生成的图片自带噪点和干扰线，为了生成无噪点无干扰线的图像，需要修改generate_image函数的代码，注释掉下段代码中被注释的两行。
```python
    def generate_image(self, chars):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = random_color(238, 255)
        color = random_color(0, 200, random.randint(220, 255))
        im = self.create_captcha_image(chars, color, background)
        #self.create_noise_dots(im, color)
        #self.create_noise_curve(im, color)
        im = im.filter(ImageFilter.SMOOTH)
        return im

```
有生成图片到文件的版本，生成了很多很多美丽动人的验证码，不过最后没用到，该版本就去掉了。

## reader.py
这是generator.py和identify.py的中间层。

封装了一些函数供identify.py使用

由于identify只想要01向量。所以本reader需要从generator获得图片然后降噪然后二值化然后转成01向量。

用class dongzj作为identify使用的窗口。封装了两个函数，1.初始化，2.获取向量。为了方便快速的在纯数字模式和数字字母混合模式之间切换，引入了flags参数进行控制。由于时间关系，本应该在generator上实现同样的控制，但并没有实现，因此改变flags参数的同时需要在generator上修改对应的代码，即characters

三个功能性函数

1. 二值化image_binarize，首先将图片转成灰度图，然后取各个像素点灰度的均值，然后判断比均值大的像素点少还是比均值小的像素点少，取像素点少的一侧赋值为1，其余赋值为0。其逻辑在于：验证码图片中验证码占比一般不到三分之一。
2. imagesolve，将一个原生的(120,80)的验证码图片降噪，然后拉伸至给定尺寸，然后二值化，返回二值化后的图
3. imagetovec，将一个二值化的图转化成一个01向量

一个用于检验生成的函数
showImg，将01向量转换成对应的图片和验证码字符串，并用matplotlib库显示出来。

一个激动人心的错误是在把图片转化为向量的时候把width和height打反了，这为我们带来了很多困扰，好在有showImg函数，我们没费多少功夫就找到了问题所在。

有从文件夹批量读取图片的版本，为了快速响应identify的请求还做了很多激烈的改动和优化，不过最后没用到，该版本就去掉了。

最后是一个简单的GetFromFile函数，输入路径返回经过一系列处理之后得到的x-vector，用于单个验证码的读取