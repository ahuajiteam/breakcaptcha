# author: loriex
# for generate captcha img files.
# you can change some parameters
# minus:  the set of charaters need reducing probability
# probability[char] = : change the probability
# characters : the set of candidate charaters
# number: the amount of generate imgs

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string

# probability: a dictionary that record the occurrence probability of every character
probability = {}
# characters: a string that serve as a set of candidate character
characters = string.digits# + string.ascii_uppercase + string.ascii_lowercase
# width height: the size of captcha image.
# n_len: the length of captcha characters
# n_class: the numbers of candidate character
width, height, n_len, n_class = 120, 80, 4, len(characters)
# generator: client of lib captcha
generator = ImageCaptcha(width=width, height=height)

#--------a part that calc the probability--------------------------------------------------------
#------------------------------------------------------------------------------------------------
# if we want to delete some characters these code won't be influenced
def recede(char):
    if char in probability:
        probability[char] = 0.5;
# now set every character's rate
for i in characters:
    probability[i] = 1
# minus is a string serve as a set of characters which need to reduce probability
minus = "017oOlLI"
for i in range(0, len(minus)):
    recede(minus[i])
# calculate the prefix sum
for i in range(1, n_class):
    probability[characters[i]] = probability[characters[i-1]] + probability[characters[i]]
# calculate the occurrence probability
for i in range(0, n_class):
    probability[characters[i]] /= probability[characters[n_class-1]]
#print(probability)
#------------------------------------------------------------------------------------------------

# random_char: generate a character randomly according to the probability & characters
def random_char():
    char = random.random()
    for i in range(0, n_class):
        if char < probability[characters[i]]:
            return characters[i]
    return characters[n_class-1]

# add_noise_dots: add 10 size-5 noise dots with 'color' to 'img'
def add_noise_dots(img, color=-1):
    if color == -1:
        color = random.randint(0, 0xffffff)
    generator.create_noise_dots(img, color, 5, 10)

# add_noise_dots: add a noise curve with 'color' to 'img'
def add_noise_curve(img, color=-1):
    color = random.randint(0, 0xffffff)
    generator.create_noise_curve(img, color)

def getimgs(number = 200):
    imgs = []
    codes = []
    for num in range(0, number):
        random_str = ''
        for i in range(0, 4):
            random_str += random_char()
        img = generator.generate_image(random_str)
        imgs.append(img)
        codes.append(random_str)
    return imgs, codes
#-----------------------------now we go to generate a captcha------------------------------------
#------------------------------------------------------------------------------------------------
# random_str is the captcha's plaintext

if __name__ == '__main__':
    number = 100
    imgs, codes = getimgs(number)
    plt.title(codes[0])
    plt.imshow(imgs[0])
    plt.show()
    #plt.savefig("./sdata/"+random_str+str(num), format='png', dpi=300)
    #plt.clf()