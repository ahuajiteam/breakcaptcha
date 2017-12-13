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


# the next two lines is what
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'


# probability: a dictionary that record the occurrence probability of every character
probability = {}
# characters: a string that serve as a set of candidate character
characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
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
print(probability)
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

#-----------------------------now we go to generate a captcha------------------------------------
#------------------------------------------------------------------------------------------------
# random_str is the captcha's plaintext
number = 10000
for num in range(0, number):
    random_str = ''
    for i in range(0, 4):
        random_str += random_char()
    #random_str = "1liI"
    #generator.create_captcha_image(characters)
    #img = generator.create_captcha_image(random_str, 0xffffff, 0x0)
    img = generator.generate_image(random_str)
    #color = random.randint(0, 0xffffff)
    #for i in range(1, 5):
    #    add_noise_dots(img, color)

    img.save("./ndata/"+random_str+str(num)+'.png', format='png')
#    plt.title(random_str)
#    plt.imshow(img)
#    plt.savefig("./ndata/"+random_str+str(num), format='png', dpi=300)
#    plt.clf()