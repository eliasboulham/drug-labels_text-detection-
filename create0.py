import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import label_extr as ext
import extraction as ext

# Load the image
dircd = "datanew"


TS = ["Vignette", "LABORATIO", "BEKER", "Metronida", "BEKER",
      "TARIF", "DE", "Boite", "160.00 DA", "REF", "LOT", "RER", "FAB", "PPA"]
def get_font():
    fonts = []
    data_dir = Path("font/")
    images = sorted(list( list(data_dir.glob("*ttf"))))
    for i in images:
        fonts.append(str(i))


    return fonts
fonts = get_font()
# Define the text, font, and other properties
def my_fitting(cc, x=50, y=200):
    a, b, _ = cc.shape
    r = a / x
    w = b / r
    if (w < y):
        w = round(w)
        cc = cv2.resize(cc, (w, x))
        cc = cv2.copyMakeBorder(cc, 0, 0, 0, y - w, cv2.BORDER_CONSTANT,
                                value=[random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)])
    else:
        cc = cv2.resize(cc, (y, x))

    return cc
def noise_cam(img):
    #img = my_fitting(img)
    a, b, c = img.shape
    for v in range(a):
        for h in range(b):
            if (random.randint(0, 2)):
                if (np.mean(img[v, h]) < 155):
                    for i in range(c):
                        img[v, h, i] = np.uint8(img[v, h, i] + random.randint(0, 10))
                else:
                    for i in range(c):
                        img[v, h, i] = np.uint8(img[v, h, i] - random.randint(0, 100))

    return img
def get_image_text(text,i):
    font =  ImageFont.truetype(random.choice(fonts), random.randint(10,40))
    text_size = font.getsize(text)

    image = Image.new('RGB', text_size, color=(0, 0, 0))

    image = np.array(image)
    image = cv2.copyMakeBorder(image, 6, 6, 6,6, cv2.BORDER_CONSTANT, value=[ random.randint(200, 255),  random.randint(200, 255),  random.randint(200, 255)])

    rr1 = random.randint(0, 2)
    image[:, :, :] = random.randint(200, 255)
    if (rr1):
        r1 = random.randint(0, image.shape[1])
        r2 = random.randint(r1, image.shape[1])
        image[:, r1:r2] = [random.randint(0, 10), random.randint(20, 180), random.randint(0, 10)]
    rr1 = random.randint(0, 2)
    image[:, :, :] = random.randint(200, 255)
    if (rr1):
        r1 = random.randint(0, image.shape[1])
        r2 = random.randint(r1, image.shape[1])
        image[:, r1:r2] = [random.randint(0, 10), random.randint(20, 180), random.randint(0, 10)]
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    color = (random.randint(5, 40), random.randint(5, 40), random.randint(5, 40))
    draw.text((3, 3), text, fill=color, font=font)
    numpy_image = np.array(image)
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    #image = cv2.copyMakeBorder(image, 20, 20, 20,20, cv2.BORDER_CONSTANT, value=[random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)])
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), random.randint(-10,10)/10, 1)
    image = cv2.warpAffine(image, rotation_matrix, (width, height))


    rr=ext.get_textimages(image)
    if(len(rr)==1):
     return rr[0]
    else:
        #print(text)
        return None
def affine(img):
    x = random.randint(0, 0)  # Translation in the x direction
    y = random.randint(0, 0)  # Translation in the y direction
    x1 = random.uniform(-0.03, 0.03)  # Translation in the x direction
    y1 = random.uniform(-0.03, 0.03)
    x2 = random.uniform(0.95, 1.05)  # Translation in the x direction
    y2 = random.uniform(0.95, 1.05)

    add = random.randint(0, 5)
    M = np.float32([[x2, x1, add + x], [y1, y2, add + y]])

    # Apply the translation
    img_translation = cv2.warpAffine(img, M, (img.shape[1] + 5, img.shape[0] + 5))

    return img_translation







def get_image_mix(chars):
    lmax = 13

    rn = random.randint(0, lmax-1)
    s =""
    s0 =""
    for j in range(rn):

        char = random.choice(chars)
        c = random.choice(char)
        if (c == '/'):
            c0 = '@'
        elif (c == '.'):
            c0 = '$'
        else:
            c0 = c
        s = s + c
        s0 = s0 + c0
    l = len(s)
    img = get_image_text(s,i)
    if isinstance(img, np.ndarray):
        #img=my_fitting(img)
        for k in range(l, lmax):
            s0 = s0 + "#"
        print("1===",len(s0))
        img = noise_cam(img)
        cv2.imwrite(dircd + '/' + s0 + '.png', img)
def get_image_start_other(char_s, chars_o,i):
    lmax = 13
    rn = random.randint(0, lmax-1)
    s = random.choice(char_s)
    s0 = "" + s
    for j in range(rn):

        char = random.choice(chars_o)
        c = random.choice(char)
        if (c == '/'):
            c0 = '@'
        elif (c == '.'):
            c0 = '$'
        else:
            c0 = c

        s = s + c
        s0 = s0 + c0
    l = len(s)
    img = get_image_text(s,i)
    if isinstance(img, np.ndarray):
        #img=my_fitting(img)

        for k in range(l, lmax):
            s0 = s0 + "#"
        print("===",len(s0))

        img = noise_cam(img)


        cv2.imwrite(dircd + '/' + s0+ '.png', img)
def show(ss, img):
    cv2.imshow(ss, img)
    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ABC = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z']
abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
       'w', 'x', 'y', 'z']
other = ['*', '+', '-', '=', '/', '(', ')', ':', '%', ',', '.']
all = [num, ABC, abc, other]
aaa = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
       'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
       'U', 'V', 'W', 'X', 'Y', 'Z',
       'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
       'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
       '*', '+', '-', '=', '_', '/', '(', ')', ':', '%', ',', '.']
o = 1
for i in range(o):
    print(i)
    get_image_start_other(ABC, [abc],i)
    get_image_mix([abc])
    get_image_mix([num])
    get_image_mix([num,other])
    get_image_mix(all)
    get_image_mix(all)
    get_image_mix([ABC])








