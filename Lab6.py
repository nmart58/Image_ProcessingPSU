import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utility import segmentation_utils

image = cv.imread('Task.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

plt.figure(figsize=(10, 14))
plt.imshow(image_rgb)
plt.title("Изображение с координатами")

plt.xlabel("X")
plt.ylabel("Y")

plt.xticks(np.arange(0, image.shape[1], 100))
plt.yticks(np.arange(0, image.shape[0], 100))

plt.grid(False)
plt.show()

seeds = [
    (492, 325),  
    (475, 399),  
    (495, 413),
    (439, 224),
    (506, 274),
    (568, 308),
    (530, 365),
    (524, 420),
    (486, 218),
    (412, 213),
    (368, 242),
    (369, 250),
    (497, 330),  
    (480, 404),  
    (500, 418),
    (444, 229),
    (511, 279),
    (573, 313),
    (535, 370),
    (529, 425),
    (491, 223),
    (417, 218),
    (373, 247),
    (374, 255),
    (580, 336),
    (577, 296),
    (554, 266),
    (527, 233),
    (506, 214),
    (472, 202),
    (448, 187),
    (542, 398),
    (548, 369),
]

threshold = 44


mask = segmentation_utils.region_growingHSV(image_hsv, seeds, threshold)
result = cv.bitwise_and(image, image, mask=mask)

plt.figure(figsize=(12, 10))
plt.subplot(1, 2, 1)
plt.title("Исходное изображение")
plt.imshow(image_rgb)

plt.subplot(1, 2, 2)
plt.title("Выделенный перец")
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))

plt.show()



