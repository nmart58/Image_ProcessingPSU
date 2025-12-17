import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

image = cv2.imread('task.jpg')


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)


lower_purple = np.array([0, 220, 80])  
upper_purple = np.array([15, 260, 260])  

mask = cv2.inRange(image_hsv, lower_purple, upper_purple)

result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Исходное изображение')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Красная маска')

plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title('Выделенный элемент')

r, g, b = cv2.split(image_hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
pixel_colors = image_rgb.reshape((np.shape(image_rgb)[0]*np.shape(image_rgb)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Value")
plt.show()


