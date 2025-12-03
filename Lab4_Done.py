import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


img = cv2.imread('my_picture.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 2. Гистограмма без учета белого фона

non_white_pixels = gray[gray < 255]
plt.figure(figsize=(6,4))
plt.hist(non_white_pixels.reshape(-1), 256, (0, 255), color='gray')
plt.title("Гистограмма без белого фона")
plt.xlabel("Яркость (0–254)")
plt.ylabel("Количество пикселей")
plt.show()

# 3. Удаляем серые линии (яркость 120–135)
cleaned_gray = gray.copy()

gray_min, gray_max = 120, 135
mask_gray_lines = (cleaned_gray >= gray_min) & (cleaned_gray <= gray_max)

cleaned_gray[mask_gray_lines] = 255  # заменяем серые линии на белый


_, thresh = cv2.threshold(cleaned_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


cv2.imwrite('result_no_gray_lines.jpg', thresh)
Image.open('result_no_gray_lines.jpg').show()
