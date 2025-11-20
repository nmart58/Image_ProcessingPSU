import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Метрики PSNR и SSIM

def getPSNR(I1, I2):
    s1 = cv.absdiff(I1, I2)
    s1 = np.float32(s1)
    sse = np.sum(s1 * s1)
    if sse <= 1e-10:
        return 0
    mse = sse / (I1.shape[0] * I1.shape[1] * I1.shape[2])
    return 10 * np.log10((255 * 255) / mse)

def getSSIM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225

    I1 = np.float32(i1)
    I2 = np.float32(i2)
    I1_2 = I1 * I1
    I2_2 = I2 * I2
    I1_I2 = I1 * I2

    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)

    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5) - mu1_2
    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5) - mu2_2
    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5) - mu1_mu2

    t1 = (2 * mu1_mu2 + C1)*(2 * sigma12 + C2)
    t2 = (mu1_2 + mu2_2 + C1)*(sigma1_2 + sigma2_2 + C2)
    ssim_map = cv.divide(t1, t2)

    ssim = cv.mean(ssim_map)
    return ssim[:3]



img = cv.imread('Image-85-0d0e70.jpg')
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)


rgb_image = 255 - rgb


# Медианная фильтрация разных размеров
median3 = cv.medianBlur(rgb, 3)
hsv = cv.cvtColor(median3, cv.COLOR_RGB2HSV)
h, s, v = cv.split(hsv)
s = cv.multiply(s, 1.4)  # повышаем насыщенность 
hsv = cv.merge([h, s, v])
bright = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
#bright = cv.convertScaleAbs(median3, alpha=1.2, beta=-20)



# Метрики качества

psnr_noisy = getPSNR(rgb, rgb_image)
ssim_noisy = getSSIM(rgb, rgb_image)

psnr3 = getPSNR(rgb, median3)


ssim3 = getSSIM(rgb, median3)


# Вывод изображений

plt.figure(figsize=(16, 8))

plt.subplot(231)
plt.imshow(rgb)
plt.title("Исходное изображение")
plt.axis("off")

plt.subplot(232)
plt.imshow(rgb_image)
plt.title("Инвертированное изображение")
plt.axis("off")

plt.subplot(234)
plt.imshow(median3)
plt.title("Медианный фильтр 3х3")
plt.axis("off")

plt.subplot(235)
plt.imshow(bright)
plt.title("Улучшенное изображение")
plt.axis("off")


plt.tight_layout()
plt.show()
