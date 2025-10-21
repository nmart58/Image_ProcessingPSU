import numpy as np
import cv2
import matplotlib.pyplot as plt

# Загружаем изображение и преобразуем в оттенки серого
image = cv2.imread('./lenna.png')

#OpenCV хранит цветные изображения в порядке каналов BGR (не RGB)
#Результат gray_image — двумерный массив формы (H, W) с типом uint8 (значения яркости в диапазоне 0–255).
#Преобразует цветное изображение в оттенки серого.
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Вычисляем гистограмму яркости
# Результат calcHist по умолчанию имеет форму (256, 1)
# .flatten() превращает это в плоский массив формы (256,
# hist[i] — количество пикселей с яркостью i
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()

# вычисляет кумулятивную сумму CDF
# cdf[i] = hist[0] + hist[1] + ... + hist[i]
# cdf[-1] — общее количество пикселей в изображении (равно H * W), если маска не используется
cdf = np.cumsum(hist)

# Нормализация CDF: делим каждый элемент cdf на общее количество пикселей cdf[-1]
# В результате cdf_normalized — массив значений в диапазоне [0.0, 1.0]
# Фактически оценка накопленной вероятности для каждого уровня яркости
cdf_normalized = cdf / cdf[-1]

# Вычисляем LUT по формуле из теории
# Создаём LUT (look-up table) — таблицу преобразования значений яркости:
# 255 * cdf_normalized переводит значения из [0,1] в [0,255]
# np.round(...) — округляем до ближайшего целого.
# .astype(np.uint8) — приводим к типу uint8, чтобы значения LUT были корректными индексами/пикселями.
# lut — массив длины 256, где lut[i] — новое значение яркости для всех пикселей со старой яркостью i.
lut = np.round(255 * cdf_normalized).astype(np.uint8)

# Применение LUT к изображению: индексирование массива.
# gray_image содержит значения 0..255; при использовании как индекса lut[gray_image]
# NumPy создаёт новое 2D-изображение той же формы (H, W),
# где каждый пиксель p заменён на lut[p].
# Это быстрый векторизованный способ применить преобразование ко всем пикселям без циклов.
equalized_image = lut[gray_image]

# Визуализация
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('После эквализации')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.hist(gray_image.reshape(-1), 256, range=(0, 255), color='gray')
plt.title('Гистограмма исходного изображения')
plt.xlabel('Яркость')
plt.ylabel('Частота')

plt.subplot(2, 2, 4)
plt.hist(equalized_image.reshape(-1), 256, range=(0, 255), color='gray')
plt.title('Гистограмма после эквализации')
plt.xlabel('Яркость')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()
