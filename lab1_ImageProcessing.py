import numpy as np

m = int(input("Введите количество строк матрицы (m): "))
n = int(input("Введите количество столбцов матрицы (n): "))

start = int(input("Введите начальное значение диапазона (start): "))
end = int(input("Введите конечное значение диапазона (end): "))

if start >= end:
    print("Ошибка: начальное значение должно быть меньше конечного!")
else:
    vector = np.linspace(start, end, n)
    print("\nСлучайно сгенерированный вектор:")
    print(vector)

    matrix = np.zeros((m, n), dtype=int) + vector

    print("\nСгенерированная матрица:")
    print(matrix)
