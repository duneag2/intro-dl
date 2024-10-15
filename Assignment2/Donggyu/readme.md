Answer of Question 2

1. print(np.sort(arr_mixed)[ : : -1])
2. print(np.argsort(arr_mixed)) # 정렬된 array의 index를 정렬 전의 index로부터 가져와서 배열한다.

Answer of Question 3

1. alpha는 투명도를 나타내고, cmap은 문자열로 특정 컬러맵을 지정하는 것이다.
2.
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

n = 20
x = np.random.rand(n)
y = np.random.rand(n)

plt.scatter(x, y, alpha = 0.1, cmap = 'Spectral')
plt.colorbar()
plt.show()

alpha = 0.5일때보다 조금 더 투명하다.
