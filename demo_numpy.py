import numpy as np

# schnell
# speichereffizient
# vektorwertig

my_list = [1, 2, 3]
my_list + 1

my_array = np.array([1, 2, 3])
my_array - 1

my_array.mean()
my_array.min()
my_array.max()

my_array.shape
my_array = my_array.reshape(1, 3)
my_array.shape


my_array = np.array([[1, 2, 3], [4, 5, 6]])
my_array.shape

my_array = my_array.reshape(3, 2)
print(my_array)

my_array.dtype
my_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
my_array.dtype