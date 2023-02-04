"""Part 2 of Homework 0: basics of numpy."""

# Make sure that you have numpy installed correctly!
import numpy as np


## Create a random matrix W of size [20, 20] using np.random.
# TODO: Your code here.
W = np.random.rand(20,20)
print('Value of W:\n', W)
print()

## Create a vector x of size [20] of all ones. Create another vector b of size [20] using np.random.
# TODO: Your code here.
x = np.ones(20)
b = np.random.rand(20)
print('Value of x:\n', x)
print('Value of b:\n', b)
print()

## Matrix multiplication: Compute Wx + b, which is a vector of size [20].
# TODO: Your code here
ans = W@x + b
print('Matrix multiplication:\n', ans)
print()

## Stacking: Use np.stack to compute the [3, 3] matrix stacking vectors.
vectors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
# TODO: Your code here.
ans_stack = np.stack(vectors)
print('stacking:\n', ans_stack)