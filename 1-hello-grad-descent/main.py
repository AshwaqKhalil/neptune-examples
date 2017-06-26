from time import sleep

# function y = f(x)
def f(x):
    return (x - 42)**2 + 137

# symbolic gradient
def df(x):
    return 2 * (x - 42)

# parameters
x = 0.         # initial x
lr = 0.1       # learning rate
n_steps = 30   # number of iterations

# searching for minimum with gradient descent
for step in range(n_steps):
    x -= lr * df(x)

    # printing new x and y
    print("x: {}".format(x))
    print("y: {}".format(f(x)))

    # a crucial part of SleepyGradientDescent
    sleep(0.5)
