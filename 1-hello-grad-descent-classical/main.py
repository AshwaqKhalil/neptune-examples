from time import sleep
# from deepsense import neptune
import neptuner

# function y = f(x)
def f(x):
    return (x - 6)**2 + 9

# symbolic gradient
def df(x):
    return 2 * (x - 6)

# parameters
x = 0.         # initial x
lr = 0.1       # learning rate
n_steps = 30   # number of iterations

channel_x = neptuner.numeric_channel('x')
channel_y = neptuner.numeric_channel('y')

# searching for minimum with gradient descent
for step in range(n_steps):
    x -= lr * df(x)

    # logging new x and y
    channel_x.send(step, x)
    channel_y.send(step, f(x))

    # a crucial part of SleepyGradientDescent
    sleep(1)
