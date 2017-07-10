from time import sleep
# from deepsense import neptune
import neptuner

# function y = f(x)
def f(x):
    return x**2 - 14 * x + 7

# symbolic gradient
def df(x):
    return 2 * x - 14

# parameters
learning_rate = 0.1
n_steps = 30
x = 0.

channel_x = neptuner.numeric_channel('x')
channel_y = neptuner.numeric_channel('y')

# searching for minimum with gradient descent
for step in range(n_steps):
    x -= learning_rate * df(x)

    # logging new x and y
    channel_x.send(step, x)
    channel_y.send(step, f(x))

    # and a crucial part of the SleepyGradientDescent algorithm ;)
    sleep(1.)
