from time import sleep
from deepsense import neptune

ctx = neptune.Context()


# I really recommend adding functions like that,
# which will greately reduce amount of code (and visual noise)
# ideally:
# neptune.numeric_channel
# WITHOUT the need of
# ctx = neptune.Context()
def numeric_channel(name, plot=True):
    channel = ctx.job.create_channel(name=name, channel_type=neptune.ChannelType.NUMERIC)
    if plot:
        ctx.job.create_chart(name='{} chart'.format(name), series={name: channel})
    return channel


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

channel_x = numeric_channel('x')
channel_y = numeric_channel('y')


# searching for minimum with gradient descent
for step in range(n_steps):
    x -= lr * df(x)

    # printing new x and y
    channel_x.send(step, x)
    channel_y.send(step, f(x))

    # a crucial part of SleepyGradientDescent
    sleep(0.5)
