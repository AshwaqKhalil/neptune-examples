# like Neptune but bettER

from deepsense import neptune
from PIL import Image

ctx = neptune.Context()

def numeric_channel(name, plot=True):
    channel = ctx.job.create_channel(name=name, channel_type=neptune.ChannelType.NUMERIC)
    if plot:
        ctx.job.create_chart(name='{} chart'.format(name), series={name: channel})
    return channel

def image_channel(channel_name):
    channel = ctx.job.create_channel(name=channel_name, channel_type=neptune.ChannelType.IMAGE)
    channel.counter = 0

    def array_2d_to_image(array, autorescale=True):
        assert array.min() >= 0
        assert len(array.shape) == 2
        if array.max() <= 1 and autorescale:
            array = 255 * array
        array = array.astype('uint8')
        return Image.fromarray(array)

    def send_array(array_2d, name="", description=""):
        channel.send(channel.counter, neptune.Image(name=str(name), description=str(description), data=array_2d_to_image(array_2d)))
        channel.counter += 1

    channel.send_array = send_array

    return channel
