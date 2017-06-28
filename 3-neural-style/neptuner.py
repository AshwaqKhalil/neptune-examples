# like Neptune but bettER

# In this way we loose the job offline parameters.
# However, they should be accessible from the yaml file even if we run the script by `python scriptname`.

from deepsense import neptune

import cv2
import PIL

ctx = neptune.Context()
params = ctx.params
register_action = ctx.job.register_action


def numeric_channel(channel_name, plot=True):
    channel = ctx.job.create_channel(name=channel_name, channel_type=neptune.ChannelType.NUMERIC)
    if plot:
        ctx.job.create_chart(name='{} chart'.format(channel_name), series={channel_name: channel})
    return channel


def image_channel(channel_name):
    channel = ctx.job.create_channel(name=channel_name, channel_type=neptune.ChannelType.IMAGE)
    channel.send_cv2 = lambda x, name, description, img: channel.send(x, neptune.Image(name=str(name), description=str(description), data=thumbnail_cv2(img)))
    return channel


def thumbnail_cv2(img, new_size=300):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_PIL = PIL.Image.fromarray(img)
    shrink_ratio = new_size / float(max(img.shape[:2]))
    if shrink_ratio < 1:
        thumbnail_size = int(round(shrink_ratio * img.shape[0])), int(round(shrink_ratio * img.shape[0]))
        img_PIL.thumbnail(thumbnail_size, PIL.Image.ANTIALIAS)
    return img_PIL