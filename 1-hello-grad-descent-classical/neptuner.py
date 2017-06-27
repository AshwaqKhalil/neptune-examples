# like Neptune but bettER

from deepsense import neptune

ctx = neptune.Context()

def numeric_channel(name, plot=True):
    channel = ctx.job.create_channel(name=name, channel_type=neptune.ChannelType.NUMERIC)
    if plot:
        ctx.job.create_chart(name='{} chart'.format(name), series={name: channel})
    return channel
