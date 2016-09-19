#
# Copyright (c) 2016, deepsense.io
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import time
from deepsense import neptune

ctx = neptune.Context()

amplitude = ctx.params.amplitude
sampling_rate = ctx.params.sampling_rate

sin_channel = ctx.job.create_channel(name='sin', channel_type=neptune.ChannelType.NUMERIC)
cos_channel = ctx.job.create_channel(name='cos', channel_type=neptune.ChannelType.NUMERIC)

logging_channel = ctx.job.create_channel(name='logging', channel_type=neptune.ChannelType.TEXT)

ctx.job.create_chart(name='sin & cos chart', series={'sin': sin_channel, 'cos': cos_channel})

ctx.job.finalize_preparation()

# The time interval between samples.
period = 1.0 / sampling_rate
# The initial timestamp, corresponding to x = 0 in the coordinate axis.
zero_x = time.time()

iteration = 0

while True:
    iteration += 1

    # Computes the values of sine and cosine.
    now = time.time()
    x = now - zero_x
    sin_y = amplitude * math.sin(x)
    cos_y = amplitude * math.cos(x)

    # Sends the computed values to the defined numeric channels.
    sin_channel.send(x=x, y=sin_y)
    cos_channel.send(x=x, y=cos_y)

    # Formats a logging entry.
    logging_entry = "sin({x})={sin_y}; cos({x})={cos_y}".format(x=x, sin_y=sin_y, cos_y=cos_y)

    # Sends a logging entry.
    logging_channel.send(x=iteration, y=logging_entry)

    time.sleep(period)