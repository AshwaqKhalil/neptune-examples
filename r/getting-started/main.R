#
# Copyright (c) 2017, deepsense.io
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

library(neptune)

samplingRate <- params("sampling_rate")
amplitude <- params("amplitude")

createNumericChannel("sin")
createNumericChannel("cos")

createTextChannel("logging")

createChart(chartName = "sin & cos chart", series = list("my_sin" = "sin", "my_cos" = "cos"))

period <- 1.0 / samplingRate
zeroX <- unclass(Sys.time())

iteration <- 0

while (TRUE) {
  iteration <- iteration + 1

  now <- unclass(Sys.time())
  x <- now - zeroX

  sinY <- amplitude * sin(x)
  cosY <- amplitude * cos(x)

  send("sin", x, sinY)
  send("cos", x, cosY)

  loggingEntry <- paste("sin(", x, ")=", sinY, "; cos(", x, ")=", cosY)
  send("logging", iteration, loggingEntry)

  Sys.sleep(period)
}
