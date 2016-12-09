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
