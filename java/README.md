# Neptune Examples - Java

## Getting Started
The goal of this example is to create a simple parameterizable job
that generates sine and cosine as functions of time (in seconds),
with the provided `amplitude` and `sampling_rate`.
The full description of the example can be found in the
<a target="_blank" href="https://neptune.deepsense.io/versions/latest/getting-started.html">documentation</a>.

### Build Commands

    cd getting-started
    gradle wrapper
    ./gradlew uberjar
    cp build/libs/getting-started.jar .

### Run Command

    neptune run --executable getting-started.jar
