# How to Run "Getting Started" Example

1. Install Gradle Wrapper.

    `gradle wrapper`

2. Build a self-contained jar for the project.

    `./gradlew uberjar`

3. Copy the resulting jar from `build/libs` to the project's main directory.

    `cp build/libs/getting-started.jar .`

4. Run the job.

    `neptune run --executable getting-started.jar`
