/**
 * Copyright 2017, deepsense.io
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package io.deepsense.neptune.gettingstarted;

import io.deepsense.neptune.clientlibrary.NeptuneContextFactory;
import io.deepsense.neptune.clientlibrary.models.Channel;
import io.deepsense.neptune.clientlibrary.models.ChartSeriesCollection;
import io.deepsense.neptune.clientlibrary.models.NeptuneContext;

import java.util.HashMap;
import java.util.Map;

public class GettingStarted {
    public static void main(String[] args) throws InterruptedException {
        NeptuneContext context = NeptuneContextFactory.createContext(args);

        double amplitude = context.getParams().get("amplitude").getValue().asDouble().get();
        double samplingRate = context.getParams().get("samplingRate").getValue().asDouble().get();

        Channel<Double> sinChannel = context.getJob().createNumericChannel("sin");
        Channel<Double> cosChannel = context.getJob().createNumericChannel("cos");

        Channel<String> loggingChannel = context.getJob().createTextChannel("logging");

        ChartSeriesCollection series = context.getJob().createChartSeriesCollection();
        series.add("sin", sinChannel);
        series.add("cos", cosChannel);

        context.getJob().createChart("sin & cos chart", series);

        long period = (long) (1000.0 / samplingRate);

        double zeroX = System.nanoTime() / 1e9;

        int iteration = 0;

        while (true) {
            iteration += 1;

            double x = System.nanoTime() / 1e9 - zeroX;

            double sinY = amplitude * Math.sin(x);
            double cosY = amplitude * Math.cos(x);

            sinChannel.send(x, sinY);
            cosChannel.send(x, cosY);

            loggingChannel.send(
                    iteration,
                    String.format("sin(%f)=%f; cos(%f)=%f",
                            x, sinY, x, cosY));

            Thread.sleep(period);
        }

    }
}
