package net.kenji.onnx_hf_core.api.speech_management;

public class AudioFeatureExtractor {

    // SenseVoice Baseline Configuration Hyperparameters
    private static final int SAMPLE_RATE = 16000;
    private static final int MEL_BINS = 80;
    private static final int FRAME_LENGTH_SAMPLES = 400; // 25ms @ 16kHz
    private static final int FRAME_SHIFT_SAMPLES = 160;  // 10ms @ 16kHz
    private static final int FFT_SIZE = 512;             // Next power of 2 for 400 samples

    public static float[][] computeMelFbank(float[] audio) {
        if (audio.length < FRAME_LENGTH_SAMPLES) {
            return new float[0][MEL_BINS];
        }

        // 1. Calculate Frame Boundaries
        int numFrames = 1 + (audio.length - FRAME_LENGTH_SAMPLES) / FRAME_SHIFT_SAMPLES;
        float[][] melFeatures = new float[numFrames][MEL_BINS];

        // 2. Pre-calculate the Mel-filterbank Weights Matrix
        float[][] melFilters = createMelFilterbank(FFT_SIZE / 2 + 1, MEL_BINS, SAMPLE_RATE);

        // 3. Process Frame Windows Step-by-Step
        float[] frameBuffer = new float[FRAME_LENGTH_SAMPLES];
        for (int f = 0; f < numFrames; f++) {
            int startSample = f * FRAME_SHIFT_SAMPLES;
            System.arraycopy(audio, startSample, frameBuffer, 0, FRAME_LENGTH_SAMPLES);

            // Apply Pre-emphasis Filter and Hamming Window function
            float lastSample = (startSample > 0) ? audio[startSample - 1] : 0.0f;
            for (int i = 0; i < FRAME_LENGTH_SAMPLES; i++) {
                float current = frameBuffer[i];
                float emphasized = current - 0.97f * lastSample; // Pre-emphasis
                lastSample = current;
                
                // Hamming Window implementation
                double windowValue = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (FRAME_LENGTH_SAMPLES - 1));
                frameBuffer[i] = (float) (emphasized * windowValue);
            }

            // Execute Fast Fourier Transform (FFT)
            float[] fftOutputComplex = computeFFT(frameBuffer, FFT_SIZE);

            // Compute Magnitude Spectrum
            int numBins = FFT_SIZE / 2 + 1;
            float[] powerSpectrum = new float[numBins];
            for (int i = 0; i < numBins; i++) {
                float real = fftOutputComplex[2 * i];
                float imag = fftOutputComplex[2 * i + 1];
                powerSpectrum[i] = (real * real + imag * imag) / FRAME_LENGTH_SAMPLES;
            }

            // Map Power Spectrum to the 80 Mel Bins and Apply Log Scaling
            for (int m = 0; m < MEL_BINS; m++) {
                float melSum = 0.0f;
                for (int i = 0; i < numBins; i++) {
                    melSum += powerSpectrum[i] * melFilters[m][i];
                }
                // Convert to Log energy with a tiny flooring value to avoid log(0) exceptions
                melFeatures[f][m] = (float) Math.log(Math.max(melSum, 1e-5f));
            }
        }
        return melFeatures;
    }

    private static float[][] createMelFilterbank(int numBins, int melBins, int sampleRate) {
        float[][] filters = new float[melBins][numBins];
        double minMel = 0.0;
        double maxMel = 2595.0 * Math.log10(1.0 + (sampleRate / 2.0) / 700.0);

        double[] melPoints = new double[melBins + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = minMel + i * (maxMel - minMel) / (melBins + 1);
        }

        int[] binPoints = new int[melBins + 2];
        for (int i = 0; i < binPoints.length; i++) {
            double freq = 700.0 * (Math.pow(10.0, melPoints[i] / 2595.0) - 1.0);
            binPoints[i] = (int) Math.floor((FFT_SIZE + 1) * freq / sampleRate);
        }

        for (int m = 1; m <= melBins; m++) {
            int startBin = binPoints[m - 1];
            int centerBin = binPoints[m];
            int endBin = binPoints[m + 1];

            for (int i = startBin; i < centerBin; i++) {
                if (centerBin != startBin) filters[m - 1][i] = (float) (i - startBin) / (centerBin - startBin);
            }
            for (int i = centerBin; i < endBin; i++) {
                if (endBin != centerBin) filters[m - 1][i] = (float) (endBin - i) / (endBin - centerBin);
            }
        }
        return filters;
    }

    private static float[] computeFFT(float[] input, int fftSize) {
        float[] complexData = new float[fftSize * 2];
        for (int i = 0; i < input.length; i++) {
            complexData[2 * i] = input[i]; // Real part
        }
        
        // In-place Radix-2 Cooley-Tukey FFT implementation
        int n = fftSize;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j >= bit; bit >>= 1) j -= bit;
            j += bit;
            if (i < j) {
                float tempReal = complexData[2 * i];
                float tempImag = complexData[2 * i + 1];
                complexData[2 * i] = complexData[2 * j];
                complexData[2 * i + 1] = complexData[2 * j + 1];
                complexData[2 * j] = tempReal;
                complexData[2 * j + 1] = tempImag;
            }
        }

        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2 * Math.PI / len;
            float wlenReal = (float) Math.cos(ang);
            float wlenImag = (float) -Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                float wReal = 1.0f;
                float wImag = 0.0f;
                for (int j = 0; j < len / 2; j++) {
                    int uIdx = 2 * (i + j);
                    int vIdx = 2 * (i + j + len / 2);
                    
                    float uReal = complexData[uIdx];
                    float uImag = complexData[uIdx + 1];
                    
                    float vReal = complexData[vIdx] * wReal - complexData[vIdx + 1] * wImag;
                    float vImag = complexData[vIdx] * wImag + complexData[vIdx + 1] * wReal;
                    
                    complexData[uIdx] = uReal + vReal;
                    complexData[uIdx + 1] = uImag + vImag;
                    complexData[vIdx] = uReal - vReal;
                    complexData[vIdx + 1] = uImag - vImag;
                    
                    float nextWReal = wReal * wlenReal - wImag * wlenImag;
                    wImag = wReal * wlenImag + wImag * wlenReal;
                    wReal = nextWReal;
                }
            }
        }
        return complexData;
    }
}
