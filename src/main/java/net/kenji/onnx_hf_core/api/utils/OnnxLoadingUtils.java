package net.kenji.onnx_hf_core.api.utils;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import net.kenji.onnx_hf_core.api.AiVocab;
import net.kenji.onnx_hf_core.api.ModelTokenizer;
import net.kenji.onnx_hf_core.api.OrtSessionEnvironment;
import org.jline.utils.Log;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OnnxLoadingUtils {


    public static OrtSessionEnvironment startOrtEnvSession(Class<?> modelClass,String modelDir, Path outputDir, String mainFileName, String[] filesToCheck, boolean initializeTokenizer) {
        try {
            Files.createDirectories(outputDir);

            Map<String, Long> modelFiles = new HashMap<>();

            for (String fileName : filesToCheck) {
                try (InputStream stream = modelClass
                        .getResourceAsStream(modelDir + fileName)) {
                    if (stream != null) {
                        long size = stream.transferTo(OutputStream.nullOutputStream());
                        modelFiles.put(fileName, size);
                        Log.info("Resource found: " + fileName + " — " + size + " bytes");
                    }
                }
            }

            boolean allFilesReady = true; // <-- track success
            for (String fileName : modelFiles.keySet()) {
                Path outPath = outputDir.resolve(fileName);
                long fileSize = modelFiles.get(fileName);
                boolean needsExtract = !Files.exists(outPath)
                        || Files.size(outPath) == 0
                        || Files.size(outPath) != fileSize;

                if (needsExtract) {
                    Log.info("Extracting " + fileName + "...");
                    try (InputStream in = modelClass.getResourceAsStream(
                            modelDir + fileName)) {
                        if (in == null) {
                            Log.warn("Resource not found in jar: " + fileName);
                            allFilesReady = false; // <-- mark failure
                            continue;
                        }

                        copyWithProgress(in, outPath, fileSize, fileName);
                        long writtenSize = Files.size(outPath);
                        Log.info("Extracted " + fileName + " — " + writtenSize + " bytes");
                        if (writtenSize == 0) {
                            Log.warn("WARNING: " + fileName + " extracted as 0 bytes!");
                            allFilesReady = false;
                        }
                    }
                } else {
                    Log.info("Already extracted: " + fileName + " (" + Files.size(outPath) + " bytes)");
                }
            }

            // Wait until all files are valid (e.g. after first-run extraction)
            int maxWaitSeconds = 60;
            int waited = 0;
            List<String> validFileNames = new ArrayList<>();
            while (!allFilesReady) {
                if (waited >= maxWaitSeconds) {
                    Log.error("❌ Timed out waiting for model files to be ready.");
                    return null;
                }
                Log.info("⏳ Waiting for model files to be ready... (" + waited + "s)");
                Thread.sleep(1000);
                waited++;

                // Re-check all files
                allFilesReady = true;

                for (Map.Entry<String, Long> file : modelFiles.entrySet()) {
                    Path outPath = outputDir.resolve(file.getKey());
                    if (Files.exists(outPath) && Files.size(outPath) >= file.getValue()) {
                        validFileNames.add(file.getKey());
                        break;
                    }
                }
                if (validFileNames.size() < modelFiles.size()) {
                    allFilesReady = false;
                }
            }
            Path modelFile = outputDir.resolve(mainFileName + ".onnx");
            Log.info("Loading session from: " + modelFile.toAbsolutePath());

            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors());
            OrtSession session = env.createSession(modelFile.toString(), options);

            AiVocab vocab = loadVocab(outputDir);
            ModelTokenizer modelTokenizer = null;
            if(initializeTokenizer)
                modelTokenizer = new ModelTokenizer(outputDir);

            OrtSessionEnvironment ortSessionEnvironment = new OrtSessionEnvironment(env, session, modelTokenizer, vocab, outputDir);
            Log.info("✅ ONNX Runtime model loaded successfully");
            return ortSessionEnvironment;
        } catch (Exception e) {
            Log.error("❌ Failed to load ONNX model", e);
            e.printStackTrace();
        }
        Log.error("❌ Unexpectedly Failed to load ONNX model with no output error!!");
        return null;
    }

    private static void copyWithProgress(InputStream in, Path outPath, long totalSize, String fileName) throws IOException, IOException {
        try (OutputStream out = Files.newOutputStream(outPath)) {
            byte[] buffer = new byte[8192];
            long bytesCopied = 0;
            int lastLoggedPercent = -1;
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
                bytesCopied += read;
                if (totalSize > 0) {
                    int percent = (int) ((bytesCopied * 100) / totalSize);
                    int bucket = percent / 10 * 10; // log every 10%
                    if (bucket != lastLoggedPercent) {
                        Log.info("  Extracting " + fileName + "... " + bucket + "%");
                        lastLoggedPercent = bucket;
                    }
                }
            }
        }
    }

    private static AiVocab loadVocab(Path modelDir) {
        try {
            Map<Integer, String> vocabMap = new HashMap<>();
            Path vocabPath = modelDir.resolve("vocab.json");

            String json = Files.readString(vocabPath);

            // VERY SIMPLE PARSER (works for HuggingFace vocab.json)
            // format: "token": index OR index: token depending on file

            com.google.gson.JsonObject obj =
                    com.google.gson.JsonParser.parseString(json).getAsJsonObject();

            for (String key : obj.keySet()) {
                String value = obj.get(key).getAsString();

                try {
                    int index = Integer.parseInt(key);
                    vocabMap.put(index, value);
                } catch (NumberFormatException e) {
                    try {
                        int index = Integer.parseInt(value);
                        vocabMap.put(index, key);
                    } catch (Exception ignored) {}
                }
            }

            Log.info("Loaded vocab size: " + vocabMap.size());
            return new AiVocab(vocabMap);
        } catch (Exception e) {
            Log.error("Failed to load vocab", e);
        }
        Log.warn("WARNING: VOCAB FAILED TO LOAD! | AI Likely Won't work!");
        return null;
    }
}
