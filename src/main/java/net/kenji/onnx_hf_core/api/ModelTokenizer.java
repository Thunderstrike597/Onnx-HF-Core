package net.kenji.onnx_hf_core.api;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.nio.file.Files;
import java.nio.file.Path;

import static com.mojang.text2speech.Narrator.LOGGER;

public class ModelTokenizer {
    private HuggingFaceTokenizer tokenizer;
    private static final int EOS_TOKEN = 50256;

    public ModelTokenizer(Path tokenizerDirectory) {
        try {
            if(Files.exists(tokenizerDirectory.resolve("tokenizer.json"))) {
                tokenizer = HuggingFaceTokenizer.newInstance(tokenizerDirectory);

                LOGGER.info("Tokenizer loaded successfully!");
            }
        } catch (Exception e) {
            LOGGER.error("Failed to load tokenizer", e);
        }
    }

    public long[] encode(String text) {
        return tokenizer.encode(text).getIds();
    }

    public String decode(long[] tokens) {
        // Decode but remove special tokens
        String text = tokenizer.decode(tokens, true); // true = skip special tokens
        return text;
    }

    public int getEosToken() {
        return EOS_TOKEN;
    }
}