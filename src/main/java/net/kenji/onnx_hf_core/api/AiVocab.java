package net.kenji.onnx_hf_core.api;

import java.util.Map;

public record AiVocab(Map<Integer, String> vocab) {
    public AiVocab(Map<Integer, String> vocab){
        this.vocab = vocab;
    }
}
