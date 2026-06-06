package net.kenji.ai_voice_lib.api;

import java.util.Map;

public record AiVocab(Map<Integer, String> vocab) {
    public AiVocab(Map<Integer, String> vocab){
        this.vocab = vocab;
    }
}
