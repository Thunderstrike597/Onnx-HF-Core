package net.kenji.ai_voice_lib.api;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.nio.file.Path;
import java.util.Map;

public record OrtSessionEnvironment(OrtEnvironment env, OrtSession session, ModelTokenizer tokenizer, AiVocab vocab, Path modelDir) {
    public OrtSessionEnvironment(OrtEnvironment env, OrtSession session, ModelTokenizer tokenizer, AiVocab vocab, Path modelDir){
        this.env = env;
        this.session = session;
        this.tokenizer = tokenizer;
        this.vocab = vocab;
        this.modelDir = modelDir;
    }
    public boolean isEnvironmentLoaded(){
        return this.env != null && this.session != null && this.vocab != null && this.modelDir != null;
    }
    public Map<Integer, String> getVocab(){
        return this.vocab.vocab();
    }
}
