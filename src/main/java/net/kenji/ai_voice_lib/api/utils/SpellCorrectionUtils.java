package net.kenji.ai_voice_lib.api.utils;

import io.github.mightguy.spellcheck.symspell.api.DataHolder;
import io.github.mightguy.spellcheck.symspell.api.SpellChecker;
import io.github.mightguy.spellcheck.symspell.common.*;
import io.github.mightguy.spellcheck.symspell.exception.SpellCheckException;
import io.github.mightguy.spellcheck.symspell.impl.InMemoryDataHolder;
import io.github.mightguy.spellcheck.symspell.impl.SymSpellCheck;
import net.kenji.ai_voice_lib.OnnxHFCore;
import org.jline.utils.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.concurrent.*;

public class SpellCorrectionUtils {
    private static final ExecutorService executor = Executors.newSingleThreadExecutor();
    private static SpellCheckSettings settings = null;
    private static DataHolder dataHolder = null;
    private static SymSpellCheck spellChecker = null;
    private static volatile boolean ready = false;

    public static void initSpellCheck() {
        // RUN THIS SYNCHRONOUSLY on application startup to guarantee memory visibility across threads
        try {
            settings = SpellCheckSettings.builder()
                    .countThreshold(1)
                    .deletionWeight(1)
                    .insertionWeight(1)
                    .replaceWeight(1)
                    .maxEditDistance(2)
                    .transpositionWeight(1)
                    .topK(5)
                    .prefixLength(7) // Reset to standard 7
                    .verbosity(Verbosity.TOP) // Change to TOP for fast execution
                    .build();

            // Use standard stable distance implementation
            WeightedDamerauLevenshteinDistance distance =
                    new WeightedDamerauLevenshteinDistance(
                            settings.getDeletionWeight(),
                            settings.getInsertionWeight(),
                            settings.getReplaceWeight(),
                            settings.getTranspositionWeight(),
                            null);
            // Standard string hash mapping
            dataHolder = new InMemoryDataHolder(settings, new Murmur3HashFunction());
            spellChecker = new SymSpellCheck(dataHolder, distance, settings);

            loadDictionary();
            ready = true; // Safe to read now
            Log.info("SpellCheck: Ready, size=" + dataHolder.getSize());
        } catch (Exception e) {
            Log.error("SpellCheck init failed", e);
        }
    }

    public static void loadDictionary() throws IOException, SpellCheckException {
        Log.info("SpellCheck: Loading dictionary...");
        try (InputStream is = OnnxHFCore.class.getResourceAsStream(
                "/assets/onnx_hf_core/spell_correction/frequency_dictionary_80k.txt");
             BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] arr = line.trim().split("\\s+");
                if (arr.length >= 2) {
                    dataHolder.addItem(new DictionaryItem(arr[0], Double.parseDouble(arr[1]), -1.0));
                }
            }
        }
        Log.info("SpellCheck: Dictionary loaded");
    }

    public static String getCorrectionText(String rawText) throws SpellCheckException {
        if (!ready || rawText == null || rawText.trim().isEmpty()) return rawText;
        rawText = rawText.replaceAll("\\d+", "").replaceAll("\\s{2,}", " ").trim();

        String[] words = rawText.split("\\s+");
        StringBuilder correctedPhrase = new StringBuilder();

        for (String word : words) {
            String cleanWord = word.replaceAll("[^a-zA-Z0-9']", "").toLowerCase();

            if (cleanWord.isEmpty() || cleanWord.length() <= 4) {
                correctedPhrase.append(word).append(" ");
                continue;
            }

            // Only attempt correction if word is NOT already in dictionary
            Double freq = dataHolder.getItemFrequency(cleanWord);
            if (freq != null) {
                // Word is valid, keep as-is
                correctedPhrase.append(word).append(" ");
                continue;
            }

            // Word not in dictionary — attempt correction with timeout
            try {
                Future<List<SuggestionItem>> future = executor.submit(() ->
                        spellChecker.lookup(cleanWord, Verbosity.TOP, cleanWord.length() >= 8 ? 2 : 1)
                );
                List<SuggestionItem> suggestions = future.get(100, TimeUnit.MILLISECONDS); // 100ms max
                if (suggestions != null && !suggestions.isEmpty()) {
                    correctedPhrase.append(suggestions.get(0).getTerm()).append(" ");
                } else {
                    correctedPhrase.append(word).append(" ");
                }
            } catch (TimeoutException e) {
                Log.warn("SpellCheck timeout on: " + cleanWord);
                correctedPhrase.append(word).append(" "); // keep original on timeout
            } catch (Exception e) {
                correctedPhrase.append(word).append(" ");
            }
        }

        return correctedPhrase.toString().trim();
    }
}