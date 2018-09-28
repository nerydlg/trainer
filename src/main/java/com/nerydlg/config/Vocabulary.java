package com.nerydlg.config;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.NoSuchFileException;
import java.util.Random;

public class Vocabulary {

    private static final Logger log = LoggerFactory.getLogger(Vocabulary.class);
    private String fileInput;
    private String fileoutput;
    private Word2Vec word2Vec;

    public Vocabulary(String fileInput, String fileoutput) {
        this.fileInput = fileInput;
        this.fileoutput = fileoutput;
    }

    public void generateVocab(int minWordFrequency, int iterations, int layerSize,
                              int seed, int windowSize) throws FileNotFoundException, NoSuchFileException {
        File input = returnFileIfExists(fileInput);

        log.info("Log and vectorize sentences...");
        SentenceIterator sentenceIterator = new BasicLineIterator(input);
        TokenizerFactory tokenizer = new DefaultTokenizerFactory();

        tokenizer.setTokenPreProcessor(new CommonPreprocessor());
        log.info("Build Model ....");

        word2Vec = new Word2Vec.Builder()
                .minWordFrequency(minWordFrequency)
                .iterations(iterations)
                .layerSize(layerSize)
                .seed(seed)
                .windowSize(windowSize)
                .iterate(sentenceIterator)
                .tokenizerFactory(tokenizer)
                .build();
    }

    public Word2Vec getWord2Vec() {
        return word2Vec;
    }

    private File returnFileIfExists(String file) throws FileNotFoundException, NoSuchFileException {
        log.info("Checking if file exists: {}", file);
        if(file == null) {
            throw new NoSuchFileException("The file input is needed to generate the vocab");
        }

        File inputF = new File(file);
        if(!inputF.exists()) {
            throw new FileNotFoundException("File not found: " + file);
        }
        return inputF;
    }
}
