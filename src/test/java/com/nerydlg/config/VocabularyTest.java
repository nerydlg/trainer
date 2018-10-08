package com.nerydlg.config;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.List;

public class VocabularyTest {

    private Vocabulary vocab;
    private Word2Vec word2Vec;
    private Logger log = LoggerFactory.getLogger(VocabularyTest.class);

    @Before
    public void init() {
        vocab = new Vocabulary("/home/nery.delgado/Documents/conference/input2.txt",
                "/home/nery.delgado/Documents/conference/output1.bin");
        try {
            vocab.generateVocab(3, 20, 128, 1234, 50);
            word2Vec = vocab.getWord2Vec();
            word2Vec.fit();
        } catch(Exception ex) {
            ex.printStackTrace();
        }
    }

    @Test
    public void testVocabulary_1() {
        wordNearestTest("quijote", 10);
    }

    @Test
    public void testVocabulary_2() {
        wordNearestTest("sancho", 4);
    }

    @Test
    public void testVocabulary_3() {
        wordNearestTest("caballo", 2);
    }

    private void wordNearestTest(String word, int nearest) {
        log.info(" Looking for word: "+word);
        Collection<String> words = word2Vec.wordsNearest(word, nearest);
        words.stream().forEach(System.out::println);
        Assert.assertEquals(nearest, words.size());
    }


}
