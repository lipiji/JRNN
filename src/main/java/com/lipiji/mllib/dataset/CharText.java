package com.lipiji.mllib.dataset;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.jblas.DoubleMatrix;

public class CharText {
    private static final String LOCAL_DATA_PATH = "./data/";

    private Map<String, Integer> charIndex = new HashMap<>();
    private Map<Integer, String> indexChar = new HashMap<>();
    private Map<String, DoubleMatrix> charVector = new HashMap<>();
    private List<String> sequence = new ArrayList<>();
    
    public CharText() {
        init();
    }
    
    public void init() {
        loadData();
        buildDistributedRepresentations();
    }
    
    private void loadData() {
        try (BufferedReader reader = new BufferedReader(
                new FileReader(new File(LOCAL_DATA_PATH + "toy.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if(!StringUtils.isBlank(line)) {
                    sequence.add(line.toLowerCase());
                    for (char c : line.toLowerCase().toCharArray()) {
                        String key = String.valueOf(c);
                        if (!charIndex.containsKey(key)) {
                            charIndex.put(key, charIndex.size());
                            indexChar.put(charIndex.get(key), key);
                        }
                    }
                }
            }
        } catch (IOException e) {
        }        
    }
    
    private void buildDistributedRepresentations() {
        for (String c : charIndex.keySet()) {
            DoubleMatrix xt = DoubleMatrix.zeros(1, charIndex.size());
            xt.put(charIndex.get(c), 1);
            charVector.put(c, xt);
        }
    }
    
    public Map<String, Integer> getCharIndex() {
        return charIndex;
    }

    public Map<String, DoubleMatrix> getCharVector() {
        return charVector;
    }

    public List<String> getSequence() {
        return sequence;
    }
    
    public Map<Integer, String> getIndexChar() {
        return indexChar;
    }

    public static void main(String[] args) {
        CharText ct = new CharText();
        ct.init();
    }
}
