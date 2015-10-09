package com.lipiji.mllib.layers;

import java.util.Random;

import org.jblas.DoubleMatrix;

public class MatIniter {
    private static Random random = new Random();

    public enum Type {
        Uniform, Gaussian
    }
    private Type type;
    private double scale = 0.01;
    private double miu = 0;
    private double sigma = 0.01;

    public MatIniter(Type type, double scale, double miu, double sigma) {
        this.type = type;
        this.scale = scale;
        this.miu = miu;
        this.sigma = sigma;
    }
    
    public DoubleMatrix uniform(int rows, int cols) {
        return DoubleMatrix.rand(rows, cols).mul(2 * scale).sub(scale);
    }
    
    public DoubleMatrix gaussian(int rows, int cols) {
        DoubleMatrix m = new DoubleMatrix(rows, cols);
        for (int i = 0; i < m.length; i++) {
            m.put(i, random.nextGaussian() * sigma + miu);
        }
        return m;
    }

    public Type getType() {
        return type;
    }

    public double getScale() {
        return scale;
    }

    public double getMiu() {
        return miu;
    }

    public double getSigma() {
        return sigma;
    }
}