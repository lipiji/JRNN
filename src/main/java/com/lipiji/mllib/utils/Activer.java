package com.lipiji.mllib.utils;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class Activer {
    
    public static double logistic(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    public static double tanh(double x) {
        double e = Math.exp(2 * x);
        return (e - 1) / (e + 1);
    }
    
    //public static double tanh(double x) {
    //    return 2 * logistic(2 * x) - 1;
    //}
        
    public static DoubleMatrix logistic(DoubleMatrix X) {
        return MatrixFunctions.pow(MatrixFunctions.exp(X.mul(-1)).add(1), -1);
    }
    
    public static DoubleMatrix tanh(DoubleMatrix X) {
        return MatrixFunctions.tanh(X);
    }
    
    public static DoubleMatrix ReLU(DoubleMatrix X) {
        DoubleMatrix pIndex = X.gt(0);
        return X.mul(pIndex);
    }
    
    // rows: samples
    public static DoubleMatrix softmax(DoubleMatrix X) {
        DoubleMatrix expM = MatrixFunctions.exp(X);
        for (int i = 0; i < X.rows; i++) {
            DoubleMatrix expMi = expM.getRow(i);
            expM.putRow(i, expMi.div(expMi.sum()));
        }
        return expM;
    }
}
