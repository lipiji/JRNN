package com.lipiji.mllib.rnn.gru;

import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.lipiji.mllib.layers.MatIniter;
import com.lipiji.mllib.layers.MatIniter.Type;
import com.lipiji.mllib.utils.Activer;

public class GRU implements Serializable {
    private static final long serialVersionUID = -1501734916541393551L;

    private int inSize;
    private int outSize;
    private int deSize;
    
    private DoubleMatrix Wxr;
    private DoubleMatrix Whr;
    private DoubleMatrix br;
    
    private DoubleMatrix Wxz;
    private DoubleMatrix Whz;
    private DoubleMatrix bz;
    
    private DoubleMatrix Wxh;
    private DoubleMatrix Whh;
    private DoubleMatrix bh;
    
    private DoubleMatrix Why;
    private DoubleMatrix by;
    
    public GRU(int inSize, int outSize, MatIniter initer) {
        this.inSize = inSize;
        this.outSize = outSize;
        
        if (initer.getType() == Type.Uniform) {
            this.Wxr = initer.uniform(inSize, outSize);
            this.Whr = initer.uniform(outSize, outSize);
            this.br = new DoubleMatrix(1, outSize);
            
            this.Wxz = initer.uniform(inSize, outSize);
            this.Whz = initer.uniform(outSize, outSize);
            this.bz = new DoubleMatrix(1, outSize);
            
            this.Wxh = initer.uniform(inSize, outSize);
            this.Whh = initer.uniform(outSize, outSize);
            this.bh = new DoubleMatrix(1, outSize);
            
            this.Why = initer.uniform(outSize, inSize);
            this.by = new DoubleMatrix(1, inSize);
        } else if (initer.getType() == Type.Gaussian) {
        }
    }
    
    public GRU(int inSize, int outSize, MatIniter initer, int deSize) {
        this(inSize, outSize, initer);
        this.deSize = deSize;
        this.Why = new DoubleMatrix(outSize, deSize);
        this.by = new DoubleMatrix(1, deSize);
    }
    
    public int getInSize() {
        return inSize;
    }

    private int getOutSize() {
        return outSize;
    }
    
    public int getDeSize() {
        return deSize;
    }
    
    public void active(int t, Map<String, DoubleMatrix> acts) {
        DoubleMatrix x = acts.get("x" + t);
        DoubleMatrix preH = null;
        if (t == 0) {
            preH = new DoubleMatrix(1, getOutSize());
        } else {
            preH = acts.get("h" + (t - 1));
        }
        
        DoubleMatrix r = Activer.logistic(x.mmul(Wxr).add(preH.mmul(Whr)).add(br));
        DoubleMatrix z = Activer.logistic(x.mmul(Wxz).add(preH.mmul(Whz)).add(bz));
        DoubleMatrix gh = Activer.tanh(x.mmul(Wxh).add(r.mul(preH).mmul(Whh)).add(bh));
        DoubleMatrix h = z.mul(preH).add((DoubleMatrix.ones(1, z.columns).sub(z)).mul(gh));
        
        acts.put("r" + t, r);
        acts.put("z" + t, z);
        acts.put("gh" + t, gh);
        acts.put("h" + t, h);
    }
    
    public void bptt(Map<String, DoubleMatrix> acts, int lastT, double lr) {
        for (int t = lastT; t > -1; t--) {
            // 1 /2 || y - py ||^2
            // model output errors:  error = -(y - py) 
            DoubleMatrix py = acts.get("py" + t);
            DoubleMatrix y = acts.get("y" + t);
            DoubleMatrix deltaY = py.sub(y);//.mul(deriveExp(py));
            acts.put("dy" + t, deltaY);
            
            // cell output errors
            DoubleMatrix h = acts.get("h" + t);
            DoubleMatrix z = acts.get("z" + t);
            DoubleMatrix r = acts.get("r" + t);
            DoubleMatrix gh = acts.get("gh" + t);
            
            DoubleMatrix deltaH = null;
            if (t == lastT) {
                deltaH = Why.mmul(deltaY.transpose()).transpose();
            } else {
                DoubleMatrix lateDh = acts.get("dh" + (t + 1));
                DoubleMatrix lateDgh = acts.get("dgh" + (t + 1));
                DoubleMatrix lateDr = acts.get("dr" + (t + 1));
                DoubleMatrix lateDz = acts.get("dz" + (t + 1));
                DoubleMatrix lateR = acts.get("r" + (t + 1));
                DoubleMatrix lateZ = acts.get("z" + (t + 1));
                deltaH = Why.mmul(deltaY.transpose()).transpose()
                        .add(Whr.mmul(lateDr.transpose()).transpose())
                        .add(Whz.mmul(lateDz.transpose()).transpose())
                        .add(Whh.mmul(lateDgh.mul(lateR).transpose()).transpose())
                        .add(lateDh.mul(lateZ));
            }
            acts.put("dh" + t, deltaH);
            
            // gh
            DoubleMatrix deltaGh = deltaH.mul(DoubleMatrix.ones(1, z.columns).sub(z)).mul(deriveTanh(gh));
            acts.put("dgh" + t, deltaGh);
            
            DoubleMatrix preH = null;
            if (t > 0) {
                preH = acts.get("h" + (t - 1));
            } else {
                preH = DoubleMatrix.zeros(1, h.length);
            }
            
            // reset gates
            DoubleMatrix deltaR = (Whh.mmul(deltaGh.mul(preH).transpose()).transpose()).mul(deriveExp(r));
            acts.put("dr" + t, deltaR);
            
            // update gates
            DoubleMatrix deltaZ = deltaH.mul(preH.sub(gh)).mul(deriveExp(z));
            acts.put("dz" + t, deltaZ);            
        }
        updateParameters(acts, lastT, lr);
    }
    
    private void updateParameters(Map<String, DoubleMatrix> acts, int lastT, double lr) {
        DoubleMatrix gWxr = new DoubleMatrix(Wxr.rows, Wxr.columns);
        DoubleMatrix gWhr = new DoubleMatrix(Whr.rows, Whr.columns);
        DoubleMatrix gbr = new DoubleMatrix(br.rows, br.columns);
        
        DoubleMatrix gWxz = new DoubleMatrix(Wxz.rows, Wxz.columns);
        DoubleMatrix gWhz = new DoubleMatrix(Whz.rows, Whz.columns);
        DoubleMatrix gbz = new DoubleMatrix(bz.rows, bz.columns);
        
        DoubleMatrix gWxh = new DoubleMatrix(Wxh.rows, Wxh.columns);
        DoubleMatrix gWhh = new DoubleMatrix(Whh.rows, Whh.columns);
        DoubleMatrix gbh = new DoubleMatrix(bh.rows, bh.columns);
        
        DoubleMatrix gWhy = new DoubleMatrix(Why.rows, Why.columns);
        DoubleMatrix gby = new DoubleMatrix(by.rows, by.columns);
        
        for (int t = 0; t < lastT + 1; t++) {
            DoubleMatrix x = acts.get("x" + t).transpose();
            gWxr = gWxr.add(x.mmul(acts.get("dr" + t)));
            gWxz = gWxz.add(x.mmul(acts.get("dz" + t)));
            gWxh = gWxh.add(x.mmul(acts.get("dgh" + t)));
            
            if (t > 0) {
                DoubleMatrix preH = acts.get("h" + (t - 1)).transpose();
                gWhr = gWhr.add(preH.mmul(acts.get("dr" + t)));
                gWhz = gWhz.add(preH.mmul(acts.get("dz" + t)));
                gWhh = gWhh.add(acts.get("r" + t).transpose().mul(preH).mmul(acts.get("dgh" + t)));
            }
            gWhy = gWhy.add(acts.get("h" + t).transpose().mmul(acts.get("dy" + t)));
            
            gbr = gbr.add(acts.get("dr" + t));
            gbz = gbz.add(acts.get("dz" + t));
            gbh = gbh.add(acts.get("dgh" + t));
            gby = gby.add(acts.get("dy" + t));
        }
        
        Wxr = Wxr.sub(clip(gWxr.div(lastT)).mul(lr));
        Whr = Whr.sub(clip(gWhr.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
        br = br.sub(clip(gbr.div(lastT)).mul(lr));
        
        Wxz = Wxz.sub(clip(gWxz.div(lastT)).mul(lr));
        Whz = Whz.sub(clip(gWhz.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
        bz = bz.sub(clip(gbz.div(lastT)).mul(lr));
        
        Wxh = Wxh.sub(clip(gWxh.div(lastT)).mul(lr));
        Whh = Whh.sub(clip(gWhh.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
        bh = bh.sub(clip(gbh.div(lastT)).mul(lr));
        
        Why = Why.sub(clip(gWhy.div(lastT)).mul(lr));
        by = by.sub(clip(gby.div(lastT)).mul(lr));
    }
    
    private DoubleMatrix deriveExp(DoubleMatrix f) {
        return f.mul(DoubleMatrix.ones(1, f.length).sub(f));
    }
    
    private DoubleMatrix deriveTanh(DoubleMatrix f) {
        return DoubleMatrix.ones(1, f.length).sub(MatrixFunctions.pow(f, 2));
    }
    
    private DoubleMatrix clip(DoubleMatrix x) {
        //double v = 10;
        //return x.mul(x.ge(-v).mul(x.le(v)));
        return x;
    }
    
    public DoubleMatrix decode (DoubleMatrix ht) {
        return Activer.softmax(ht.mmul(Why).add(by));
    }
}
