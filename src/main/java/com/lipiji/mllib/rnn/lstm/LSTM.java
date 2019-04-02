package com.lipiji.mllib.rnn.lstm;

import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.lipiji.mllib.layers.MatIniter;
import com.lipiji.mllib.layers.MatIniter.Type;
import com.lipiji.mllib.utils.Activer;

public class LSTM implements Serializable {
    private static final long serialVersionUID = -7059290852389115565L;
    
    private int inSize;
    private int outSize;
    private int deSize;
    
    private DoubleMatrix Wxi;
    private DoubleMatrix Whi;
    private DoubleMatrix Wci;
    private DoubleMatrix bi;
    
    private DoubleMatrix Wxf;
    private DoubleMatrix Whf;
    private DoubleMatrix Wcf;
    private DoubleMatrix bf;
    
    private DoubleMatrix Wxc;
    private DoubleMatrix Whc;
    private DoubleMatrix bc;
    
    private DoubleMatrix Wxo;
    private DoubleMatrix Who;
    private DoubleMatrix Wco;
    private DoubleMatrix bo;
    
    private DoubleMatrix Why;
    private DoubleMatrix by;
    
    public LSTM(int inSize, int outSize, MatIniter initer) {
        this.inSize = inSize;
        this.outSize = outSize;
        
        if (initer.getType() == Type.Uniform) {
            this.Wxi = initer.uniform(inSize, outSize);
            this.Whi = initer.uniform(outSize, outSize);
            this.Wci = initer.uniform(outSize, outSize);
            this.bi = new DoubleMatrix(1, outSize);
            
            this.Wxf = initer.uniform(inSize, outSize);
            this.Whf = initer.uniform(outSize, outSize);
            this.Wcf = initer.uniform(outSize, outSize);
            this.bf = new DoubleMatrix(1, outSize);
            
            this.Wxc = initer.uniform(inSize, outSize);
            this.Whc = initer.uniform(outSize, outSize);
            this.bc = new DoubleMatrix(1, outSize);
            
            this.Wxo = initer.uniform(inSize, outSize);
            this.Who = initer.uniform(outSize, outSize);
            this.Wco = initer.uniform(outSize, outSize);
            this.bo = new DoubleMatrix(1, outSize);
            
            this.Why = initer.uniform(outSize, inSize);
            this.by = new DoubleMatrix(1, inSize);
        } else if (initer.getType() == Type.Gaussian) {
            this.Wxi = initer.gaussian(inSize, outSize);
            this.Whi = initer.gaussian(outSize, outSize);
            this.Wci = initer.gaussian(outSize, outSize);
            this.bi = new DoubleMatrix(1, outSize);
            
            this.Wxf = initer.gaussian(inSize, outSize);
            this.Whf = initer.gaussian(outSize, outSize);
            this.Wcf = initer.gaussian(outSize, outSize);
            this.bf = new DoubleMatrix(1, outSize);
            
            this.Wxc = initer.gaussian(inSize, outSize);
            this.Whc = initer.gaussian(outSize, outSize);
            this.bc = new DoubleMatrix(1, outSize);
            
            this.Wxo = initer.gaussian(inSize, outSize);
            this.Who = initer.gaussian(outSize, outSize);
            this.Wco = initer.gaussian(outSize, outSize);
            this.bo = new DoubleMatrix(1, outSize);
            
            this.Why = initer.gaussian(outSize, inSize);
            this.by = new DoubleMatrix(1, inSize);
        }
    }
    
    public LSTM(int inSize, int outSize, MatIniter initer, int deSize) {
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
        DoubleMatrix preH = null, preC = null;
        if (t == 0) {
            preH = new DoubleMatrix(1, getOutSize());
            preC = preH.dup();
        } else {
            preH = acts.get("h" + (t - 1));
            preC = acts.get("c" + (t - 1));
        }
        
        DoubleMatrix i = Activer.logistic(x.mmul(Wxi).add(preH.mmul(Whi)).add(preC.mmul(Wci)).add(bi));
        DoubleMatrix f = Activer.logistic(x.mmul(Wxf).add(preH.mmul(Whf)).add(preC.mmul(Wcf)).add(bf));
        DoubleMatrix gc = Activer.tanh(x.mmul(Wxc).add(preH.mmul(Whc)).add(bc));
        DoubleMatrix c = f.mul(preC).add(i.mul(gc));
        DoubleMatrix o = Activer.logistic(x.mmul(Wxo).add(preH.mmul(Who)).add(c.mmul(Wco)).add(bo));
        DoubleMatrix gh = Activer.tanh(c);
        DoubleMatrix h = o.mul(gh);
        
        acts.put("i" + t, i);
        acts.put("f" + t, f);
        acts.put("gc" + t, gc);
        acts.put("c" + t, c);
        acts.put("o" + t, o);
        acts.put("gh" + t, gh);
        acts.put("h" + t, h);
    }
    
    public void bptt(Map<String, DoubleMatrix> acts, int lastT, double lr) {
        for (int t = lastT; t > -1; t--) {
            DoubleMatrix py = acts.get("py" + t);
            DoubleMatrix y = acts.get("y" + t);
            DoubleMatrix deltaY = py.sub(y);
            acts.put("dy" + t, deltaY);
            
            // cell output errors
            DoubleMatrix h = acts.get("h" + t);
            DoubleMatrix deltaH = null;
            if (t == lastT) {
                deltaH = Why.mmul(deltaY.transpose()).transpose();
            } else {
                DoubleMatrix lateDgc = acts.get("dgc" + (t + 1));
                DoubleMatrix lateDf = acts.get("df" + (t + 1));
                DoubleMatrix lateDo = acts.get("do" + (t + 1));
                DoubleMatrix lateDi = acts.get("di" + (t + 1));
                deltaH = Why.mmul(deltaY.transpose()).transpose()
                        .add(Whc.mmul(lateDgc.transpose()).transpose())
                        .add(Whi.mmul(lateDi.transpose()).transpose())
                        .add(Who.mmul(lateDo.transpose()).transpose())
                        .add(Whf.mmul(lateDf.transpose()).transpose());
            }
            acts.put("dh" + t, deltaH);
            
            
            // output gates
            DoubleMatrix gh = acts.get("gh" + t);
            DoubleMatrix o = acts.get("o" + t);
            DoubleMatrix deltaO = deltaH.mul(gh).mul(deriveExp(o));
            acts.put("do" + t, deltaO);
            
            // status
            DoubleMatrix deltaC = null;
            if (t == lastT) {
                deltaC = deltaH.mul(o).mul(deriveTanh(gh))
                        .add(Wco.mmul(deltaO.transpose()).transpose());
            } else {
                DoubleMatrix lateDc = acts.get("dc" + (t + 1));
                DoubleMatrix lateDf = acts.get("df" + (t + 1));
                DoubleMatrix lateF = acts.get("f" + (t + 1));
                DoubleMatrix lateDi = acts.get("di" + (t + 1));
                deltaC = deltaH.mul(o).mul(deriveTanh(gh))
                        .add(Wco.mmul(deltaO.transpose()).transpose())
                        .add(lateF.mul(lateDc))
                        .add(Wcf.mmul(lateDf.transpose()).transpose())
                        .add(Wci.mmul(lateDi.transpose()).transpose());
            }
            acts.put("dc" + t, deltaC);
            
            // cells
            DoubleMatrix gc = acts.get("gc" + t);
            DoubleMatrix i = acts.get("i" + t);
            DoubleMatrix deltaGc = deltaC.mul(i).mul(deriveTanh(gc));
            acts.put("dgc" + t, deltaGc);
        
            DoubleMatrix preC = null;
            if (t > 0) {
                preC = acts.get("c" + (t - 1));
            } else {
                preC = DoubleMatrix.zeros(1, h.length);
            }
            // forget gates
            DoubleMatrix f = acts.get("f" + t);
            DoubleMatrix deltaF = deltaC.mul(preC).mul(deriveExp(f));
            acts.put("df" + t, deltaF);
        
            // input gates
            DoubleMatrix deltaI = deltaC.mul(gc).mul(deriveExp(i));
            acts.put("di" + t, deltaI);
        }
        updateParameters(acts, lastT, lr);
    }
    
    private void updateParameters(Map<String, DoubleMatrix> acts, int lastT, double lr) {
        DoubleMatrix gWxi = new DoubleMatrix(Wxi.rows, Wxi.columns);
        DoubleMatrix gWhi = new DoubleMatrix(Whi.rows, Whi.columns);
        DoubleMatrix gWci = new DoubleMatrix(Wci.rows, Wci.columns);
        DoubleMatrix gbi = new DoubleMatrix(bi.rows, bi.columns);
        
        DoubleMatrix gWxf = new DoubleMatrix(Wxf.rows, Wxf.columns);
        DoubleMatrix gWhf = new DoubleMatrix(Whf.rows, Whf.columns);
        DoubleMatrix gWcf = new DoubleMatrix(Wcf.rows, Wcf.columns);
        DoubleMatrix gbf = new DoubleMatrix(bf.rows, bf.columns);
        
        DoubleMatrix gWxc = new DoubleMatrix(Wxc.rows, Wxc.columns);
        DoubleMatrix gWhc = new DoubleMatrix(Whc.rows, Whc.columns);
        DoubleMatrix gbc = new DoubleMatrix(bc.rows, bc.columns);
        
        DoubleMatrix gWxo = new DoubleMatrix(Wxo.rows, Wxo.columns);
        DoubleMatrix gWho = new DoubleMatrix(Who.rows, Who.columns);
        DoubleMatrix gWco = new DoubleMatrix(Wco.rows, Wco.columns);
        DoubleMatrix gbo = new DoubleMatrix(bo.rows, bo.columns);
        
        DoubleMatrix gWhy = new DoubleMatrix(Why.rows, Why.columns);
        DoubleMatrix gby = new DoubleMatrix(by.rows, by.columns);
        
        for (int t = 0; t < lastT + 1; t++) {
            DoubleMatrix x = acts.get("x" + t).transpose();
            gWxi = gWxi.add(x.mmul(acts.get("di" + t)));
            gWxf = gWxf.add(x.mmul(acts.get("df" + t)));
            gWxc = gWxc.add(x.mmul(acts.get("dgc" + t)));
            gWxo = gWxo.add(x.mmul(acts.get("do" + t)));
            
            if (t > 0) {
                DoubleMatrix preH = acts.get("h" + (t - 1)).transpose();
                DoubleMatrix preC = acts.get("c" + (t - 1)).transpose();
                gWhi = gWhi.add(preH.mmul(acts.get("di" + t)));
                gWhf = gWhf.add(preH.mmul(acts.get("df" + t)));
                gWhc = gWhc.add(preH.mmul(acts.get("dgc" + t)));
                gWho = gWho.add(preH.mmul(acts.get("do" + t)));
                gWci = gWci.add(preC.mmul(acts.get("di" + t)));
                gWcf = gWcf.add(preC.mmul(acts.get("df" + t)));
            }
            gWco = gWco.add(acts.get("c" + t).transpose().mmul(acts.get("do" + t)));
            gWhy = gWhy.add(acts.get("h" + t).transpose().mmul(acts.get("dy" + t)));
            
            gbi = gbi.add(acts.get("di" + t));
            gbf = gbf.add(acts.get("df" + t));
            gbc = gbc.add(acts.get("dgc" + t));
            gbo = gbo.add(acts.get("do" + t));
            gby = gby.add(acts.get("dy" + t));
        }
        
        Wxi = Wxi.sub(clip(gWxi.div(lastT)).mul(lr));
        Whi = Whi.sub(clip(gWhi.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
        Wci = Wci.sub(clip(gWci.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
        bi = bi.sub(clip(gbi.div(lastT)).mul(lr));
        
        Wxf = Wxf.sub(clip(gWxf.div(lastT)).mul(lr));
        Whf = Whf.sub(clip(gWhf.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
        Wcf = Wcf.sub(clip(gWcf.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
        bf = bf.sub(clip(gbf.div(lastT)).mul(lr));
        
        Wxc = Wxc.sub(clip(gWxc.div(lastT)).mul(lr));
        Whc = Whc.sub(clip(gWhc.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
        bc = bc.sub(clip(gbc.div(lastT)).mul(lr));

        Wxo = Wxo.sub(clip(gWxo.div(lastT)).mul(lr));
        Who = Who.sub(clip(gWho.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
        Wco = Wco.sub(clip(gWco.div(lastT)).mul(lr));
        bo = bo.sub(clip(gbo.div(lastT)).mul(lr));
        
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