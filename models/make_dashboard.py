"""
make_dashboard.py - train the three models, score the test windows, and emit a
self-contained dashboard.html that REPLAYS the real test windows as a live anomaly
stream (LSTM / BiLSTM / LogBERT), with the comparison table and ROC/PR curves.

Run on the laptop after the dataset is built:  python make_dashboard.py
Then open dashboard.html in any browser. No internet, no dependencies to serve.
"""
import json
import numpy as np
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, roc_curve, precision_recall_curve)
from admodel import config, data as D, train, evaluate

def collect():
    ds = D.load_dataset()
    vs, mask_id = ds["vocab_size"], ds["mask_id"]
    Xtr = ds["X_train"][ds["y_train"] == 0]
    Xva = ds["X_val"][ds["y_val"] == 0]
    out = {"models": {}, "test_n": int(len(ds["X_test"])),
           "attack_n": int(ds["y_test"].sum())}
    y_val, y_test = ds["y_val"].astype(int), ds["y_test"].astype(int)
    atk_test = [str(a) for a in ds["atk_test"]]
    for name in ("lstm", "bilstm", "logbert"):
        print(f"[dash] training {name} ...")
        model, _ = train.train_model(name, Xtr, Xva, vs, mask_id, seed=0)
        s_val = evaluate.window_scores(model, name, ds["X_val"], mask_id)
        s_test = evaluate.window_scores(model, name, ds["X_test"], mask_id)
        thr = evaluate.pick_threshold(y_val, s_val)
        pred = (s_test >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        tn = int(((pred==0)&(y_test==0)).sum()); fp = int(((pred==1)&(y_test==0)).sum())
        fpr = fp/(fp+tn) if (fp+tn) else 0.0
        roc = roc_curve(y_test, s_test); pr = precision_recall_curve(y_test, s_test)
        # normalise scores to 0..1 for display
        lo, hi = float(s_test.min()), float(s_test.max())
        norm = (s_test - lo) / (hi - lo + 1e-9)
        thr_n = (thr - lo) / (hi - lo + 1e-9)
        windows = [{"s": round(float(norm[i]), 4), "y": int(y_test[i]),
                    "a": atk_test[i]} for i in range(len(y_test))]
        per = {}
        for atk in sorted(set(atk_test) - {"benign"}):
            m = np.array([a == atk for a in atk_test])
            per[atk] = round(float(pred[m].mean()), 3) if m.sum() else 0.0
        out["models"][name] = {
            "metrics": {"precision": round(p,3), "recall": round(r,3), "f1": round(f1,3),
                        "fpr": round(fpr,3),
                        "roc_auc": round(roc_auc_score(y_test,s_test),3) if y_test.sum() else None,
                        "pr_auc": round(average_precision_score(y_test,s_test),3) if y_test.sum() else None,
                        "params": sum(p_.numel() for p_ in model.parameters())},
            "threshold": round(float(thr_n),4),
            "windows": windows,
            "per_attack": per,
            "roc": [[round(float(a),3), round(float(b),3)] for a,b in zip(roc[0], roc[1])],
            "pr":  [[round(float(a),3), round(float(b),3)] for a,b in zip(pr[1], pr[0])],
        }
    return out

HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AD Anomaly Detection — Live Monitor</title>
<style>
:root{--bg:#0b0f17;--panel:#121826;--line:#1f2a3c;--ink:#e6edf6;--sub:#8aa0bd;
--benign:#3b82f6;--attack:#f59e0b;--hit:#22c55e;--miss:#ef4444;--acc:#a78bfa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,Segoe UI,system-ui,Arial;font-size:14px}
.wrap{max-width:1080px;margin:0 auto;padding:22px}
h1{font-size:20px;margin:0 0 2px}.sub{color:var(--sub);font-size:13px;margin-bottom:18px}
.tabs{display:flex;gap:8px;margin:14px 0}
.tab{padding:8px 16px;border:1px solid var(--line);background:var(--panel);color:var(--sub);
border-radius:8px;cursor:pointer;font-weight:600}
.tab.on{color:var(--ink);border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
.row{display:flex;gap:16px;flex-wrap:wrap}.col{flex:1;min-width:300px}
canvas{width:100%;display:block;background:#0e1420;border-radius:8px}
.controls{display:flex;gap:10px;align-items:center;margin-top:10px}
button{background:var(--acc);color:#1a1033;border:0;padding:8px 14px;border-radius:8px;font-weight:700;cursor:pointer}
button.ghost{background:transparent;color:var(--sub);border:1px solid var(--line)}
.stats{display:flex;gap:18px;margin-top:12px;flex-wrap:wrap}
.stat{min-width:92px}.stat .v{font-size:22px;font-weight:700}.stat .l{color:var(--sub);font-size:12px}
table{width:100%;border-collapse:collapse}th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--sub);font-weight:600}td.best{color:var(--hit);font-weight:700}
.legend{display:flex;gap:16px;color:var(--sub);font-size:12px;margin-top:8px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:middle}
.bar{height:10px;border-radius:5px;background:var(--acc)}
small.note{color:var(--sub)}
</style></head><body><div class="wrap">
<h1>Active Directory Anomaly Detection — Live Monitor</h1>
<div class="sub">Replaying __TESTN__ held-out test windows (__ATTACKN__ attacks) through the trained models. Benign-trained; anomaly = max masked-key surprisal per window.</div>
<div class="tabs" id="tabs"></div>
<div class="panel">
  <canvas id="stream" height="240"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:var(--benign)"></span>benign window</span>
    <span><span class="dot" style="background:var(--attack)"></span>attack window</span>
    <span><span class="dot" style="background:var(--hit)"></span>detected (above threshold)</span>
    <span>— — threshold</span>
  </div>
  <div class="controls">
    <button id="play">▶ Play</button>
    <button class="ghost" id="reset">⟲ Reset</button>
    <span style="color:var(--sub)">speed</span>
    <input id="speed" type="range" min="1" max="60" value="24">
  </div>
  <div class="stats" id="stats"></div>
</div>
<div class="row">
  <div class="col panel"><b>Model comparison</b><div id="cmp"></div></div>
  <div class="col panel"><b>Per-attack recall</b><div id="per"></div></div>
</div>
<div class="row">
  <div class="col panel"><b>ROC curve (test)</b><canvas id="roc" height="260"></canvas></div>
  <div class="col panel"><b>Precision–Recall (test)</b><canvas id="pr" height="260"></canvas></div>
</div>
<small class="note">Generated from real model outputs on the frozen test split. LSTM / BiLSTM / LogBERT trained on benign windows only.</small>
</div>
<script>
const DATA = __DATA__;
const NAMES = ["lstm","bilstm","logbert"];
const LABEL = {lstm:"LSTM",bilstm:"BiLSTM",logbert:"LogBERT"};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let cur = "logbert", timer=null, idx=0;

const tabs = document.getElementById('tabs');
NAMES.forEach(n=>{const b=document.createElement('div');b.className='tab'+(n==cur?' on':'');
  b.textContent=LABEL[n];b.onclick=()=>{cur=n;document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  b.classList.add('on');resetStream();drawAll();};tabs.appendChild(b);});

function dpr(c){const r=c.getBoundingClientRect();const d=window.devicePixelRatio||1;
  c.width=r.width*d;c.height=c.height*d;const x=c.getContext('2d');x.scale(d,d);return[x,r.width,c.height/d];}

function drawStream(upto){
  const c=document.getElementById('stream');const [x,W,H]=dpr(c);
  const m=DATA.models[cur];const wins=m.windows;const n=wins.length;
  const padL=36,padB=18,plotH=H-padB-8;
  x.clearRect(0,0,W,H);
  // threshold line
  const ty=8+plotH*(1-m.threshold);
  x.strokeStyle=css('--sub');x.setLineDash([5,4]);x.beginPath();x.moveTo(padL,ty);x.lineTo(W-6,ty);x.stroke();x.setLineDash([]);
  x.fillStyle=css('--sub');x.font='11px Inter';x.fillText('thr',6,ty+3);
  x.fillText('anomaly score',6,14);
  const dx=(W-padL-8)/Math.max(n,1);
  let tp=0,fp=0,fn=0;
  for(let i=0;i<upto;i++){const w=wins[i];const px=padL+dx*i+dx/2;const py=8+plotH*(1-w.s);
    const det=w.s>=m.threshold;
    if(w.y){det?tp++:fn++;}else if(det){fp++;}
    let col=w.y?css('--attack'):css('--benign');if(det)col=css('--hit');
    x.fillStyle=col;x.beginPath();x.arc(px,py,w.y?4.5:2.6,0,7);x.fill();
    if(w.y){x.strokeStyle=css('--attack');x.globalAlpha=.5;x.beginPath();x.moveTo(px,8);x.lineTo(px,8+plotH);x.stroke();x.globalAlpha=1;}
  }
  // cursor
  if(upto<n){const cx=padL+dx*upto;x.strokeStyle=css('--acc');x.beginPath();x.moveTo(cx,8);x.lineTo(cx,8+plotH);x.stroke();}
  const rec=(tp+fn)?tp/(tp+fn):0, prec=(tp+fp)?tp/(tp+fp):0;
  document.getElementById('stats').innerHTML=
   stat(i0(upto)+' / '+n,'windows')+stat(tp,'detected',css('--hit'))+stat(fn,'missed',css('--miss'))
   +stat(fp,'false alarms',css('--attack'))+stat((rec*100).toFixed(0)+'%','recall')+stat((prec*100).toFixed(0)+'%','precision');
}
function i0(v){return v;}
function stat(v,l,c){return '<div class="stat"><div class="v" style="color:'+(c||css('--ink'))+'">'+v+'</div><div class="l">'+l+'</div></div>';}

function resetStream(){clearInterval(timer);timer=null;idx=0;document.getElementById('play').textContent='▶ Play';drawStream(0);}
function step(){const n=DATA.models[cur].windows.length;idx++;drawStream(idx);if(idx>=n){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ Play';}}
document.getElementById('play').onclick=()=>{const b=document.getElementById('play');
  if(timer){clearInterval(timer);timer=null;b.textContent='▶ Play';return;}
  if(idx>=DATA.models[cur].windows.length)idx=0;b.textContent='❚❚ Pause';
  const sp=+document.getElementById('speed').value;timer=setInterval(step,1000/sp);};
document.getElementById('reset').onclick=resetStream;
document.getElementById('speed').oninput=()=>{if(timer){clearInterval(timer);const sp=+document.getElementById('speed').value;timer=setInterval(step,1000/sp);}};

function drawCurve(id,pts,xlab,ylab,diag){const c=document.getElementById(id);const[x,W,H]=dpr(c);
  const p=34,pb=24;x.clearRect(0,0,W,H);x.strokeStyle=css('--line');x.strokeRect(p,6,W-p-8,H-pb-6);
  const X=v=>p+(W-p-8)*v, Y=v=>6+(H-pb-6)*(1-v);
  if(diag){x.strokeStyle=css('--sub');x.setLineDash([4,4]);x.beginPath();x.moveTo(X(0),Y(0));x.lineTo(X(1),Y(1));x.stroke();x.setLineDash([]);}
  x.strokeStyle=css('--acc');x.lineWidth=2;x.beginPath();pts.forEach((pt,i)=>{const fx=X(pt[0]),fy=Y(pt[1]);i?x.lineTo(fx,fy):x.moveTo(fx,fy);});x.stroke();
  x.fillStyle=css('--sub');x.font='11px Inter';x.fillText(xlab,W/2-20,H-6);
  x.save();x.translate(12,H/2+20);x.rotate(-Math.PI/2);x.fillText(ylab,0,0);x.restore();}

function drawAll(){
  drawStream(idx);
  const m=DATA.models[cur];
  drawCurve('roc',m.roc,'FPR','TPR',true);
  drawCurve('pr',m.pr,'Recall','Precision',false);
  // comparison table
  let best={}; ['f1','precision','recall','roc_auc','pr_auc'].forEach(k=>{best[k]=Math.max(...NAMES.map(n=>DATA.models[n].metrics[k]||0));});
  let h='<table><tr><th>Model</th><th>F1</th><th>Prec</th><th>Recall</th><th>FPR</th><th>ROC-AUC</th><th>PR-AUC</th></tr>';
  NAMES.forEach(n=>{const x=DATA.models[n].metrics;const cl=k=>x[k]==best[k]?' class="best"':'';
    h+='<tr><td>'+LABEL[n]+'</td><td'+cl('f1')+'>'+x.f1+'</td><td'+cl('precision')+'>'+x.precision+'</td><td'+cl('recall')+'>'+x.recall+'</td><td>'+x.fpr+'</td><td'+cl('roc_auc')+'>'+x.roc_auc+'</td><td'+cl('pr_auc')+'>'+x.pr_auc+'</td></tr>';});
  h+='</table>';document.getElementById('cmp').innerHTML=h;
  // per-attack bars
  const pa=m.per_attack;let ph='';Object.keys(pa).forEach(a=>{ph+='<div style="margin:8px 0"><div style="display:flex;justify-content:space-between"><span>'+a+'</span><span style="color:var(--sub)">'+(pa[a]*100).toFixed(0)+'%</span></div><div style="background:#0e1420;border-radius:5px"><div class="bar" style="width:'+(pa[a]*100)+'%"></div></div></div>';});
  document.getElementById('per').innerHTML=ph||'<small class="note">no attacks in test</small>';
}
window.addEventListener('resize',drawAll);
drawAll();
</script></body></html>"""

def main():
    data = collect()
    html = (HTML.replace("__DATA__", json.dumps(data))
                .replace("__TESTN__", str(data["test_n"]))
                .replace("__ATTACKN__", str(data["attack_n"])))
    open("dashboard.html","w",encoding="utf-8").write(html)
    print("[dash] wrote dashboard.html  (open it in any browser)")

if __name__ == "__main__":
    main()
