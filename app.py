# -*- coding: utf-8 -*-
"""UTAYA v27 | 機関投資家・マルチアセット歪みランキング＆90%複利自律AIエンジン。利益保証なし。"""
from __future__ import annotations
import json, re, os, time
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="UTAYA v27 | 機関マルチアセットAI", page_icon="⚡", layout="wide")

SYMBOLS = {
    "BTC": {"ticker":"BTC-USD","range":50.0},
    "ETH": {"ticker":"ETH-USD","range":15.0},
    "SOL": {"ticker":"SOL-USD","range":2.0},
    "XRP": {"ticker":"XRP-USD","range":0.05},
    "BNB": {"ticker":"BNB-USD","range":5.0},
    "WTI原油": {"ticker":"CL=F","range":0.40},
    "金先物": {"ticker":"GC=F","range":5.0},
    "日経平均": {"ticker":"^N225","range":100.0},
    "S&P 500": {"ticker":"^GSPC","range":15.0},
    "Nasdaq 100": {"ticker":"^NDX","range":50.0},
    "USD/JPY": {"ticker":"USDJPY=X","range":0.15},
    "米10年債利回り": {"ticker":"^TNX","range":0.05},
}
DEFAULT_DIR = Path(r"E:\utaya_3tb_ai_storage\utaya_v27")
ROOT = Path(os.environ.get("UTAYA_DATA_DIR", str(DEFAULT_DIR)))
HISTORY, MODELS, LOGS = ROOT/"history", ROOT/"models", ROOT/"logs"
for p in (ROOT,HISTORY,MODELS,LOGS): p.mkdir(parents=True, exist_ok=True)

def slug(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", SYMBOLS[name]["ticker"])

@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(ticker, period, interval):
    path = HISTORY / f"{ticker.replace('^','').replace('=','_')}_{interval}.csv"
    try:
        d = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                        progress=False, threads=False, timeout=10)
        if d is not None and not d.empty:
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            required = ["Open","High","Low","Close"]
            if not all(c in d.columns for c in required): raise ValueError("OHLC missing")
            cols = required + (["Volume"] if "Volume" in d.columns else [])
            d = d[cols].copy()
            if "Volume" not in d: d["Volume"] = np.nan
            d = d.apply(pd.to_numeric, errors="coerce").dropna(subset=required)
            d.index = pd.to_datetime(d.index)
            if getattr(d.index, "tz", None) is not None: d.index = d.index.tz_localize(None)
            d = d[(d.Close > 0) & (d.High >= d.Low)]
            d = d[~d.index.duplicated(keep="last")].sort_index()
            if path.exists():
                try:
                    old = pd.read_csv(path, index_col=0, parse_dates=True)
                    old.index = pd.to_datetime(old.index)
                    if getattr(old.index, "tz", None) is not None: old.index = old.index.tz_localize(None)
                    d = pd.concat([old,d]); d = d[~d.index.duplicated(keep="last")].sort_index()
                except Exception: pass
            d.to_csv(path, encoding="utf-8-sig")
            return d
    except Exception:
        pass
    if path.exists():
        try:
            d = pd.read_csv(path,index_col=0,parse_dates=True)
            d.index = pd.to_datetime(d.index)
            if getattr(d.index,"tz",None) is not None: d.index=d.index.tz_localize(None)
            return d.sort_index()
        except Exception: pass
    return pd.DataFrame()

def make_features(d):
    if d.empty: return pd.DataFrame()
    c,o,h,l=d.Close.astype(float),d.Open.astype(float),d.High.astype(float),d.Low.astype(float)
    v=d.Volume.astype(float) if "Volume" in d else pd.Series(np.nan,index=d.index)
    ret=c.pct_change(); delta=c.diff()
    gain=delta.clip(lower=0).rolling(14,min_periods=14).mean()
    loss=(-delta.clip(upper=0)).rolling(14,min_periods=14).mean()
    rs=gain/loss.replace(0,np.nan); rsi=100-100/(1+rs)
    rsi=rsi.where(loss!=0,100.0).where(gain!=0,50.0)
    vol=ret.rolling(14,min_periods=14).std()
    candle=(h-l).replace(0,np.nan)
    f=pd.DataFrame(index=d.index)
    f["return_1"]=c.pct_change(1); f["return_3"]=c.pct_change(3); f["return_6"]=c.pct_change(6)
    for n in (10,30):
        sma=c.rolling(n,min_periods=n).mean(); f[f"sma_ratio_{n}"]=(c-sma)/sma
    f["rsi"]=rsi
    f["squeeze_index"]=vol/vol.rolling(30,min_periods=10).mean().replace(0,np.nan)
    f["body_ratio"]=(c-o).abs()/candle
    f["pressure_diff"]=(np.minimum(o,c)-l)/candle-(h-np.maximum(o,c))/candle
    f["volatility"]=vol
    f["volume_change"]=v.pct_change().replace([np.inf,-np.inf],np.nan)
    return f.replace([np.inf,-np.inf],np.nan)

def labels_for(d, width, hold):
    y=pd.Series(np.nan,index=d.index,dtype=float)
    op,hi,lo=d.Open.to_numpy(float),d.High.to_numpy(float),d.Low.to_numpy(float)
    for i in range(max(0,len(d)-hold)):
        ei=i+1
        if ei>=len(d) or not np.isfinite(op[ei]) or op[ei]<=0: continue
        up,down=op[ei]+width,op[ei]-width
        for j in range(ei,min(ei+hold,len(d))):
            hu,hd=hi[j]>=up,lo[j]<=down
            if hu and hd: break
            if hu: y.iloc[i]=1.; break
            if hd: y.iloc[i]=0.; break
    return y

def build_data(name, all_data, width, hold):
    d=all_data.get(name)
    if d is None or d.empty: return pd.DataFrame(),pd.Series(dtype=int),pd.DataFrame()
    x=make_features(d)
    for other_name,other in all_data.items():
        if other_name==name or other.empty: continue
        tag=slug(other_name).lower()[:16]
        aligned=other.Close.astype(float).reindex(x.index,method="ffill")
        x[f"cross_{tag}"]=aligned.pct_change()
    y=labels_for(d,width,hold)
    x=x.ffill().bfill()
    valid=x.notna().all(axis=1)&y.notna()
    return x.loc[valid].copy(),y.loc[valid].astype(int).copy(),x

class ModelManager:
    def __init__(self,name):
        self.name=name; self.key=slug(name)
        self.mp=MODELS/f"{self.key}.v27.model.joblib"; self.sp=MODELS/f"{self.key}.v27.scaler.joblib"; self.meta_path=MODELS/f"{self.key}.v27.json"
        self.model=SGDClassifier(loss="log_loss",alpha=0.0005,random_state=42)
        self.scaler=StandardScaler(); self.meta={"trained_samples":0,"feature_columns":[]}; self.ready=False
    def load(self,columns):
        try:
            if not all(p.exists() for p in (self.mp,self.sp,self.meta_path)): return False
            m=json.loads(self.meta_path.read_text(encoding="utf-8"))
            if m.get("feature_columns")!=list(columns): return False
            self.model=joblib.load(self.mp); self.scaler=joblib.load(self.sp); self.meta=m; self.ready=True
            return True
        except Exception: return False
    def fit_initial(self,x,y):
        if len(x)<20 or y.nunique()<2: raise ValueError("初期学習データ不足")
        self.scaler.fit(x.values); self.model.partial_fit(self.scaler.transform(x.values),y.values,classes=np.array([0,1]))
        self.meta.update(trained_samples=int(len(x)),feature_columns=list(x.columns),last_fit=datetime.now(timezone.utc).isoformat())
        self.ready=True; self.save()
    def save(self):
        joblib.dump(self.model,self.mp); joblib.dump(self.scaler,self.sp)
        self.meta_path.write_text(json.dumps(self.meta,ensure_ascii=False,indent=2),encoding="utf-8")

st.markdown("""<style>.banner{background:linear-gradient(135deg,#0d1b2a,#1b263b,#415a77);padding:18px;border-radius:12px;color:#00ffcc;text-align:center;border:1px solid #00ffcc}</style>""",unsafe_allow_html=True)
st.markdown('<div class="banner"><h1>⚡ UTAYA v27 | 機関マルチアセット歪みランキング＆90%複利AI</h1><p>全12市場常時同時スキャン / オッズ歪み（確率差）による本命順位付け / 90%複利運用</p></div>',unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ v27 機関ポートフォリオ設定")
    if "account_capital" not in st.session_state: st.session_state.account_capital = 15.0
    if "target_profit" not in st.session_state: st.session_state.target_profit = 2.0
    
    current_cap = st.number_input("現在口座残高 ($)", 1.0, 100000.0, st.session_state.account_capital, 1.0)
    st.session_state.account_capital = current_cap
    stake_ratio = st.slider("投資比率 (%)", 10, 100, 90)
    invest_amount = current_cap * (stake_ratio / 100.0)
    
    hold=st.slider("判定期限（足数）",1,20,5)
    period=st.selectbox("データ期間",["30d","60d","3mo","6mo","1y"],index=1)
    interval=st.selectbox("時間足",["15m","30m","1h","1d"],index=2)
    threshold=st.slider("最低確率差（歪みフィルター）",0.,.5,.06,.01)
    
    st.divider()
    auto_refresh = st.checkbox("🔄 24時間機関自動巡回稼働 (15秒毎)", value=False)
    st.caption("15秒ごとに全12市場を同時スキャンし、最もプロが狙う歪み（本命）を上から順位付けして表示します。")

st.subheader("🏛️ 機関投資家スタイル：全12市場 歪みランキング＆本命抽出")

st.markdown(f"""
<div style="background: rgba(0,255,204,0.1); border: 1px solid #00ffcc; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
    <h3>💰 90%複利ステーク＆ターゲット管理</h3>
    <p><b>現在口座残高:</b> ${current_cap:.2f} ➔ <b>今回投資額 ({stake_ratio}%):</b> <b>${invest_amount:.2f}</b></p>
    <p><b>目標クリア条件 (+${st.session_state.target_profit:.2f}達成):</b> 資産が <b>${current_cap + st.session_state.target_profit:.2f}</b> に到達したら、元本を ${current_cap + st.session_state.target_profit:.2f} に引き上げてその90%を最上位の本命銘柄に再投資します。</p>
</div>
""", unsafe_allow_html=True)

placeholder = st.empty()

def scan_all_markets():
    all_data = {}
    for name, cfg in SYMBOLS.items():
        d = fetch_data(cfg["ticker"], period, interval)
        if not d.empty:
            all_data[name] = d
            
    rankings = []
    for name, cfg in SYMBOLS.items():
        if name not in all_data:
            continue
        try:
            x, y, x_all = build_data(name, all_data, cfg["range"], hold)
            if len(x) < 20 or y.nunique() < 2:
                continue
            manager = ModelManager(name)
            if not manager.load(list(x.columns)):
                manager.fit_initial(x, y)
                
            latest = x_all.dropna()
            if len(latest) < 2:
                continue
            row = latest.iloc[[-2]]
            pr = manager.model.predict_proba(manager.scaler.transform(row.values))[0]
            m = {int(c):float(p) for c,p in zip(manager.model.classes_, pr)}
            pl, ph = m.get(0,.5), m.get(1,.5)
            gap = abs(ph - pl)
            direction = "HIGH (買い)" if ph > pl else "LOW (売り)" if pl > ph else "見送り"
            
            score = gap
            if gap >= threshold:
                rankings.append({
                    "市場": name,
                    "判定": direction,
                    "HIGH確率": ph,
                    "LOW確率": pl,
                    "オッズ歪み(確率差)": gap,
                    "現在価格": float(all_data[name].Close.iloc[-1]),
                    "学習サンプル": manager.meta["trained_samples"],
                    "スコア": score
                })
        except Exception:
            pass
            
    rankings = sorted(rankings, key=lambda r: r["スコア"], reverse=True)
    return rankings

def render_ranking_dashboard():
    with st.spinner("全12市場を同時スキャンし、オッズの歪み（本命）を解析中..."):
        rankings = scan_all_markets()
        
    if not rankings:
        st.warning("現在、十分なサンプルを持つ市場がありません。データ期間や時間足を調整してください。")
        return
        
    df_rank = pd.DataFrame(rankings)
    df_disp = df_rank.copy()
    df_disp["HIGH確率"] = df_disp["HIGH確率"].apply(lambda v: f"{v:.1%}")
    df_disp["LOW確率"] = df_disp["LOW確率"].apply(lambda v: f"{v:.1%}")
    df_disp["オッズ歪み(確率差)"] = df_disp["オッズ歪み(確率差)"].apply(lambda v: f"{v:.3f}")
    df_disp["現在価格"] = df_disp["现在価格" if "现在価格" in df_disp.columns else "現在価格"].apply(lambda v: f"${v:,.2f}" if isinstance(v, (int,float)) else v)
    
    df_disp.insert(0, "順位", range(1, len(df_disp) + 1))
    
    st.markdown("### 🏆 機関投資家が狙う「本命・歪みランキング」トップ")
    st.dataframe(df_disp[["順位", "市場", "判定", "HIGH確率", "LOW確率", "オッズ歪み(確率差)", "現在価格", "学習サンプル"]], use_container_width=True, hide_index=True)
    
    top_pick = rankings[0]
    st.markdown(f"""
    <div style="background: rgba(0,255,204,0.15); border: 2px solid #00ffcc; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h2>🔥 本日の最高優位性（本命）：【 {top_pick['市場']} 】</h2>
        <p><b>推奨アクション:</b> <b>{top_pick['判定']}</b> | <b>オッズ歪み(確率差):</b> <b>{top_pick['オッズ歪み(確率差)']:.3f}</b></p>
        <p><b>推奨投資額 ({stake_ratio}%):</b> <b>${invest_amount:.2f}</b> (現在価格: ${top_pick['現在価格']:,.2f})</p>
        <p>分散投資の中核として、このトップ銘柄に資金の90%を集中投入し、+${st.session_state.target_profit:.2f} の利益回収を目指します。</p>
    </div>
    """, unsafe_allow_html=True)

if auto_refresh:
    with placeholder.container():
        st.success(f"🟢 機関自動巡回中 (最終スキャン: {datetime.now().strftime('%H:%M:%S')})")
        render_ranking_dashboard()
    time.sleep(15)
    st.rerun()
else:
    with placeholder.container():
        render_ranking_dashboard()
    st.info("サイドバーの「24時間機関自動巡回稼働 (15秒毎)」にチェックを入れると、15秒ごとに全12市場の歪みが自動で再計算され、ランキングがリアルタイム更新されます。")

st.divider()
st.caption("UTAYA v27 | 機関マルチアセット歪みランキング＆複利AI。利益・勝率は一切保証されません。自己責任で運用してください。")
