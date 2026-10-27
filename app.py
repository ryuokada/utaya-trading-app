# -*- coding: utf-8 -*-
"""UTAYA v32 | 30秒自動監査・オンライン学習版（利益・勝率保証なし）"""
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

st.set_page_config(
    page_title="UTAYA v32 | 自動市場監査",
    page_icon="⚡",
    layout="wide"
)

SYMBOLS = {
    "BTC": {"ticker": "BTC-USD"},
    "ETH": {"ticker": "ETH-USD"},
    "SOL": {"ticker": "SOL-USD"},
    "XRP": {"ticker": "XRP-USD"},
    "BNB": {"ticker": "BNB-USD"},
    "WTI原油": {"ticker": "CL=F"},
    "金先物": {"ticker": "GC=F"},
    "日経平均": {"ticker": "^N225"},
    "S&P 500": {"ticker": "^GSPC"},
    "Nasdaq 100": {"ticker": "^NDX"},
    "USD/JPY": {"ticker": "USDJPY=X"},
    "米10年債利回り": {"ticker": "^TNX"},
}

DEFAULT_DIR = Path("./utaya_v32_data")
ROOT = Path(os.environ.get("UTAYA_DATA_DIR", str(DEFAULT_DIR)))
HISTORY, MODELS, LOGS = ROOT/"history", ROOT/"models", ROOT/"logs"
for p in (ROOT, HISTORY, MODELS, LOGS): p.mkdir(parents=True, exist_ok=True)

def slug(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", SYMBOLS[name]["ticker"])

@st.cache_data(ttl=25, show_spinner=False)
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
                    d = pd.concat([old, d])
                    d = d[~d.index.duplicated(keep="last")].sort_index()
                except Exception: pass
            d.to_csv(path, encoding="utf-8-sig")
            return d
    except Exception:
        pass
    if path.exists():
        try:
            d = pd.read_csv(path, index_col=0, parse_dates=True)
            d.index = pd.to_datetime(d.index)
            if getattr(d.index, "tz", None) is not None: d.index = d.index.tz_localize(None)
            return d.sort_index()
        except Exception: pass
    return pd.DataFrame()

def make_features(d):
    if d.empty or len(d) < 35: return pd.DataFrame(), False
    c, o, h, l = d.Close.astype(float), d.Open.astype(float), d.High.astype(float), d.Low.astype(float)
    has_vol = "Volume" in d and d["Volume"].sum() > 0
    v = d.Volume.astype(float) if has_vol else pd.Series(0.0, index=d.index)
    
    ret = c.pct_change()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(loss != 0, 100.0).where(gain != 0, 50.0)
    
    vol = ret.rolling(14, min_periods=14).std()
    candle = (h - l).replace(0, np.nan)
    
    f = pd.DataFrame(index=d.index)
    f["return_1"] = c.pct_change(1)
    f["return_3"] = c.pct_change(3)
    f["return_6"] = c.pct_change(6)
    for n in (10, 30):
        sma = c.rolling(n, min_periods=n).mean()
        f[f"sma_ratio_{n}"] = (c - sma) / sma
        
    f["rsi"] = rsi
    f["squeeze_index"] = vol / vol.rolling(30, min_periods=10).mean().replace(0, np.nan)
    f["body_ratio"] = (c - o).abs() / candle
    f["pressure_diff"] = (np.minimum(o, c) - l) / candle - (h - np.maximum(o, c)) / candle
    f["volatility"] = vol
    f["volume_change"] = v.pct_change().replace([np.inf, -np.inf], np.nan) if has_vol else 0.0
    
    return f.replace([np.inf, -np.inf], np.nan), has_vol

def labels_for_binary(d, hold):
    y = pd.Series(np.nan, index=d.index, dtype=float)
    close = d.Close.to_numpy(float)
    for i in range(len(d) - hold):
        target_idx = i + hold
        if target_idx >= len(d): break
        entry_p = close[i]
        future_p = close[target_idx]
        if future_p > entry_p:
            y.iloc[i] = 1.0
        elif future_p < entry_p:
            y.iloc[i] = 0.0
        else:
            y.iloc[i] = np.nan
    return y

def build_data(name, all_data, hold):
    d = all_data.get(name)
    if d is None or d.empty: return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(), False, "データ空"
    x, has_vol = make_features(d)
    if x.empty: return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(), has_vol, "特徴量生成失敗"
    
    for other_name, other in all_data.items():
        if other_name == name or other.empty: continue
        tag = slug(other_name).lower()[:16]
        aligned = other.Close.astype(float).reindex(x.index, method="ffill")
        x[f"cross_{tag}"] = aligned.pct_change()
        
    y = labels_for_binary(d, hold)
    x = x.ffill()
    valid = x.notna().all(axis=1) & y.notna()
    if valid.sum() < 20:
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(), has_vol, f"有効サンプル不足 ({valid.sum()}件)"
    return x.loc[valid].copy(), y.loc[valid].astype(int).copy(), x, has_vol, "OK"

class ModelManager:
    def __init__(self, name):
        self.name = name; self.key = slug(name)
        self.mp = MODELS / f"{self.key}.v32.model.joblib"
        self.sp = MODELS / f"{self.key}.v32.scaler.joblib"
        self.meta_path = MODELS / f"{self.key}.v32.json"
        self.model = SGDClassifier(loss="log_loss", alpha=0.0005, random_state=42)
        self.scaler = StandardScaler()
        self.meta = {"trained_samples": 0, "feature_columns": [], "prediction_count": 0, "online_updates": 0}
        self.ready = False
        
    def load(self, columns):
        try:
            if not all(p.exists() for p in (self.mp, self.sp, self.meta_path)): return False
            m = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if m.get("feature_columns") != list(columns): return False
            self.model = joblib.load(self.mp); self.scaler = joblib.load(self.sp); self.meta = m; self.ready = True
            return True
        except Exception: return False
        
    def fit_initial(self, x, y):
        if len(x) < 20 or y.nunique() < 2: raise ValueError("初期学習不足")
        self.scaler.fit(x.values)
        self.model.partial_fit(self.scaler.transform(x.values), y.values, classes=np.array([0, 1]))
        self.meta.update(trained_samples=int(len(x)), feature_columns=list(x.columns), last_fit=datetime.now(timezone.utc).isoformat())
        self.ready = True; self.save()

    def online_update(self, row_features, true_label):
        if not self.ready: return
        try:
            X_scaled = self.scaler.transform(row_features.values)
            self.model.partial_fit(X_scaled, np.array([int(true_label)]))
            self.meta["online_updates"] = self.meta.get("online_updates", 0) + 1
            self.save()
        except Exception: pass
        
    def save(self):
        joblib.dump(self.model, self.mp); joblib.dump(self.scaler, self.sp)
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

st.markdown("""<style>
.banner{background:linear-gradient(135deg,#0d1b2a,#1b263b,#415a77);padding:18px;border-radius:12px;color:#00ffcc;text-align:center;border:1px solid #00ffcc}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="banner"><h1>⚡ UTAYA v32 | 30秒自動監査・オンライン学習版</h1><p>多市場並列監視 ＋ 適応型オンライン学習エンジン</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ v32 監査設定")
    if "account_capital" not in st.session_state: st.session_state.account_capital = 15.0
    
    current_cap = st.number_input("現在口座残高 ($)", 1.0, 100000.0, st.session_state.account_capital, 1.0)
    st.session_state.account_capital = current_cap
    stake_mode = st.selectbox("資金管理モード", ["固定額 ($1)", "複利 (残高の50%)", "複利 (残高の90%)"], index=1)
    
    invest_amount = 1.0 if "固定" in stake_mode else (current_cap * 0.5 if "50%" in stake_mode else current_cap * 0.9)
    payout_ratio = st.number_input("支払い率 (例: 0.75 = 75%)", 0.1, 5.0, 0.75, 0.05)
    break_even_winrate = 1.0 / (payout_ratio + 1.0) * 100.0
    
    hold = st.slider("判定期限（足数）", 1, 20, 5)
    period = st.selectbox("データ期間", ["30d", "60d", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("時間足", ["1m", "5m", "15m", "30m", "1h", "1d"], index=0)
    threshold = st.slider("シグナル点灯フィルター（確率差）", 0., .5, .03, .01)
    
    st.divider()
    auto_refresh = st.checkbox("🔄 30秒ごとに自動更新（オートパイロット）", value=True)

st.subheader(f"📊 期待値＆リファレンス (支払い率: {payout_ratio*100:.0f}%)")
st.info(f"🎯 **損益分岐勝率: {break_even_winrate:.2f}%** | **自動投資額:** ${invest_amount:.2f}")

placeholder = st.empty()

def run_v32_audit():
    all_data = {}
    for name, cfg in SYMBOLS.items():
        d = fetch_data(cfg["ticker"], period, interval)
        if not d.empty: all_data[name] = d
            
    rankings = []
    audit_summary = []
    
    for name, cfg in SYMBOLS.items():
        if name not in all_data: continue
        try:
            d_market = all_data[name]
            x, y, x_all, has_vol, reason = build_data(name, all_data, hold)
            if reason != "OK": continue
                
            manager = ModelManager(name)
            if not manager.load(list(x.columns)):
                try: manager.fit_initial(x, y)
                except: continue
                    
            latest = x_all.dropna()
            if len(latest) < 2: continue
                
            row = latest.iloc[[-1]]
            pr = manager.model.predict_proba(manager.scaler.transform(row.values))[0]
            m = {int(c):float(p) for c,p in zip(manager.model.classes_, pr)}
            pl, ph = m.get(0,.5), m.get(1,.5)
            gap = abs(ph - pl)
            
            if ph > pl:
                direction = "HIGH (上がる)"
                raw_dir = 1
                conf = ph
            else:
                direction = "LOW (下がる)"
                raw_dir = 0
                conf = pl
                
            manager.meta["prediction_count"] = manager.meta.get("prediction_count", 0) + 1
            manager.save()
            
            # 確率差が閾値未満なら「見送り」
            if gap >= threshold:
                verdict = "🔥 買いシグナル点灯"
            else:
                verdict = "🛡️ 見送り (確率差不足)"
                
            rankings.append({
                "市場": name,
                "判定": verdict,
                "予測方向": direction,
                "確信度": conf,
                "HIGH確率": ph,
                "LOW確率": pl,
                "確率差(歪み)": gap,
                "現在価格": float(d_market.Close.iloc[-1]),
                "オンライン学習回数": manager.meta.get("online_updates", 0),
                "スコア": gap
            })
        except: pass
            
    return sorted(rankings, key=lambda r: r["スコア"], reverse=True)

def render_dashboard():
    with st.spinner("⚡ 30秒自動監査・市場データ並列スキャン中..."):
        rankings = run_v32_audit()
        
    tab1, tab2 = st.tabs(["🚀 30秒リアルタイム・シグナル監視塔", "📊 全12市場・監査マトリクス"])
    
    with tab1:
        if not rankings:
            st.warning("現在有効な市場データが取得できていません。")
        else:
            top = rankings[0]
            with st.container(border=True):
                col_t, col_b = st.columns([3, 1])
                with col_t:
                    st.caption("🔥 最優位・一瞬のエントリーシグナル（BTC最優先監視）")
                    st.markdown(f"### **{top['市場']}** ── 判定: **{top['判定']}**")
                with col_b:
                    if "LOW" in top['予測方向']: st.error(f"**{top['予測方向']}**")
                    else: st.success(f"**{top['予測方向']}**")
                
                st.write("---")
                conf_p = float(top['確信度'])
                st.markdown(f"🟢 HIGH確率: **{top['HIGH確率']*100:.1f}%** | 🔴 LOW確率: **{top['LOW確率']*100:.1f}%** | 歪み(差): **{top['確率差(歪み)']:.3f}**")
                st.progress(conf_p)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("現在価格", f"${top['現在価格']:,.2f}")
                m2.metric("モデル確信度", f"{conf_p*100:.1f}%")
                m3.metric(f"自動投資額 ({stake_mode})", f"${invest_amount:.2f}")
                m4.metric("オンライン学習回数", f"{top['オンライン学習回数']}回")

    with tab2:
        st.markdown("### 📊 全12市場・並列監査マトリクス")
        if rankings:
            df_r = pd.DataFrame(rankings)
            df_disp = df_r.copy()
            for col in ["HIGH確率", "LOW確率", "確信度"]: df_disp[col] = df_disp[col].apply(lambda v: f"{v:.1%}")
            df_disp["確率差(歪み)"] = df_disp["確率差(歪み)"].apply(lambda v: f"{v:.3f}")
            df_disp["現在価格"] = df_disp["現在価格"].apply(lambda v: f"${v:,.2f}")
            df_disp.insert(0, "順位", range(1, len(df_disp) + 1))
            st.dataframe(df_disp, use_container_width=True, hide_index=True)

if auto_refresh:
    with placeholder.container():
        render_dashboard()
    time.sleep(30)
    st.rerun()
else:
    with placeholder.container():
        render_dashboard()

st.divider()
st.caption("UTAYA v32 | 30秒自動監査・オンライン学習版。デモ環境にて予測と結果の整合性を検証しながら運用してください。")
