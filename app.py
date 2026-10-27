# -*- coding: utf-8 -*-
"""UTAYA v29 | 群集心理・市場歪み監査＆90%複利自律AIエンジン。利益保証なし。"""
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

st.set_page_config(page_title="UTAYA v29 | 群集心理監査AI", page_icon="⚡", layout="wide")

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
DEFAULT_DIR = Path(r"E:\utaya_3tb_ai_storage\utaya_v29")
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
                        progress=False, threads=False, timeout=12)
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
    if d.empty: return pd.DataFrame(), False
    c,o,h,l = d.Close.astype(float), d.Open.astype(float), d.High.astype(float), d.Low.astype(float)
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
    
    if has_vol:
        f["volume_change"] = v.pct_change().replace([np.inf, -np.inf], np.nan)
    else:
        f["volume_change"] = 0.0
        
    return f.replace([np.inf, -np.inf], np.nan), has_vol

def labels_for(d, width, hold):
    y = pd.Series(np.nan, index=d.index, dtype=float)
    op, hi, lo = d.Open.to_numpy(float), d.High.to_numpy(float), d.Low.to_numpy(float)
    for i in range(max(0, len(d) - hold)):
        ei = i + 1
        if ei >= len(d) or not np.isfinite(op[ei]) or op[ei] <= 0: continue
        up, down = op[ei] + width, op[ei] - width
        for j in range(ei, min(ei + hold, len(d))):
            hu, hd = hi[j] >= up, lo[j] <= down
            if hu and hd: break
            if hu: y.iloc[i] = 1.; break
            if hd: y.iloc[i] = 0.; break
    return y

def build_data(name, all_data, width, hold):
    d = all_data.get(name)
    if d is None or d.empty: return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(), False, "データ空"
    
    x, has_vol = make_features(d)
    if x.empty: return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(), has_vol, "特徴量生成失敗"
    
    for other_name, other in all_data.items():
        if other_name == name or other.empty: continue
        tag = slug(other_name).lower()[:16]
        aligned = other.Close.astype(float).reindex(x.index, method="ffill")
        x[f"cross_{tag}"] = aligned.pct_change()
        
    y = labels_for(d, width, hold)
    x = x.ffill()
    valid = x.notna().all(axis=1) & y.notna()
    
    if valid.sum() < 20:
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(), has_vol, f"有効サンプル不足 ({valid.sum()}件)"
        
    return x.loc[valid].copy(), y.loc[valid].astype(int).copy(), x, has_vol, "OK"

class ModelManager:
    def __init__(self, name):
        self.name = name; self.key = slug(name)
        self.mp = MODELS / f"{self.key}.v29.model.joblib"
        self.sp = MODELS / f"{self.key}.v29.scaler.joblib"
        self.meta_path = MODELS / f"{self.key}.v29.json"
        self.model = SGDClassifier(loss="log_loss", alpha=0.0005, random_state=42)
        self.scaler = StandardScaler()
        self.meta = {"trained_samples": 0, "feature_columns": [], "prediction_count": 0}
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
        
    def save(self):
        joblib.dump(self.model, self.mp); joblib.dump(self.scaler, self.sp)
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

st.markdown("""<style>.banner{background:linear-gradient(135deg,#0d1b2a,#1b263b,#415a77);padding:18px;border-radius:12px;color:#00ffcc;text-align:center;border:1px solid #00ffcc}</style>""", unsafe_allow_html=True)
st.markdown('<div class="banner"><h1>⚡ UTAYA v29 | 群集心理・市場歪み監査エンジン</h1><p>全12市場常時監査 / 逆転警戒・過熱スコアリング / 90%複利運用モード</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ v29 監査＆複利設定")
    if "account_capital" not in st.session_state: st.session_state.account_capital = 15.0
    if "target_profit" not in st.session_state: st.session_state.target_profit = 2.0
    
    current_cap = st.number_input("現在口座残高 ($)", 1.0, 100000.0, st.session_state.account_capital, 1.0)
    st.session_state.account_capital = current_cap
    stake_ratio = st.slider("投資比率 (%)", 10, 100, 90)
    invest_amount = current_cap * (stake_ratio / 100.0)
    
    hold = st.slider("判定期限（足数）", 1, 20, 5)
    period = st.selectbox("データ期間", ["30d", "60d", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("時間足", ["15m", "30m", "1h", "1d"], index=2)
    threshold = st.slider("最低確率差（歪みフィルター）", 0., .5, .06, .01)
    
    st.divider()
    auto_refresh = st.checkbox("🔄 24時間群集心理監査巡回 (15秒毎)", value=False)
    st.caption("15秒ごとに全12市場を同時スキャンし、心理の偏りと反転警戒リスクを判定して表示します。")

st.subheader("🏛️ 群集心理・市場歪み監査ダッシュボード")

st.markdown(f"""
<div style="background: rgba(0,255,204,0.1); border: 1px solid #00ffcc; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
    <h3>💰 90%複利ステーク＆ターゲット管理</h3>
    <p><b>現在口座残高:</b> ${current_cap:.2f} ➔ <b>今回投資額 ({stake_ratio}%):</b> <b>${invest_amount:.2f}</b></p>
    <p><b>目標クリア条件 (+${st.session_state.target_profit:.2f}達成):</b> 資産が <b>${current_cap + st.session_state.target_profit:.2f}</b> に到達したら、元本を ${current_cap + st.session_state.target_profit:.2f} に引き上げてその90%を最上位の本命銘柄に再投資します。</p>
</div>
""", unsafe_allow_html=True)

placeholder = st.empty()

def audit_crowd_psychology():
    all_data = {}
    for name, cfg in SYMBOLS.items():
        d = fetch_data(cfg["ticker"], period, interval)
        if not d.empty:
            all_data[name] = d
            
    audits = []
    audit_logs = []
    
    for name, cfg in SYMBOLS.items():
        if name not in all_data:
            audit_logs.append({"市場": name, "ステータス": "❌ データなし", "理由": "市場データ取得失敗"})
            continue
        try:
            x, y, x_all, has_vol, reason = build_data(name, all_data, cfg["range"], hold)
            if reason != "OK":
                audit_logs.append({"市場": name, "ステータス": "⚠️ 除外", "理由": reason})
                continue
                
            manager = ModelManager(name)
            if not manager.load(list(x.columns)):
                try:
                    manager.fit_initial(x, y)
                except Exception as ex:
                    audit_logs.append({"市場": name, "ステータス": "⚠️ 初期学習失敗", "理由": str(ex)})
                    continue
                    
            latest = x_all.dropna()
            if len(latest) < 2:
                audit_logs.append({"市場": name, "ステータス": "⚠️ 除外", "理由": "最新特徴量不足"})
                continue
                
            row = latest.iloc[[-2]]
            pr = manager.model.predict_proba(manager.scaler.transform(row.values))[0]
            m = {int(c):float(p) for c,p in zip(manager.model.classes_, pr)}
            pl, ph = m.get(0,.5), m.get(1,.5)
            gap = abs(ph - pl)
            direction = "HIGH (買い)" if ph > pl else "LOW (売り)" if pl > ph else "見送り"
            
            manager.meta["prediction_count"] = manager.meta.get("prediction_count", 0) + 1
            manager.save()
            
            rsi_val = float(row["rsi"].iloc[0]) if "rsi" in row else 50.0
            vol_val = float(row["volatility"].iloc[0]) if "volatility" in row else 0.02
            squeeze = float(row["squeeze_index"].iloc[0]) if "squeeze_index" in row else 1.0
            
            if gap >= 0.15 and squeeze > 1.2:
                attention = "高 (資金集中)"
            elif gap >= 0.08:
                attention = "中 (通常)"
            else:
                attention = "低 (閑散)"
                
            if rsi_val > 75 or rsi_val < 25 or vol_val > 0.05:
                reversal_risk = "高 (過熱・反転警戒)"
                verdict = "反転警戒・見送り"
            elif gap >= threshold:
                reversal_risk = "低〜中 (順張り可能)"
                verdict = "順張り候補"
            else:
                reversal_risk = "低"
                verdict = "見送り"
                
            audits.append({
                "市場": name,
                "最終判断": verdict,
                "判定": direction,
                "注目度": attention,
                "逆転警戒": reversal_risk,
                "HIGH確率": ph,
                "LOW確率": pl,
                "オッズ歪み(確率差)": gap,
                "RSI": rsi_val,
                "現在価格": float(all_data[name].Close.iloc[-1]),
                "スコア": gap
            })
            audit_logs.append({"市場": name, "ステータス": "✅ 監査完了", "理由": f"判定: {verdict} (歪み: {gap:.3f})"})
        except Exception as ex:
            audit_logs.append({"市場": name, "ステータス": "❌ 例外", "理由": str(ex)})
            
    audits = sorted(audits, key=lambda r: r["スコア"], reverse=True)
    return audits, audit_logs

def render_v29_dashboard():
    with st.spinner("全12市場の群集心理・過熱・逆転リスクを監査中..."):
        audits, audit_logs = audit_crowd_psychology()
        
    tab1, tab2 = st.tabs(["🏛️ 群集心理・市場歪み監査結果", "🔍 システム全銘柄監査ログ"])
    
    with tab1:
        if not audits:
            st.warning("現在、有効な監査データを持つ市場がありません。「システム全銘柄監査ログ」で各銘柄の状況を確認してください。")
        else:
            df_aud = pd.DataFrame(audits)
            df_disp = df_aud.copy()
            df_disp["HIGH確率"] = df_disp["HIGH確率"].apply(lambda v: f"{v:.1%}")
            df_disp["LOW確率"] = df_disp["LOW確率"].apply(lambda v: f"{v:.1%}")
            df_disp["オッズ歪み(確率差)"] = df_disp["オッズ歪み(確率差)"].apply(lambda v: f"{v:.3f}")
            df_disp["RSI"] = df_disp["RSI"].apply(lambda v: f"{v:.1f}")
            df_disp["現在価格"] = df_disp["現在価格"].apply(lambda v: f"${v:,.2f}")
            df_disp.insert(0, "順位", range(1, len(df_disp) + 1))
            
            st.markdown("### 🏆 市場歪み・群集心理監査ランキング")
            st.dataframe(df_disp[["順位", "市場", "最終判断", "判定", "注目度", "逆転警戒", "オッズ歪み(確率差)", "RSI", "現在価格"]], use_container_width=True, hide_index=True)
            
            top_pick = audits[0]
            st.markdown(f"""
            <div style="background: rgba(0,255,204,0.15); border: 2px solid #00ffcc; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h2>🔥 本日の最上位・監査通過銘柄：【 {top_pick['市場']} 】</h2>
                <p><b>最終判断:</b> <b>{top_pick['最終判断']}</b> ({top_pick['判定']})</p>
                <p><b>注目度:</b> {top_pick['注目度']} | <b>逆転警戒:</b> {top_pick['逆転警戒']}</p>
                <p><b>推奨投資額 ({stake_ratio}%):</b> <b>${invest_amount:.2f}</b> (現在価格: ${top_pick['現在価格']:,.2f})</p>
                <p>群集心理の偏りと歪みを総合監査し、過熱リスクを管理した上でこのトップ銘柄に資金の90%を投じ、+${st.session_state.target_profit:.2f} の利益回収を狙います。</p>
            </div>
            """, unsafe_allow_html=True)
            
    with tab2:
        st.markdown("### 🔍 全12市場の監査・除外診断ステータス")
        st.dataframe(pd.DataFrame(audit_logs), use_container_width=True, hide_index=True)

if auto_refresh:
    with placeholder.container():
        st.success(f"🟢 群集心理監査巡回中 (最終更新: {datetime.now().strftime('%H:%M:%S')})")
        render_v29_dashboard()
    time.sleep(15)
    st.rerun()
else:
    with placeholder.container():
        render_v29_dashboard()
    st.info("サイドバーのアプリ設定から「24時間群集心理監査巡回 (15秒毎)」にチェックを入れると、15秒ごとに全12市場の心理偏りと反転リスクが自動更新されます。")

st.divider()
st.caption("UTAYA v29 | 群集心理・市場歪み監査AI。利益・勝率は一切保証されません。自己責任で運用してください。")
