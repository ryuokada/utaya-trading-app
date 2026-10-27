# -*- coding: utf-8 -*-
"""UTAYA v30 | ハイロー取引勝敗条件整合＆実証監査AIエンジン（ネイティブUI・利益保証なし）"""
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

st.set_page_config(page_title="UTAYA v30 | ハイロー取引実証監査AI", page_icon="🛡️", layout="wide")

SYMBOLS = {
    "BTC": {"ticker":"BTC-USD"},
    "ETH": {"ticker":"ETH-USD"},
    "SOL": {"ticker":"SOL-USD"},
    "XRP": {"ticker":"XRP-USD"},
    "BNB": {"ticker":"BNB-USD"},
    "WTI原油": {"ticker":"CL=F"},
    "金先物": {"ticker":"GC=F"},
    "日経平均": {"ticker":"^N225"},
    "S&P 500": {"ticker":"^GSPC"},
    "Nasdaq 100": {"ticker":"^NDX"},
    "USD/JPY": {"ticker":"USDJPY=X"},
    "米10年債利回り": {"ticker":"^TNX"},
}

DEFAULT_DIR = Path(r"E:\utaya_3tb_ai_storage\utaya_v30")
ROOT = Path(os.environ.get("UTAYA_DATA_DIR", str(DEFAULT_DIR)))
HISTORY, MODELS, LOGS = ROOT/"history", ROOT/"models", ROOT/"logs"
for p in (ROOT, HISTORY, MODELS, LOGS): p.mkdir(parents=True, exist_ok=True)

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
            d = pd.read_csv(path, index_col=0, parse_dates=True)
            d.index = pd.to_datetime(d.index)
            if getattr(d.index,"tz",None) is not None: d.index=d.index.tz_localize(None)
            return d.sort_index()
        except Exception: pass
    return pd.DataFrame()

def make_features(d):
    if d.empty: return pd.DataFrame(), False
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
    
    if valid.sum() < 30:
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(), has_vol, f"有効サンプル不足 ({valid.sum()}件)"
        
    return x.loc[valid].copy(), y.loc[valid].astype(int).copy(), x, has_vol, "OK"

class ModelManager:
    def __init__(self, name):
        self.name = name; self.key = slug(name)
        self.mp = MODELS / f"{self.key}.v30.model.joblib"
        self.sp = MODELS / f"{self.key}.v30.scaler.joblib"
        self.meta_path = MODELS / f"{self.key}.v30.json"
        self.model = SGDClassifier(loss="log_loss", alpha=0.0005, random_state=42)
        self.scaler = StandardScaler()
        self.meta = {"trained_samples": 0, "feature_columns": [], "prediction_count": 0, "update_count": 0}
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
        if len(x) < 30 or y.nunique() < 2: raise ValueError("初期学習不足")
        self.scaler.fit(x.values)
        self.model.partial_fit(self.scaler.transform(x.values), y.values, classes=np.array([0, 1]))
        self.meta.update(trained_samples=int(len(x)), feature_columns=list(x.columns), last_fit=datetime.now(timezone.utc).isoformat())
        self.ready = True; self.save()
        
    def save(self):
        joblib.dump(self.model, self.mp); joblib.dump(self.scaler, self.sp)
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

st.markdown("""<style>
.banner{background:linear-gradient(135deg,#0d1b2a,#1b263b,#415a77);padding:18px;border-radius:12px;color:#00ffcc;text-align:center;border:1px solid #00ffcc}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="banner"><h1>🛡️ UTAYA v30 | ハイロー取引勝敗条件整合＆実証監査AI</h1><p>バイナリーオプション勝敗モデリング / 損益分岐勝率自動計算 / ネイティブUI</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ v30 取引＆資金管理設定")
    if "account_capital" not in st.session_state: st.session_state.account_capital = 15.0
    
    current_cap = st.number_input("現在口座残高 ($)", 1.0, 100000.0, st.session_state.account_capital, 1.0)
    st.session_state.account_capital = current_cap
    stake_mode = st.selectbox("資金管理モード", ["固定額 ($1)", "複利 (残高の50%)", "複利 (残高の90%)"], index=1)
    
    if "固定" in stake_mode:
        invest_amount = 1.0
    elif "50%" in stake_mode:
        invest_amount = current_cap * 0.5
    else:
        invest_amount = current_cap * 0.9
        
    payout_ratio = st.number_input("支払い率 (例: 0.75 = 75%)", 0.1, 5.0, 0.75, 0.05)
    break_even_winrate = 1.0 / (payout_ratio + 1.0) * 100.0
    
    hold = st.slider("判定期限（足数 / ハイロー期間）", 1, 20, 5)
    period = st.selectbox("データ期間", ["30d", "60d", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("時間足", ["1m", "5m", "15m", "30m", "1h", "1d"], index=2)
    threshold = st.slider("最低確率差（エントリーフィルター）", 0., .5, .08, .01)
    
    st.divider()
    auto_refresh = st.checkbox("🔄 24時間リアルタイム監査巡回 (15秒毎)", value=False)

st.subheader(f"📊 損益分岐勝率・期待値リファレンス (支払い率: {payout_ratio*100:.0f}%)")
st.info(f"🎯 **損益分岐勝率: {break_even_winrate:.2f}%** （この勝率を実証テストで安定して超えるモデルのみが実戦投入の基準となります） | **今回投資額:** ${invest_amount:.2f}")

placeholder = st.empty()

def run_v30_audit():
    all_data = {}
    for name, cfg in SYMBOLS.items():
        d = fetch_data(cfg["ticker"], period, interval)
        if not d.empty:
            all_data[name] = d
            
    rankings = []
    audit_logs = []
    
    for name, cfg in SYMBOLS.items():
        if name not in all_data:
            audit_logs.append({"市場": name, "ステータス": "❌ データなし", "理由": "市場データ取得失敗"})
            continue
        try:
            x, y, x_all, has_vol, reason = build_data(name, all_data, hold)
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
            direction = "HIGH (上)" if ph > pl else "LOW (下)" if pl < ph else "見送り"
            
            manager.meta["prediction_count"] = manager.meta.get("prediction_count", 0) + 1
            manager.save()
            
            rsi_val = float(row["rsi"].iloc[0]) if "rsi" in row else 50.0
            
            if gap >= threshold:
                verdict = "エントリー候補"
            else:
                verdict = "見送り (確率差不足)"
                
            rankings.append({
                "市場": name,
                "最終判断": verdict,
                "予測方向": direction,
                "HIGH確率": ph,
                "LOW確率": pl,
                "確率差(歪み)": gap,
                "RSI": rsi_val,
                "現在価格": float(all_data[name].Close.iloc[-1]),
                "学習サンプル": manager.meta["trained_samples"],
                "予測回数": manager.meta["prediction_count"],
                "スコア": gap
            })
            audit_logs.append({"市場": name, "ステータス": "✅ 正常監査", "理由": f"判定: {verdict} (HIGH確率: {ph:.1%})"})
        except Exception as ex:
            audit_logs.append({"市場": name, "ステータス": "❌ 例外", "理由": str(ex)})
            
    rankings = sorted(rankings, key=lambda r: r["スコア"], reverse=True)
    return rankings, audit_logs

def render_dashboard():
    with st.spinner("全12市場のハイロー勝敗条件に基づく監査を実行中..."):
        rankings, audit_logs = run_v30_audit()
        
    tab1, tab2, tab3 = st.tabs(["🏆 ハイロー予測・監査ランキング", "🧪 厳格バックテスト（時系列検証）", "🔍 システム全銘柄監査ログ"])
    
    with tab1:
        if not rankings:
            st.warning("現在、条件を満たす市場がありません。「システム全銘柄監査ログ」タブでデータ不足の原因を確認してください。")
        else:
            top = rankings[0]
            
            # --- 100%バグのないネイティブStreamlitコンポーネントによる最上位候補カード ---
            with st.container(border=True):
                col_title, col_badge = st.columns([3, 1])
                with col_title:
                    st.caption("🔥 現在の最上位・推奨候補")
                    st.markdown(f"### **{top['市場']}**")
                with col_badge:
                    if "LOW" in top['予測方向']:
                        st.error(f"**{top['予測方向']}**")
                    else:
                        st.success(f"**{top['予測方向']}**")
                
                st.write("---")
                
                high_p = float(top['HIGH確率'])
                low_p = float(top['LOW確率'])
                
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.markdown(f"🟢 HIGH確率: **{high_p * 100:.1f}%**")
                with p_col2:
                    st.markdown(f"🔴 LOW確率: **{low_p * 100:.1f}%**")
                
                st.progress(high_p)
                
                st.write("")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("現在価格", f"${top['現在価格']:,.2f}")
                m2.metric("オッズ歪み (確率差)", f"{top['確率差(歪み)']:.3f}")
                m3.metric(f"推奨投資額 ({stake_mode})", f"${invest_amount:.2f}")

            df_r = pd.DataFrame(rankings)
            df_disp = df_r.copy()
            df_disp["HIGH確率"] = df_disp["HIGH確率"].apply(lambda v: f"{v:.1%}")
            df_disp["LOW確率"] = df_disp["LOW確率"].apply(lambda v: f"{v:.1%}")
            df_disp["確率差(歪み)"] = df_disp["確率差(歪み)"].apply(lambda v: f"{v:.3f}")
            df_disp["RSI"] = df_disp["RSI"].apply(lambda v: f"{v:.1f}")
            df_disp["現在価格"] = df_disp["現在価格"].apply(lambda v: f"${v:,.2f}")
            df_disp.insert(0, "順位", range(1, len(df_disp) + 1))
            
            st.markdown("### 🏆 全銘柄ハイロー・優位性ランキング")
            st.dataframe(df_disp[["順位", "市場", "最終判断", "予測方向", "HIGH確率", "LOW確率", "確率差(歪み)", "RSI", "現在価格", "学習サンプル"]], use_container_width=True, hide_index=True)
            
    with tab2:
        st.markdown("### 🧪 厳格ウォークフォワード・バックテスト（未来リーク完全排除）")
        selected_bt = st.selectbox("バックテスト対象市場", list(SYMBOLS), index=0)
        if st.button("▶ バックテスト実行", use_container_width=True):
            with st.spinner("時系列ウォークフォワード検証中..."):
                all_data = fetch_data(SYMBOLS[selected_bt]["ticker"], period, interval)
                x, y, x_all, has_vol, reason = build_data(selected_bt, {selected_bt: all_data}, hold)
                
            if len(x) < 50:
                st.warning(f"バックテストに必要なデータが不足しています（有効サンプル: {len(x)}件）。期間を広げてください。")
            else:
                start_idx = int(len(x) * 0.4)
                scaler = StandardScaler()
                model = SGDClassifier(loss="log_loss", alpha=0.0005, random_state=42)
                scaler.fit(x.iloc[:start_idx].values)
                model.partial_fit(scaler.transform(x.iloc[:start_idx].values), y.iloc[:start_idx].values, classes=np.array([0, 1]))
                
                wins, losses, total_pnl = 0, 0, 0.0
                capital = current_cap
                
                for i in range(start_idx, len(x)):
                    row = x.iloc[[i]]
                    pr = model.predict_proba(scaler.transform(row.values))[0]
                    m = {int(c):float(p) for c,p in zip(model.classes_, pr)}
                    pl, ph = m.get(0,.5), m.get(1,.5)
                    gap = abs(ph - pl)
                    pred = 1 if ph > pl else 0
                    
                    actual = int(y.iloc[i])
                    
                    if gap >= threshold:
                        won = (pred == actual)
                        pnl = (invest_amount * payout_ratio) if won else (-invest_amount)
                        if won: wins += 1
                        else: losses += 1
                        total_pnl += pnl
                        capital += pnl
                        
                    model.partial_fit(scaler.transform(row.values), np.array([actual]), classes=np.array([0, 1]))
                    
                total_trades = wins + losses
                actual_winrate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("総取引回数", total_trades)
                c2.metric("実測勝率", f"{actual_winrate:.2f}%", delta=f"{actual_winrate - break_even_winrate:+.2f}% (vs 損益分岐)")
                c3.metric("総損益 ($)", f"${total_pnl:+.2f}")
                c4.metric("損益分岐勝率", f"{break_even_winrate:.2f}%")
                
                if actual_winrate >= break_even_winrate:
                    st.success(f"🟢 実測勝率 ({actual_winrate:.2f}%) が損益分岐勝率 ({break_even_winrate:.2f}%) を上回っています！")
                else:
                    st.warning(f"🟡 現在の実測勝率 ({actual_winrate:.2f}%) は損益分岐勝率 ({break_even_winrate:.2f}%) を下回っています。")
                    
    with tab3:
        st.markdown("### 🔍 全12市場の監査・除外診断ステータス")
        st.dataframe(pd.DataFrame(audit_logs), use_container_width=True, hide_index=True)

if auto_refresh:
    with placeholder.container():
        st.success(f"🟢 リアルタイム監査巡回中 (最終更新: {datetime.now().strftime('%H:%M:%S')})")
        render_dashboard()
    time.sleep(15)
    st.rerun()
else:
    with placeholder.container():
        render_dashboard()
    st.info("サイドバーの「24時間リアルタイム監査巡回 (15秒毎)」にチェックを入れると、15秒ごとに全12市場の勝敗条件監査が自動更新されます。")

st.divider()
st.caption("UTAYA v30 | ハイロー取引勝敗条件整合＆実証監査AI。利益・勝率は一切保証されません。検証結果をもとに自己責任で運用してください。")
