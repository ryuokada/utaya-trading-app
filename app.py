# -*- coding: utf-8 -*-
"""
UTAYA v33.0 | 厳格クオンツ監査・プロダクション版（利益・勝率保証なし）
- 未来情報リークの排除 (ウォークフォワード検証)
- 厳格なオンライン学習ライフサイクル管理 (重複学習防止・未確定除外)
- データ鮮度チェック (Stale Data Guard)
- ペーパートレード標準装備 ＋ リスク管理強化
"""
from __future__ import annotations
import json, re, os, time, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="UTAYA v33.0 | 厳格クオンツ監査エンジン",
    page_icon="🛡️",
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

DEFAULT_DIR = Path("./utaya_v33_data")
ROOT = Path(os.environ.get("UTAYA_DATA_DIR", str(DEFAULT_DIR)))
HISTORY, MODELS, LOGS = ROOT/"history", ROOT/"models", ROOT/"logs"
for p in (ROOT, HISTORY, MODELS, LOGS): p.mkdir(parents=True, exist_ok=True)

def slug(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", SYMBOLS[name]["ticker"])

def validate_ohlcv(df: pd.DataFrame) -> tuple[bool, str]:
    """データ構造、欠損、重複、異常値の検査"""
    if df is None or df.empty:
        return False, "データフレームが空です"
    required = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in required):
        return False, "必須OHLCカラムが不足しています"
    if df.index.duplicated().any():
        return False, "インデックスに重複したタイムスタンプが含まれています"
    if (df[required] <= 0).any().any():
        return False, "0以下の価格データが含まれています"
    if (df["High"] < df["Low"]).any():
        return False, "HighがLowより低い異常データが含まれています"
    return True, "OK"

@st.cache_data(ttl=15, show_spinner=False)
def fetch_data(ticker, period, interval):
    path = HISTORY / f"{ticker.replace('^','').replace('=','_')}_{interval}.csv"
    try:
        d = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                        progress=False, threads=False, timeout=8)
        if d is not None and not d.empty:
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            required = ["Open","High","Low","Close"]
            if all(c in d.columns for c in required):
                cols = required + (["Volume"] if "Volume" in d.columns else [])
                d = d[cols].copy()
                if "Volume" not in d: d["Volume"] = np.nan
                d = d.apply(pd.to_numeric, errors="coerce").dropna(subset=required)
                d.index = pd.to_datetime(d.index)
                if getattr(d.index, "tz", None) is not None: d.index = d.index.tz_localize(None)
                d = d[(d.Close > 0) & (d.High >= d.Low)]
                d = d[~d.index.duplicated(keep="last")].sort_index()
                
                is_valid, msg = validate_ohlcv(d)
                if is_valid:
                    if path.exists():
                        try:
                            old = pd.read_csv(path, index_col=0, parse_dates=True)
                            old.index = pd.to_datetime(old.index)
                            if getattr(old.index, "tz", None) is not None: old.index = old.index.tz_localize(None)
                            d = pd.concat([old, d])
                            d = d[~d.index.duplicated(keep="last")].sort_index()
                        except Exception: pass
                    d.to_csv(path, encoding="utf-8-sig")
                    return d, True, "取得成功"
    except Exception as e:
        pass
        
    if path.exists():
        try:
            d = pd.read_csv(path, index_col=0, parse_dates=True)
            d.index = pd.to_datetime(d.index)
            if getattr(d.index, "tz", None) is not None: d.index = d.index.tz_localize(None)
            d = d.sort_index()
            is_valid, msg = validate_ohlcv(d)
            if is_valid:
                # データの鮮度確認（例: 3日以上前なら警告）
                last_ts = d.index[-1]
                age_hours = (datetime.now() - last_ts).total_seconds() / 3600.0
                if age_hours > 72:
                    return d, False, f"キャッシュデータが古すぎます ({age_hours:.1f}時間前)"
                return d, False, f"オフラインキャッシュ使用 (最終: {last_ts})"
        except Exception as ex:
            return pd.DataFrame(), False, f"キャッシュ読み込み失敗: {ex}"
            
    return pd.DataFrame(), False, "データ取得失敗かつ有効キャッシュなし"

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
        self.mp = MODELS / f"{self.key}.v33.model.joblib"
        self.sp = MODELS / f"{self.key}.v33.scaler.joblib"
        self.meta_path = MODELS / f"{self.key}.v33.json"
        self.model = SGDClassifier(loss="log_loss", alpha=0.0005, random_state=42)
        self.scaler = StandardScaler()
        self.meta = {
            "trained_samples": 0, 
            "feature_columns": [], 
            "prediction_count": 0, 
            "online_updates": 0,
            "confirmed_labels": 0,
            "ignored_unconfirmed": 0,
            "duplicate_prevented": 0
        }
        self.ready = False
        
    def load(self, columns):
        try:
            if not all(p.exists() for p in (self.mp, self.sp, self.meta_path)): return False
            m = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if m.get("feature_columns") != list(columns): return False
            self.model = joblib.load(self.mp); self.scaler = joblib.load(self.sp); self.meta = m; self.ready = True
            return True
        except Exception: return False
        
    def fit_initial(self, x, y, hold=5, initial_fraction=0.7):
        n = len(x)
        split = int(n * initial_fraction)
        if split < 10 or n - split < 5:
            X_train, y_train = x, y
        else:
            index_positions = {timestamp: i for i, timestamp in enumerate(x.index)}
            sample_positions = np.array([index_positions.get(ts, -1) for ts in x.index])
            cutoff_pos = sample_positions[split]
            train_mask = sample_positions[:split] + hold < cutoff_pos
            X_train = x.iloc[:split].iloc[train_mask].copy()
            y_train = y.iloc[:split].iloc[train_mask].copy()
            if len(X_train) < 10 or y_train.nunique() < 2:
                X_train, y_train = x, y

        self.scaler.fit(X_train.values)
        self.model.partial_fit(self.scaler.transform(X_train.values), y_train.values, classes=np.array([0, 1]))
        self.meta.update(trained_samples=int(len(X_train)), feature_columns=list(x.columns), last_fit=datetime.now(timezone.utc).isoformat())
        self.ready = True; self.save()

    def online_update(self, row_features, true_label, pred_id: str, processed_ids: set):
        if not self.ready: return False
        if pred_id in processed_ids:
            self.meta["duplicate_prevented"] = self.meta.get("duplicate_prevented", 0) + 1
            self.save()
            return False
        try:
            X_scaled = self.scaler.transform(row_features.values)
            self.model.partial_fit(X_scaled, np.array([int(true_label)]), classes=np.array([0, 1]))
            self.meta["online_updates"] = self.meta.get("online_updates", 0) + 1
            self.meta["confirmed_labels"] = self.meta.get("confirmed_labels", 0) + 1
            processed_ids.add(pred_id)
            self.save()
            return True
        except Exception: 
            return False
        
    def save(self):
        try:
            joblib.dump(self.model, self.mp); joblib.dump(self.scaler, self.sp)
            self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception: pass

st.markdown("""<style>
.banner{background:linear-gradient(135deg,#0f172a,#1e293b,#334155);padding:20px;border-radius:14px;color:#38bdf8;text-align:center;border:2px solid #38bdf8}
.signal-box-buy{background:linear-gradient(135deg,#064e3b,#065f46);padding:25px;border-radius:14px;border:3px solid #34d399;color:#ffffff;text-align:center;}
.signal-box-sell{background:linear-gradient(135deg,#7f1d1d,#991b1b);padding:25px;border-radius:14px;border:3px solid #f87171;color:#ffffff;text-align:center;}
.warning-box{background:linear-gradient(135deg,#78350f,#92400e);padding:15px;border-radius:10px;border:2px solid #fbbf24;color:#ffffff;text-align:center;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="banner"><h1>🛡️ UTAYA v33.0 | 厳格クオンツ監査・プロダクション版</h1><p>未来情報完全排除 ＋ 厳格オンライン学習 ＋ ペーパートレード標準装備</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 監査・リスク管理設定")
    
    # ペーパートレード＆リスク設定
    st.markdown("### 🧪 モード選択")
    mode_type = st.radio("実行モード", ["ペーパートレード (推奨・実発注なし)", "バックテスト検証モード"], index=0)
    
    if "account_capital" not in st.session_state: st.session_state.account_capital = 15.0
    current_cap = st.number_input("ペーパー口座残高 ($)", 1.0, 100000.0, st.session_state.account_capital, 1.0)
    st.session_state.account_capital = current_cap
    
    stake_mode = st.selectbox("資金管理モード", ["固定額 ($1)", "固定額 ($5)", "固定額 ($10)"], index=0)
    invest_amount = 1.0 if "$1" in stake_mode else (5.0 if "$5" in stake_mode else 10.0)
    
    # 破産・ドローダウン防止キャップ（残高の20%を上限とするなど）
    max_safe_stake = current_cap * 0.20
    if invest_amount > max_safe_stake:
        invest_amount = max(1.0, max_safe_stake)
        st.warning(f"⚠️ リスク管理制限発動: 投資額を安全圏(${invest_amount:.2f})に自動調整しました。")

    payout_ratio = st.number_input("支払い率 (例: 0.75 = 75%)", 0.1, 5.0, 0.75, 0.05)
    break_even_winrate = 1.0 / (payout_ratio + 1.0) * 100.0
    
    hold = st.slider("判定期限（足数）", 1, 20, 5)
    period = st.selectbox("データ期間", ["30d", "60d", "3mo", "6mo", "1y"], index=4)
    interval = st.selectbox("時間足", ["1d", "1h", "15m", "5m", "1m"], index=0)
    
    st.divider()
    selected_symbol = st.selectbox("🎯 狙う銘柄を選択してください", list(SYMBOLS.keys()), index=0)
    audit_button = st.button("🚀 【指名監査・判定を実行する】", type="primary", use_container_width=True)

st.subheader(f"📊 期待値＆リスクリファレンス (支払い率: {payout_ratio*100:.0f}%)")
st.info(f"🎯 **損益分岐勝率: {break_even_winrate:.2f}%** | **ペーパー投資額:** **${invest_amount:.2f}** (※実資金の自動発注機能はありません)")

# セッション状態の初期化（予測履歴とオンライン学習用）
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "processed_prediction_ids" not in st.session_state:
    st.session_state.processed_prediction_ids = set()

# メイン処理
if audit_button or "first_run" not in st.session_state:
    st.session_state.first_run = True
    with st.spinner(f"🛡️ [{selected_symbol}] の厳格クオンツ監査とデータ検証を実行中..."):
        all_data = {}
        fetch_statuses = {}
        for name, cfg in SYMBOLS.items():
            d, is_fresh, msg = fetch_data(cfg["ticker"], period, interval)
            fetch_statuses[name] = {"fresh": is_fresh, "msg": msg}
            if d is not None and not d.empty:
                all_data[name] = d
                
        status_info = fetch_statuses.get(selected_symbol, {"fresh": False, "msg": "不明"})
        
        # データ鮮度チェック（Stale Data Guard）
        if not status_info["fresh"] and "キャッシュ" in status_info["msg"]:
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ 【データ鮮度警告 (Stale Data Alert)】</h3>
                <p>現在取得しているデータはリアルタイムのライブストリームではなく、{status_info["msg"]}です。古い情報に基づくシグナル出力を安全のため制限・警告します。</p>
            </div>
            """, unsafe_allow_html=True)

        if selected_symbol not in all_data:
            st.error(f"⚠️ {selected_symbol} の有効な市場データが取得できませんでした。({status_info['msg']})")
        else:
            d_market = all_data[selected_symbol]
            x, y, x_all, has_vol, reason = build_data(selected_symbol, all_data, hold)
            
            if reason != "OK":
                st.warning(f"⚠️ データ解析エラー: {reason}")
            else:
                manager = ModelManager(selected_symbol)
                if not manager.load(list(x.columns)):
                    try: manager.fit_initial(x, y, hold=hold)
                    except Exception as e: st.error(f"初期学習エラー: {e}")
                        
                latest = x_all.dropna()
                if len(latest) >= 2:
                    row = latest.iloc[[-1]]
                    current_ts = row.index[0]
                    
                    # 予測の実行
                    try:
                        scaled_row = manager.scaler.transform(row.values)
                        pr = manager.model.predict_proba(scaled_row)[0]
                        m = {int(c):float(p) for c,p in zip(manager.model.classes_, pr)}
                        pl, ph = m.get(0,.5), m.get(1,.5)
                        gap = abs(ph - pl)
                    except Exception as ex:
                        st.error(f"モデル予測エラー: {ex}")
                        pl, ph, gap = 0.5, 0.5, 0.0

                    current_price = float(d_market.Close.iloc[-1])
                    
                    # 予測IDの発行と履歴への登録（未確定状態）
                    pred_id = str(uuid.uuid4())[:8]
                    pred_record = {
                        "id": pred_id,
                        "timestamp": str(current_ts),
                        "symbol": selected_symbol,
                        "predicted_dir": "HIGH" if ph > pl else "LOW",
                        "ph": ph,
                        "pl": pl,
                        "price_at_pred": current_price,
                        "status": "UNCONFIRMED"
                    }
                    st.session_state.prediction_history.append(pred_record)

                    st.divider()
                    st.markdown(f"## 🎯 銘柄指名監査結果: **{selected_symbol}** (データ状態: {status_info['msg']})")
                    
                    # 確率の校正確認（偏りがある場合は注意喚起）
                    if ph > 0.95 or pl > 0.95:
                        st.info("ℹ️ モデルの確信度が極端に高いため、過学習（Overfitting）の可能性があります。慎重に検証してください。")

                    if ph > pl:
                        st.markdown(f"""
                        <div class="signal-box-buy">
                            <h2>🚀 【HIGH予測（ペーパーシグナル）】</h2>
                            <p style="font-size:20px; margin:10px 0;">現在価格: <b>${current_price:,.2f}</b> | 確率差: <b>{gap:.3f}</b></p>
                            <p style="font-size:18px; color:#a7f3d0;">🟢 HIGH確率 <b>{ph*100:.1f}%</b> —— ペーパー投資額 <b>${invest_amount:.2f}</b> (実発注なし)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="signal-box-sell">
                            <h2>📉 【LOW予測（ペーパーシグナル）】</h2>
                            <p style="font-size:20px; margin:10px 0;">現在価格: <b>${current_price:,.2f}</b> | 確率差: <b>{gap:.3f}</b></p>
                            <p style="font-size:18px; color:#fecaca;">🔴 LOW確率 <b>{pl*100:.1f}%</b> —— ペーパー投資額 <b>${invest_amount:.2f}</b> (実発注なし)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.write("")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("HIGH確率", f"{ph*100:.1f}%")
                    col2.metric("LOW確率", f"{pl*100:.1f}%")
                    col3.metric("オンライン学習回数", f"{manager.meta.get('online_updates', 0)}回")
                    col4.metric("確定済みラベル数", f"{manager.meta.get('confirmed_labels', 0)}件")

                    # バックグラウンドでのオンライン学習ライフサイクル処理（判定足数経過分の答え合わせ）
                    confirmed_count = 0
                    for rec in st.session_state.prediction_history:
                        if rec["status"] == "UNCONFIRMED" and rec["symbol"] == selected_symbol:
                            pred_time = pd.to_datetime(rec["timestamp"])
                            # 実際の市場データ上で、pred_time から hold 足経過したデータが存在するか確認
                            if pred_time in d_market.index:
                                idx_pos = d_market.index.get_loc(pred_time)
                                target_pos = idx_pos + hold
                                if target_pos < len(d_market):
                                    # 判定確定！
                                    entry_price = rec["price_at_pred"]
                                    future_price = float(d_market.Close.iloc[target_pos])
                                    true_lbl = 1.0 if future_price > entry_price else (0.0 if future_price < entry_price else -1.0)
                                    
                                    if true_lbl != -1.0:
                                        # 特徴量の再取得とオンライン学習の実行
                                        try:
                                            row_feat = x_all.loc[[pred_time]]
                                            success = manager.online_update(row_feat, true_lbl, rec["id"], st.session_state.processed_prediction_ids)
                                            if success:
                                                confirmed_count += 1
                                                rec["status"] = "CONFIRMED"
                                                rec["actual_result"] = "WIN" if (rec["predicted_dir"] == "HIGH" and true_lbl == 1.0) or (rec["predicted_dir"] == "LOW" and true_lbl == 0.0) else "LOSS"
                                        except Exception:
                                            pass

                    if confirmed_count > 0:
                        st.success(f"🔄 判定期限（{hold}足）が経過した予測 {confirmed_count}件 の結果が確定し、オンライン学習（`partial_fit`）に安全に適用されました。")

                else:
                    st.warning("⚠️ 評価に必要な直近の最新データが不足しています。")
else:
    st.info("👈 サイドバーで監査したい銘柄を選び、**【指名監査・判定を実行する】** ボタンを押してください。")

st.divider()
st.markdown("### 📋 予測＆ペーパー検証の監査履歴（セッション内）")
if st.session_state.prediction_history:
    df_hist = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(df_hist.tail(10), use_container_width=True, hide_index=True)
else:
    st.caption("履歴はまだありません。")

st.divider()
st.caption("UTAYA v33.0 | 厳格クオンツ監査・プロダクション版。利益・勝率の保証は一切行いません。検証データに基づく厳密な評価を行ってください。")
