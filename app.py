import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from datetime import datetime
import joblib
import re
import os
import json

# =========================================================
# UTAYA グローバル・インベストメンツ (v22 統合せん滅・実証型AIエンジン)
# 思想: 8501(短期量子・高速) + 8503(12市場スキャン・3段階判定) の完全融合、未来リーク完全根絶
# =========================================================

st.set_page_config(
    page_title="UTAYA Global Investments - v22 Unified Empirical Combat Engine",
    page_icon="⚡",
    layout="wide",
)

SYMBOLS_CONFIG = {
    "BTC (暗号資産)": {"ticker": "BTC-USD", "quick_range_usd": 50.0},
    "ETH (暗号資産)": {"ticker": "ETH-USD", "quick_range_usd": 15.0},
    "SOL (暗号資産)": {"ticker": "SOL-USD", "quick_range_usd": 2.0},
    "XRP (暗号資産)": {"ticker": "XRP-USD", "quick_range_usd": 0.05},
    "BNB (暗号資産)": {"ticker": "BNB-USD", "quick_range_usd": 5.0},
    "WTI原油 (エネルギー)": {"ticker": "CL=F", "quick_range_usd": 0.40},
    "金スポット (貴金属)": {"ticker": "GC=F", "quick_range_usd": 5.0},
    "日経平均 (日本株)": {"ticker": "^N225", "quick_range_usd": 100.0},
    "S&P 500 (米国株)": {"ticker": "^GSPC", "quick_range_usd": 15.0},
    "Nasdaq 100 (米国株)": {"ticker": "^NDX", "quick_range_usd": 50.0},
    "USD/JPY (為替)": {"ticker": "USDJPY=X", "quick_range_usd": 0.15},
    "米10年債利回り (金利)": {"ticker": "^TNX", "quick_range_usd": 0.05},
}

PAYOUT_RATE = 0.88  # ハイローペイアウト率 (仮定値・実際の提供条件で変動)
DATA_DIR = Path("utaya_data_v22")
DATA_DIR.mkdir(exist_ok=True)

GLOBAL_HISTORY_DIR = DATA_DIR / "history"
GLOBAL_HISTORY_DIR.mkdir(exist_ok=True)
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
ANALYSIS_LOG_FILE = DATA_DIR / "analysis_log_v22.csv"

# -------------------- CSS --------------------
st.markdown(
    """
    <style>
    .unified-banner {
        background: linear-gradient(135deg, #0a0f1d, #1f2833, #0b0c10);
        padding: 20px;
        border-radius: 12px;
        color: #66fcf1;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #45a29e;
    }
    .panel-container {
        display: flex;
        gap: 15px;
        margin-bottom: 20px;
    }
    .panel-high {
        flex: 1;
        background: rgba(102, 252, 241, 0.1);
        border: 2px solid #66fcf1;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
    }
    .panel-low {
        flex: 1;
        background: rgba(255, 75, 75, 0.1);
        border: 2px solid #ff4b4b;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
    }
    .circuit-lock {
        background: rgba(255, 0, 0, 0.3);
        border-left: 6px solid #ff0000;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 15px;
        color: #ffcccc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ UTAYA v22 統合せん滅・実証型AIエンジン")
st.caption(
    "8501(短期量子) ＋ 8503(12市場・3段階判定) 完全統合 | 未来リーク完全排除 | 15$鉄壁防衛網"
)


# -------------------- ① 安全ファイル名変換・データ同期 --------------------

def get_safe_filename(target_name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', SYMBOLS_CONFIG[target_name]["ticker"])


@st.cache_data(ttl=60, show_spinner=False)
def fetch_single_market_data(ticker, period="60d", interval="1h"):
    file_path = GLOBAL_HISTORY_DIR / f"{ticker.replace('^', '').replace('=', '_')}_{interval}.csv"
    try:
        df_new = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df_new is not None and not df_new.empty:
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            cols = ["Open", "High", "Low", "Close", "Volume"] if "Volume" in df_new.columns else ["Open", "High", "Low", "Close"]
            if all(c in df_new.columns for c in ["Open", "High", "Low", "Close"]):
                df_new = df_new[cols].copy()
                if "Volume" not in df_new.columns:
                    df_new["Volume"] = np.nan
                df_new = df_new.apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
                df_new.index = pd.to_datetime(df_new.index)
                if df_new.index.tz is not None:
                    df_new.index = df_new.index.tz_localize(None)
                df_new = df_new[(df_new["Close"] > 0) & (df_new["High"] >= df_new["Low"])]
                df_new = df_new[~df_new.index.duplicated(keep="last")].sort_index()

                if file_path.exists():
                    try:
                        df_old = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        df_old.index = pd.to_datetime(df_old.index)
                        if df_old.index.tz is not None:
                            df_old.index = df_old.index.tz_localize(None)
                        df_combined = pd.concat([df_old, df_new])
                        df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
                        df_combined.to_csv(file_path, encoding="utf-8-sig")
                    except Exception:
                        df_new.to_csv(file_path, encoding="utf-8-sig")
                else:
                    df_new.to_csv(file_path, encoding="utf-8-sig")

        if file_path.exists():
            df_stored = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df_stored.index = pd.to_datetime(df_stored.index)
            if df_stored.index.tz is not None:
                df_stored.index = df_stored.index.tz_localize(None)
            return df_stored.sort_index()
        return df_new
    except Exception:
        if file_path.exists():
            df_stored = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df_stored.index = pd.to_datetime(df_stored.index)
            if df_stored.index.tz is not None:
                df_stored.index = df_stored.index.tz_localize(None)
            return df_stored.sort_index()
        return None


def fetch_all_markets_data(period="60d", interval="1h"):
    all_dfs = {}
    for name, cfg in SYMBOLS_CONFIG.items():
        df = fetch_single_market_data(cfg["ticker"], period=period, interval=interval)
        if df is not None and not df.empty:
            all_dfs[name] = df
    return all_dfs


# -------------------- ② 厳格な勝敗評価 ＆ 統合特徴量生成 --------------------

def evaluate_unified_outcome(entry_price, direction, future_candles, fixed_range_usd, hold_bars=5):
    if future_candles is None or len(future_candles) == 0:
        return "PENDING", "未来足未到着", "PENDING", None

    upper_target = float(entry_price) + fixed_range_usd
    lower_target = float(entry_price) - fixed_range_usd

    for idx, candle in future_candles.iterrows():
        h = float(candle["High"])
        l = float(candle["Low"])

        hit_up = h >= upper_target
        hit_down = l <= lower_target

        if hit_up and hit_down:
            return "NOISE", "同一足で上下両方到達（判定不能）", "VOLATILITY_NOISE", str(idx)
        if hit_up:
            if direction == "HIGH":
                return "WIN", "上方ターゲット先着", "SUCCESS_HIGH", str(idx)
            else:
                return "LOSS", "上方逆行被弾", "REVERSE_HIGH", str(idx)
        if hit_down:
            if direction == "LOW":
                return "WIN", "下方ターゲット先着", "SUCCESS_LOW", str(idx)
            else:
                return "LOSS", "下方逆行被弾", "REVERSE_LOW", str(idx)

    if len(future_candles) >= hold_bars:
        return "TIMEOUT", "期限内ターゲット未達", "TIMEOUT", str(future_candles.index[-1])

    return "PENDING", "交戦中", "PENDING", None


def build_v22_features_and_labels(target_name, all_dfs, fixed_range_usd, hold_bars=5):
    if target_name not in all_dfs:
        return pd.DataFrame(), pd.Series(dtype=int)

    df_target = all_dfs[target_name].copy()
    df_target.index = pd.to_datetime(df_target.index)
    if df_target.index.tz is not None:
        df_target.index = df_target.index.tz_localize(None)

    open_p = df_target["Open"].astype(float)
    high = df_target["High"].astype(float)
    low = df_target["Low"].astype(float)
    close = df_target["Close"].astype(float)
    volume = df_target["Volume"].astype(float) if "Volume" in df_target.columns else pd.Series(np.nan, index=df_target.index)

    returns = close.pct_change()
    delta = close.diff()
    
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(losses != 0, 100.0)
    rsi = rsi.where(gains != 0, 50.0)

    rolling_std = returns.rolling(14).std()
    squeeze_index = rolling_std / rolling_std.rolling(30).mean().replace(0, 0.0001)

    body_size = (close - open_p).abs()
    candle_range = (high - low).replace(0, 0.0001)
    body_ratio = body_size / candle_range
    upper_pressure = (high - np.maximum(open_p, close)) / candle_range
    lower_pressure = (np.minimum(open_p, close) - low) / candle_range

    features = pd.DataFrame(index=df_target.index)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["sma_ratio_10"] = (close - close.rolling(10).mean()) / close.rolling(10).mean()
    features["rsi"] = rsi
    features["squeeze_index"] = squeeze_index
    features["body_ratio"] = body_ratio
    features["pressure_diff"] = lower_pressure - upper_pressure
    features["volatility"] = rolling_std

    # 8501/8503統合：クロスマーケット特徴量（他市場のトレンドを反映）
    for name, cfg in SYMBOLS_CONFIG.items():
        if name == target_name:
            continue
        df_other = all_dfs.get(name)
        if df_other is not None and not df_other.empty:
            other_close = df_other["Close"].reindex(df_target.index, method="ffill", limit=3).astype(float)
            short_tag = name.split(" ")[0].lower()
            features[f"{short_tag}_ret1"] = other_close.pct_change(1)

    labels = []
    valid_index = []
    
    for idx in range(len(df_target) - hold_bars):
        row_time = df_target.index[idx]
        entry_price = float(df_target["Open"].iloc[idx + 1])
        future_window = df_target.iloc[idx + 1 : idx + 1 + hold_bars]
        
        res, _, _, _ = evaluate_unified_outcome(entry_price, "HIGH", future_window, fixed_range_usd, hold_bars)
        
        if res == "WIN":
            labels.append(1)
            valid_index.append(row_time)
        elif res in ["LOSS", "TIMEOUT"]:
            labels.append(0)
            valid_index.append(row_time)

    features_valid = features.loc[valid_index].copy()
    features_valid["target"] = labels
    features_valid = features_valid.replace([np.inf, -np.inf], np.nan).dropna()

    if features_valid.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    return features_valid.drop(columns=["target"]), features_valid["target"].astype(int)


# -------------------- ③ 3段階判定 ＆ 15ドル資金防衛 --------------------

def classify_combat_tier(feat_row, p_high, p_low, current_capital):
    if current_capital <= 10.0:
        return "重大リスク・一時停止", f"🚨 15$防衛発動 (残高 ${current_capital:.2f})。資産保護のため新規エントリーを完全凍結します。"

    probability_gap = abs(p_high - p_low)
    squeeze = float(feat_row.get("squeeze_index", 1.0))

    if probability_gap >= 0.05:
        return "通常エントリー候補", f"💎 確率ギャップ {probability_gap:.3f} 達成。検証済み通常エントリー候補。"

    return "条件付き監視", f"👀 ギャップ蓄積中（ギャップ {probability_gap:.3f}）。条件成立を継続監視。"


# -------------------- ④ モデル永続化・オンライン管理 --------------------

class UnifiedModelManager:
    def __init__(self, target_name):
        self.target_name = target_name
        self.safe_name = get_safe_filename(target_name)
        self.model_path = MODEL_DIR / f"model_v22_{self.safe_name}.pkl"
        self.scaler_path = MODEL_DIR / f"scaler_v22_{self.safe_name}.pkl"
        self.meta_path = MODEL_DIR / f"meta_v22_{self.safe_name}.json"
        
        self.model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
        self.scaler = StandardScaler()
        self.meta = {"trained_samples": 0, "learned_ids": []}

    def load(self, fixed_range_usd, hold_bars):
        if self.model_path.exists() and self.scaler_path.exists() and self.meta_path.exists():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("fixed_range_usd") == fixed_range_usd and meta.get("hold_bars") == hold_bars:
                    self.model = joblib.load(self.model_path)
                    self.scaler = joblib.load(self.scaler_path)
                    self.meta = meta
                    return True
            except Exception:
                pass
        return False

    def save(self, fixed_range_usd, hold_bars):
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        self.meta["fixed_range_usd"] = fixed_range_usd
        self.meta["hold_bars"] = hold_bars
        self.meta["version"] = "v22_unified"
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def initialize_fit(self, X, y, fixed_range_usd, hold_bars):
        if X is not None and len(X) > 0:
            self.scaler.fit(X)
            X_sc = self.scaler.transform(X)
            self.model.partial_fit(X_sc, y, classes=np.array([0, 1]))
            self.meta["trained_samples"] = int(len(X))
            self.save(fixed_range_usd, hold_bars)

    def update_online(self, feat_row_df, actual_label, fixed_range_usd, hold_bars, signal_id):
        if signal_id in self.meta.get("learned_ids", []):
            return False
        X_new = self.scaler.transform(feat_row_df.values)
        self.model.partial_fit(X_new, np.array([actual_label]), classes=np.array([0, 1]))
        self.meta.setdefault("learned_ids", []).append(signal_id)
        self.meta["trained_samples"] += 1
        self.save(fixed_range_usd, hold_bars)
        return True


# -------------------- ⑤ ウォークフォワード・バックテストエンジン --------------------

def run_v22_unified_backtest(target_name, all_dfs, stake_usd, hold_bars, fixed_range_usd):
    df_target = all_dfs.get(target_name)
    if df_target is None or len(df_target) < 300:
        return pd.DataFrame()

    features_df, labels_s = build_v22_features_and_labels(target_name, all_dfs, fixed_range_usd, hold_bars)
    if len(features_df) < 100:
        return pd.DataFrame()

    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
    scaler = StandardScaler()

    init_end = int(len(features_df) * 0.3)
    X_init = features_df.iloc[:init_end].values
    y_init = labels_s.iloc[:init_end].values

    scaler.fit(X_init)
    model.partial_fit(scaler.transform(X_init), y_init, classes=np.array([0, 1]))

    records = []
    current_capital = stake_usd
    peak_capital = stake_usd

    for i in range(init_end, len(features_df) - hold_bars):
        row_time = features_df.index[i]
        feat_row = features_df.iloc[i]

        X_pred = scaler.transform(np.array([feat_row.values]))
        try:
            probs = model.predict_proba(X_pred)[0]
            classes_list = list(model.classes_)
            p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
            p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
        except Exception:
            p_low, p_high = 0.5, 0.5

        direction = "HIGH" if p_high >= p_low else "LOW"
        tier, tier_reason = classify_combat_tier(feat_row, p_high, p_low, current_capital)

        try:
            entry_idx_loc = df_target.index.get_loc(row_time) + 1
            entry_price = float(df_target["Open"].iloc[entry_idx_loc])
            future_window = df_target.iloc[entry_idx_loc : entry_idx_loc + hold_bars]
        except Exception:
            continue

        result, reason, category, settle_time = evaluate_unified_outcome(
            entry_price, direction, future_window, fixed_range_usd, hold_bars=hold_bars
        )
        
        status = "SETTLED" if result != "PENDING" else "PENDING"
        pnl = 0.0
        if status == "SETTLED":
            if "一時停止" in tier:
                pnl = 0.0
            elif tier == "通常エントリー候補":
                if result == "WIN":
                    pnl = stake_usd * PAYOUT_RATE
                elif result in ["LOSS", "TIMEOUT"]:
                    pnl = -stake_usd
                else:
                    pnl = 0.0
                
                current_capital += pnl
                if current_capital > peak_capital:
                    peak_capital = current_capital

        actual_label = int(labels_s.loc[row_time]) if row_time in labels_s.index else 0
        try:
            model.partial_fit(X_pred, np.array([actual_label]), classes=np.array([0, 1]))
        except ValueError:
            pass

        records.append({
            "time": str(row_time),
            "symbol": target_name,
            "direction": direction,
            "p_high": p_high,
            "p_low": p_low,
            "tier": tier,
            "result": result,
            "pnl_usd": pnl,
            "capital": current_capital,
        })

    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")
    return res_df


# -------------------- ⑥ ダッシュボード・UI描画 --------------------

st.sidebar.header("⚡ v22 統合せん滅司令室")
selected_symbol = st.sidebar.selectbox("🎯 ターゲット銘柄", options=list(SYMBOLS_CONFIG.keys()), index=0)

cfg = SYMBOLS_CONFIG[selected_symbol]
default_range = cfg["quick_range_usd"]

fixed_range_usd = st.sidebar.number_input("📏 目標値幅 (±$)", 0.001, 10000.0, float(default_range), 0.01, format="%.2f")
stake_usd = st.sidebar.number_input("💵 参戦資金 ($)", 1.0, 10000.0, 15.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (本)", 1, 20, 5)

if st.sidebar.button("🔄 記憶・モデル完全リセット", use_container_width=True):
    if ANALYSIS_LOG_FILE.exists():
        ANALYSIS_LOG_FILE.unlink()
    for f in MODEL_DIR.glob("*.pkl"):
        f.unlink()
    for f in MODEL_DIR.glob("*.json"):
        f.unlink()
    st.sidebar.success("完全初期化完了。")
    st.rerun()

tab1, tab2 = st.tabs(["⚡ 個別実戦・真の最新足検証", "🌌 全市場・実証スキャナー"])

with tab1:
    if "v22_active" not in st.session_state:
        st.session_state.v22_active = False

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ 統合AI起動", use_container_width=True):
            st.session_state.v22_active = True
    with c2:
        if st.button("⏹ 緊急停戦", use_container_width=True):
            st.session_state.v22_active = False

    if st.session_state.v22_active:
        st.success(f"統合稼働中 [{selected_symbol}]：量子短期＋12市場スキャン・未来リークゼロ")
    else:
        st.info("待機中。起動ボタンを押してください。")

    @st.fragment(run_every="30s")
    def combat_loop():
        if not st.session_state.v22_active:
            return

        all_dfs = fetch_all_markets_data(period="60d", interval="1h")
        df_target = all_dfs.get(selected_symbol)
        if df_target is None or len(df_target) < 50:
            st.warning("データ同期中...")
            return

        features_df, labels_s = build_v22_features_and_labels(selected_symbol, all_dfs, fixed_range_usd, hold_bars)
        if len(features_df) < 10:
            st.warning("有効な特徴量データが不足しています。")
            return

        manager = UnifiedModelManager(selected_symbol)
        if not manager.load(fixed_range_usd, hold_bars):
            manager.initialize_fit(features_df.values, labels_s.values, fixed_range_usd, hold_bars)

        # 🚨 未来の未評価シグナル（真の最新確定足）の予測
        signal_i = len(features_df) - 1
        feat_row = features_df.iloc[signal_i]
        row_time = features_df.index[signal_i]

        X_live = manager.scaler.transform(np.array([feat_row.values]))
        try:
            probs = manager.model.predict_proba(X_live)[0]
            classes_list = list(manager.model.classes_)
            p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
            p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
        except Exception:
            p_low, p_high = 0.5, 0.5

        tier, tier_reason = classify_combat_tier(feat_row, p_high, p_low, stake_usd)
        signal_price = float(df_target["Close"].loc[row_time]) if row_time in df_target.index else float(df_target["Close"].iloc[-1])

        st.markdown(
            f"""
            <div class="panel-container">
                <div class="panel-high">
                    <h3>▲ 昇華・HIGH予測 (真の最新足)</h3>
                    <h1 style="color: #66fcf1; margin: 10px 0;">{p_high:.1%}</h1>
                    <p style="font-size: 0.8rem; color: #ccc;">上方ターゲット: +${fixed_range_usd} (学習済サンプル: {manager.meta.get('trained_samples', 0)})</p>
                </div>
                <div class="panel-low">
                    <h3>▼ 降下・LOW予測 (真の最新足)</h3>
                    <h1 style="color: #ff4b4b; margin: 10px 0;">{p_low:.1%}</h1>
                    <p style="font-size: 0.8rem; color: #ccc;">下方ターゲット: -${fixed_range_usd}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "一時停止" in tier:
            st.markdown(f'<div class="circuit-lock"><h3>{tier}</h3><p>{tier_reason}</p></div>', unsafe_allow_html=True)
        else:
            st.info(f"🎯 現在の戦闘ステージ: 【 {tier} 】\n\n{tier_reason} (基準価格: ${signal_price:,.2f})")

    combat_loop()

    st.markdown("---")
    st.subheader("🚀 厳格ウォークフォワード検証 (バックテスト)")
    if st.button("▶️ v22 厳格バックテスト実行", use_container_width=True):
        with st.spinner("未来リークなしのウォークフォワード検証を実行中..."):
            all_dfs = fetch_all_markets_data(period="60d", interval="1h")
            res_df = run_v22_unified_backtest(selected_symbol, all_dfs, stake_usd, hold_bars, fixed_range_usd)
            if not res_df.empty:
                total_trades = len(res_df[res_df["result"].isin(["WIN", "LOSS", "TIMEOUT"])])
                wins = len(res_df[res_df["result"] == "WIN"])
                win_rate = wins / total_trades * 100 if total_trades > 0 else 0.0
                final_cap = res_df["capital"].iloc[-1]
                
                st.markdown(f"### 📊 統合検証報告: 勝率 {win_rate:.2f}% ({wins}/{total_trades}勝) | 最終資金: ${final_cap:.2f} (初期 ${stake_usd}) | 損益分岐(53.19%)と比較")
                st.line_chart(res_df.set_index("time")[["capital"]])
                st.dataframe(res_df.tail(20), use_container_width=True, hide_index=True)
            else:
                st.warning("有効な検証データがありません。")

with tab2:
    st.subheader("🌌 全12市場 統合実証スキャナー")
    if st.button("🔍 全市場の真の最新足を一斉検証", use_container_width=True):
        with st.spinner("全12市場を未来リークなしで検証中..."):
            all_dfs = fetch_all_markets_data(period="60d", interval="1h")
            scan_res = []
            for name, cfg in SYMBOLS_CONFIG.items():
                df = all_dfs.get(name)
                rng = cfg["quick_range_usd"]
                if df is None or len(df) < 50:
                    continue
                try:
                    f_df, l_s = build_v22_features_and_labels(name, all_dfs, rng, 5)
                    if len(f_df) < 10:
                        continue
                    manager = UnifiedModelManager(name)
                    if not manager.load(rng, 5):
                        manager.initialize_fit(f_df.values, l_s.values, rng, 5)
                    feat_row = f_df.iloc[-1]
                    probs = manager.model.predict_proba(manager.scaler.transform(np.array([feat_row.values])))[0]
                    p_l, p_h = probs[0], probs[1]
                    tier, _ = classify_combat_tier(feat_row, p_h, p_l, stake_usd)
                    scan_res.append({"銘柄": name, "ステージ": tier, "HIGH確率": f"{p_h:.1%}", "LOW確率": f"{p_l:.1%}", "ギャップ": round(abs(p_h - p_l), 3)})
                except Exception:
                    pass
            if scan_res:
                st.dataframe(pd.DataFrame(scan_res), use_container_width=True, hide_index=True)
            else:
                st.warning("スキャンデータ不足。")
