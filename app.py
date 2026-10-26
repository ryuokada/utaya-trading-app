import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
import joblib
from datetime import datetime
import os

# =========================================================
# UTAYA グローバル・インベストメンツ (v12 v10ベース・クイックモード値幅完全統合版)
# 思想: v10のデータリーク完全防止・監査基盤 ＋ XTrend等クイックモード値幅（BTC=50$等）の融合
# =========================================================

st.set_page_config(
    page_title="UTAYA Global Investments - v10/v12 Quick Mode Audit Engine",
    page_icon="🎯",
    layout="wide",
)

# 銘柄ごとのクイックモード想定値幅（エントリーから上下に動く基準ドル/ポイント単位）
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

PAYOUT_RATE = 0.88  # ペイアウト率88% (損益分岐勝率 約53.19%)
TARGET_ROI = 0.20   # 目標ROI 20% (10$ -> 12$)
DATA_DIR = Path("utaya_data_v10")  # v10のデータをそのまま継承・活用
DATA_DIR.mkdir(exist_ok=True)

GLOBAL_HISTORY_DIR = DATA_DIR / "history"
GLOBAL_HISTORY_DIR.mkdir(exist_ok=True)
ANALYSIS_LOG_FILE = DATA_DIR / "analysis_log_v10.csv"  # 既存ログファイルを維持

# -------------------- CSS --------------------
st.markdown(
    """
    <style>
    .genius-banner {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #00b09b;
    }
    .emergency-alert-card {
        background: rgba(255, 0, 0, 0.2);
        border-left: 6px solid #ff0000;
        padding: 20px;
        border-radius: 8px;
        color: #fff;
        margin-bottom: 20px;
    }
    .scenario-card-high {
        background: rgba(0, 176, 155, 0.15);
        border-left: 5px solid #00b09b;
        padding: 15px;
        border-radius: 5px;
    }
    .scenario-card-low {
        background: rgba(255, 65, 108, 0.15);
        border-left: 5px solid #ff416c;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 UTAYA v12 (v10ベース・クイックモード値幅スコープ統合エンジン)")
st.caption(
    "データリーク完全防止 | 銘柄別クイック値幅スコープ（BTC=50$等）同期 | Brierスコア校正"
)


# -------------------- ① データ取得・整合性エンジン --------------------

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


# -------------------- ② v10インデックス整合 ＋ クイック値幅スコープ評価 --------------------

def build_v12_features_and_labels(target_name, all_dfs):
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

    vol_mean = volume.rolling(14).mean()
    vol_std = volume.rolling(14).std().replace(0, 1.0)
    volume_z = ((volume - vol_mean) / vol_std).fillna(0.0)

    body_size = (close - open_p).abs()
    candle_range = (high - low).replace(0, 0.0001)
    body_ratio = body_size / candle_range

    features = pd.DataFrame(index=df_target.index)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["sma_ratio_10"] = (close - close.rolling(10).mean()) / close.rolling(10).mean()
    features["rsi"] = rsi
    features["volume_z"] = volume_z
    features["body_ratio"] = body_ratio
    features["volatility"] = returns.rolling(14).std()

    friction_signal = ((volume_z > 2.0) & (body_ratio < 0.25)).astype(float)
    features["emergency_friction"] = friction_signal

    for name, cfg in SYMBOLS_CONFIG.items():
        if name == target_name:
            continue
        df_other = all_dfs.get(name)
        if df_other is not None and not df_other.empty:
            other_close = df_other["Close"].reindex(df_target.index, method="ffill", limit=3).astype(float)
            short_tag = name.split(" ")[0].lower()
            features[f"{short_tag}_ret1"] = other_close.pct_change(1)

    labels = (close.shift(-1) > close).astype(int)

    combined = features.copy()
    combined["target"] = labels
    combined = combined.dropna()

    return combined.drop(columns=["target"]), combined["target"].astype(int)


def evaluate_quick_mode_outcome(entry_price, direction, future_candles, fixed_range_usd, hold_bars=5):
    """
    クイックモード固有の値幅スコープ（例: BTCなら上下50$）を基準に勝敗を判定。
    """
    if future_candles is None or len(future_candles) == 0:
        return "PENDING", "未来足の到着待ち", "PENDING", None

    upper_target = float(entry_price) + fixed_range_usd
    lower_target = float(entry_price) - fixed_range_usd

    for idx, candle in future_candles.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])

        hit_up = high >= upper_target
        hit_down = low <= lower_target

        if hit_up and hit_down:
            return "LOSS", "同一足で上下両方到達（クイックノイズ・逆発）", "VOLATILITY_NOISE", str(idx)
        if hit_up:
            if direction == "HIGH":
                return "WIN", "上方スコープ先着達成", "TREND_SUCCESS", str(idx)
            else:
                return "LOSS", "上方逆行（反発被弾）", "REVERSE_FAILURE_HIGH", str(idx)
        if hit_down:
            if direction == "LOW":
                return "WIN", "下方スコープ先着達成", "TREND_SUCCESS", str(idx)
            else:
                return "LOSS", "下方逆行（反発被弾）", "REVERSE_FAILURE_LOW", str(idx)

    if len(future_candles) >= hold_bars:
        return "TIMEOUT", "スコープ未到達（タイムアウト・膠着）", "TIMEOUT_RANGE", str(future_candles.index[-1])

    return "PENDING", "判定期間内未到達・データ待機中", "PENDING", None


def update_prediction_status(prediction, df_target, hold_bars, fixed_range_usd, stake_usd):
    if prediction["status"] != "PENDING":
        return prediction

    signal_time = pd.to_datetime(prediction["time"])
    future_candles = df_target.loc[signal_time:].iloc[1 : hold_bars + 1]

    result, reason, category, settle_time = evaluate_quick_mode_outcome(
        float(prediction["entry_price"]),
        prediction["direction"],
        future_candles,
        fixed_range_usd,
        hold_bars=hold_bars,
    )

    if result == "PENDING":
        return prediction

    prediction["status"] = "SETTLED"
    prediction["result"] = result
    prediction["reason"] = reason
    prediction["category"] = category
    prediction["settle_time"] = str(settle_time)
    
    if "EMERGENCY_HOLD" in str(prediction["trade_decision"]):
        prediction["pnl_usd"] = 0.0
    elif prediction["trade_decision"] == "PAPER_TRADE_CANDIDATE":
        prediction["pnl_usd"] = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd
    else:
        prediction["pnl_usd"] = 0.0

    return prediction


# -------------------- ③ v10厳格ウォークフォワード検証エンジン --------------------

def run_v12_walkforward(target_name, all_dfs, stake_usd, hold_bars, fixed_range_usd):
    df_target = all_dfs.get(target_name)
    if df_target is None or len(df_target) < 300:
        return pd.DataFrame()

    features_df, labels_s = build_v12_features_and_labels(target_name, all_dfs)
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
    progress_bar = st.progress(0)
    total_steps = len(features_df) - init_end - hold_bars

    for idx, i in enumerate(range(init_end, len(features_df) - hold_bars)):
        if total_steps > 0 and idx % 50 == 0:
            progress_bar.progress(min(1.0, max(0.0, idx / total_steps)))

        row_time = features_df.index[i]
        feat_row = features_df.iloc[i]
        is_emergency = bool(feat_row.get("emergency_friction", 0) > 0)

        X_pred = scaler.transform(np.array([feat_row.values]))
        
        try:
            if hasattr(model, "classes_") and len(model.classes_) == 2:
                probs = model.predict_proba(X_pred)[0]
                classes_list = list(model.classes_)
                p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
                p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
                model_status = "v12スコープ監査稼働中"
            else:
                p_low, p_high = 0.5, 0.5
                model_status = "初期化中"
        except Exception:
            p_low, p_high = 0.5, 0.5
            model_status = "エラー"

        direction = "HIGH" if p_high >= p_low else "LOW"
        probability_gap = abs(p_high - p_low)
        trade_decision = "EMERGENCY_HOLD (大口摩擦・逆発回避)" if is_emergency else ("PAPER_TRADE_CANDIDATE" if probability_gap >= 0.035 else "TRACK_ONLY")

        try:
            entry_idx_loc = df_target.index.get_loc(row_time) + 1
            entry_price = float(df_target["Open"].iloc[entry_idx_loc])
            future_window = df_target.iloc[entry_idx_loc : entry_idx_loc + hold_bars]
        except Exception:
            continue

        result, reason, category, settle_time = evaluate_quick_mode_outcome(
            entry_price, direction, future_window, fixed_range_usd, hold_bars=hold_bars
        )
        
        status = "SETTLED" if result != "PENDING" else "PENDING"
        pnl = 0.0
        if status == "SETTLED":
            if "EMERGENCY_HOLD" in str(trade_decision):
                pnl = 0.0
            elif trade_decision == "PAPER_TRADE_CANDIDATE":
                pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

        actual_label = int(labels_s.loc[row_time])
        try:
            model.partial_fit(X_pred, np.array([actual_label]), classes=np.array([0, 1]))
        except ValueError:
            model.partial_fit(X_pred, np.array([actual_label]))

        records.append({
            "time": str(row_time),
            "symbol": target_name,
            "direction": direction,
            "p_high": p_high,
            "p_low": p_low,
            "probability_gap": probability_gap,
            "trade_decision": trade_decision,
            "status": status,
            "result": result,
            "reason": reason,
            "category": category,
            "settle_time": str(settle_time),
            "entry_price": entry_price,
            "pnl_usd": pnl,
            "actual_label": actual_label,
        })

    progress_bar.empty()
    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")

    return res_df


# -------------------- ④ ダッシュボード・監査指標描画 --------------------

def show_audit_dashboard(res_df, stake_usd, fixed_range_usd):
    if res_df.empty:
        st.warning("検証データがありません。バックテストを実行してください。")
        return

    settled_df = res_df[res_df["status"] == "SETTLED"].copy()
    executed_df = settled_df[settled_df["trade_decision"] == "PAPER_TRADE_CANDIDATE"].copy()
    emergency_held_df = settled_df[settled_df["trade_decision"].str.contains("EMERGENCY_HOLD", na=False)].copy()

    total_exec = len(executed_df)
    total_emergency = len(emergency_held_df)
    
    wins = int((executed_df["result"] == "WIN").sum())
    win_rate = wins / total_exec * 100 if total_exec > 0 else 0.0
    total_pnl = float(executed_df["pnl_usd"].sum())
    be_rate = (1 / (1 + PAYOUT_RATE)) * 100

    current_capital = stake_usd + total_pnl
    actual_roi = (total_pnl / stake_usd) * 100 if stake_usd > 0 else 0.0

    gross_profit = float(executed_df[executed_df["pnl_usd"] > 0]["pnl_usd"].sum())
    gross_loss = abs(float(executed_df[executed_df["pnl_usd"] < 0]["pnl_usd"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Brierスコア（確率校正評価）
    brier = 0.0
    if not settled_df.empty and "actual_label" in settled_df.columns:
        try:
            brier = brier_score_loss(settled_df["actual_label"], settled_df["p_high"])
        except Exception:
            brier = 0.0

    st.markdown(
        f"""
        <div class="genius-banner">
            <h2>🎯 クイックモード値幅スコープ運用 (目標±${fixed_range_usd})</h2>
            <p>資金オーディット: 初期 ${stake_usd:.2f} ➔ 現在 ${current_capital:.2f} (実績ROI: {actual_roi:+.1f}%)</p>
            <p>確定勝負: <strong>{total_exec}回</strong> (勝率: <strong>{win_rate:.2f}%</strong> / 損益分岐: {be_rate:.2f}%) ｜ 🚨 **逆発・摩擦防御: {total_emergency}回** ｜ PF: <strong>{profit_factor:.2f}</strong></p>
            <p>確率校正 (Brier Score): <strong>{brier:.4f}</strong> (※0に近いほど確率予測が正確)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧭 方向予測の分布")
        st.bar_chart(res_df["direction"].value_counts().reindex(["HIGH", "LOW"], fill_value=0))
    with col2:
        st.subheader("📊 判定結果の内訳")
        if not executed_df.empty:
            st.bar_chart(executed_df["result"].value_counts())

    st.subheader("📈 資金成長曲線（累積損益）")
    chart_df = executed_df[["time", "pnl_usd"]].copy()
    if not chart_df.empty:
        chart_df["time"] = pd.to_datetime(chart_df["time"])
        chart_df["資産推移"] = stake_usd + chart_df["pnl_usd"].cumsum()
        chart_df = chart_df.set_index("time")
        st.line_chart(chart_df[["資産推移"]])
    else:
        st.info("確定した実戦データがありません。")

    st.subheader("📝 詳細プロ監査ログ (直近50件)")
    st.dataframe(res_df.sort_values("time", ascending=False).head(50), use_container_width=True, hide_index=True)


# -------------------- ⑤ サイドバー設定 --------------------

st.sidebar.header("🎯 v12 クイックモード監査設定")
selected_symbol = st.sidebar.selectbox("🎯 分析対象銘柄 (個別)", options=list(SYMBOLS_CONFIG.keys()), index=0)

cfg = SYMBOLS_CONFIG[selected_symbol]
default_range = cfg["quick_range_usd"]

# 会長がいつでも微調整できるクイックモード目標値幅
fixed_range_usd = st.sidebar.number_input(
    f"📏 クイックモード目標値幅 (±$ / pt)", 
    min_value=0.001, max_value=10000.0, value=float(default_range), step=0.01,
    format="%.2f",
    help="XTrend等のクイックモードでエントリーから上下に何ドル動いたら勝ちとするかのスコープ幅です。"
)

timeframe = st.sidebar.selectbox("⏱️ 分析時間軸", options=["1分足 (短期・クイック向け)", "1時間足 (中期)"], index=0)
interval_str = "1m" if "1分足" in timeframe else "1h"
period_str = "1d" if "1分足" in timeframe else "60d"

stake_usd = st.sidebar.number_input("💵 初期投資資金 ($)", 1.0, 10000.0, 10.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (足の本数)", 1, 20, 5)

if st.sidebar.button("🔄 学習ログをリセット", use_container_width=True):
    if ANALYSIS_LOG_FILE.exists():
        try:
            ANALYSIS_LOG_FILE.unlink()
        except OSError:
            pass
    st.sidebar.success("リセット完了しました。")
    st.rerun()

st.markdown(f"### 現在選択中の銘柄: **{selected_symbol}** (クイック基準スコープ: **±{fixed_range_usd}**)")

if st.button("▶️ クイックモード値幅スコープ監査を実行", use_container_width=True):
    with st.spinner(f"{selected_symbol} のクイックモード検証（±{fixed_range_usd}幅）を実行中..."):
        all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
        res_df = run_v12_walkforward(selected_symbol, all_dfs, stake_usd=stake_usd, hold_bars=hold_bars, fixed_range_usd=fixed_range_usd)
        show_audit_dashboard(res_df, stake_usd, fixed_range_usd)
elif ANALYSIS_LOG_FILE.exists():
    try:
        saved_df = pd.read_csv(ANALYSIS_LOG_FILE)
        if not saved_df.empty:
            show_audit_dashboard(saved_df, stake_usd, fixed_range_usd)
    except Exception:
        pass
