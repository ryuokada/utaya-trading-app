import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import os

# =========================================================
# UTAYA 世界市場スキャナーAIシステム (監査・方向予測分離プロ版)
# 思想: 50/50偽装排除、方向予測と取引判断の分離、PENDING動的追跡
# =========================================================

st.set_page_config(
    page_title="UTAYA Full-Market Pro Audit System v3",
    page_icon="🌌",
    layout="wide",
)

SYMBOLS = {
    "BTC (暗号資産)": "BTC-USD",
    "ETH (暗号資産)": "ETH-USD",
    "SOL (暗号資産)": "SOL-USD",
    "XRP (暗号資産)": "XRP-USD",
    "BNB (暗号資産)": "BNB-USD",
    "S&P 500 (米国株)": "^GSPC",
    "Nasdaq 100 (米国株)": "^NDX",
    "USD/JPY (為替)": "USDJPY=X",
    "米10年債利回り (金利)": "^TNX",
}

PAYOUT_RATE = 0.88  # ペイアウト率88% -> 損益分岐勝率 約53.19%
DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(exist_ok=True)

GLOBAL_HISTORY_DIR = DATA_DIR / "full_market_history_v3"
GLOBAL_HISTORY_DIR.mkdir(exist_ok=True)
VIRTUAL_TRADES_FILE = DATA_DIR / "full_market_virtual_trades_v3.csv"
ANALYSIS_LOG_FILE = DATA_DIR / "full_market_analysis_log_v3.csv"

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
    }
    .scenario-card-high {
        background: rgba(0, 176, 155, 0.1);
        border-left: 5px solid #00b09b;
        padding: 15px;
        border-radius: 5px;
    }
    .scenario-card-low {
        background: rgba(255, 65, 108, 0.1);
        border-left: 5px solid #ff416c;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌌 UTAYA 世界市場スキャナーAI (完全分離・動的追跡プロ版)")
st.caption(
    "50/50偽装排除 | 方向予測の常時提示 | PENDING動的ライフサイクル管理 | PF検証"
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
            cols = ["Open", "High", "Low", "Close"]
            if all(c in df_new.columns for c in cols):
                df_new = df_new[cols].copy()
                df_new = df_new.apply(pd.to_numeric, errors="coerce").dropna()
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
    for name, ticker in SYMBOLS.items():
        df = fetch_single_market_data(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            all_dfs[name] = df
    return all_dfs


# -------------------- ② 特徴量と方向・取引判定ロジック --------------------

def build_full_integrated_features(target_name, all_dfs):
    if target_name not in all_dfs:
        return pd.DataFrame()

    df_target = all_dfs[target_name].copy()
    df_target.index = pd.to_datetime(df_target.index)
    if df_target.index.tz is not None:
        df_target.index = df_target.index.tz_localize(None)

    close = df_target["Close"].astype(float)
    returns = close.pct_change()
    delta = close.diff()
    
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(losses != 0, 100.0)
    rsi = rsi.where(gains != 0, 50.0)

    features = pd.DataFrame(index=df_target.index)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["return_6"] = close.pct_change(6)
    features["sma_ratio_10"] = (close - close.rolling(10).mean()) / close.rolling(10).mean()
    features["sma_ratio_30"] = (close - close.rolling(30).mean()) / close.rolling(30).mean()
    features["rsi"] = rsi
    features["volatility"] = returns.rolling(14).std()

    for name, df_other in all_dfs.items():
        if name == target_name:
            continue
        df_other = df_other.copy()
        df_other.index = pd.to_datetime(df_other.index)
        if df_other.index.tz is not None:
            df_other.index = df_other.index.tz_localize(None)

        other_close = df_other["Close"].reindex(df_target.index, method="ffill").astype(float)
        short_tag = name.split(" ")[0].lower()
        features[f"{short_tag}_ret1"] = other_close.pct_change(1)
        features[f"{short_tag}_sma10"] = (other_close - other_close.rolling(10).mean()) / other_close.rolling(10).mean()

    return features.replace([np.inf, -np.inf], np.nan)


def predict_direction(p_high, p_low):
    """方向予測は必ずHIGHかLOWを返す。"""
    direction = "HIGH" if p_high >= p_low else "LOW"
    confidence = max(p_high, p_low)
    probability_gap = abs(p_high - p_low)
    return {
        "direction": direction,
        "p_high": p_high,
        "p_low": p_low,
        "confidence": confidence,
        "probability_gap": probability_gap,
    }


def assess_trade(signal, min_gap=0.04):
    """取引候補の判定。方向予測とは別の処理。"""
    if signal["probability_gap"] >= min_gap:
        return "PAPER_TRADE_CANDIDATE"
    return "TRACK_ONLY"


def evaluate_outcome(entry_price, direction, future_candles, target_pct=0.01, fee_pct=0.0005):
    """±目標変動率への先着判定。"""
    if future_candles is None or len(future_candles) == 0:
        return "PENDING", "未来足の到着待ち", "PENDING", None

    upper = float(entry_price) * (1.0 + target_pct + fee_pct)
    lower = float(entry_price) * (1.0 - target_pct - fee_pct)

    for idx, candle in future_candles.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])

        hit_up = high >= upper
        hit_down = low <= lower

        if hit_up and hit_down:
            return "LOSS", "同一足で上下両方到達（ボラティリティノイズ）", "VOLATILITY_NOISE", str(idx)
        if hit_up:
            if direction == "HIGH":
                return "WIN", "上目標先着：トレンド順行", "TREND_SUCCESS", str(idx)
            else:
                return "LOSS", "逆方向到達：逆行失敗", "REVERSE_FAILURE_HIGH", str(idx)
        if hit_down:
            if direction == "LOW":
                return "WIN", "下目標先着：下落予測的中", "TREND_SUCCESS", str(idx)
            else:
                return "LOSS", "逆方向到達：下落逆行失敗", "REVERSE_FAILURE_LOW", str(idx)

    return "PENDING", "判定期間内未到達・データ待機中", "PENDING", None


def update_prediction_status(prediction, df_target, hold_bars, target_pct, fee_pct):
    if prediction["status"] != "PENDING":
        return prediction

    signal_time = pd.to_datetime(prediction["time"])
    future_candles = df_target.loc[signal_time:].iloc[1 : hold_bars + 1]

    if len(future_candles) < hold_bars:
        prediction["reason"] = "未来足の到着待ち"
        return prediction

    result, reason, category, settle_time = evaluate_outcome(
        float(prediction["entry_price"]),
        prediction["direction"],
        future_candles,
        target_pct,
        fee_pct,
    )

    if result == "PENDING":
        return prediction

    prediction["status"] = "SETTLED"
    prediction["result"] = result
    prediction["reason"] = reason
    prediction["category"] = category
    prediction["settle_time"] = str(settle_time)
    
    if prediction["trade_decision"] == "PAPER_TRADE_CANDIDATE":
        prediction["pnl_usd"] = PAYOUT_RATE * 5.0 if result == "WIN" else -5.0
    else:
        prediction["pnl_usd"] = 0.0

    return prediction


# -------------------- ③ モデル管理 --------------------

def load_or_init_model(symbol_key):
    safe_name = symbol_key.replace(" ", "_").replace("(", "").replace(")", "")
    m_file = DATA_DIR / f"pro_model_v3_{safe_name}.joblib"
    s_file = DATA_DIR / f"pro_scaler_v3_{safe_name}.joblib"

    if m_file.exists() and s_file.exists():
        try:
            model = joblib.load(m_file)
            scaler = joblib.load(s_file)
            return model, scaler, True
        except Exception:
            pass
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
    scaler = StandardScaler()
    return model, scaler, False


def save_model_state(model, scaler, symbol_key):
    safe_name = symbol_key.replace(" ", "_").replace("(", "").replace(")", "")
    m_file = DATA_DIR / f"pro_model_v3_{safe_name}.joblib"
    s_file = DATA_DIR / f"pro_scaler_v3_{safe_name}.joblib"
    joblib.dump(model, m_file)
    joblib.dump(scaler, s_file)


# -------------------- ④ ウォークフォワード検証エンジン --------------------

def run_pro_walkforward(target_name, all_dfs, target_pct, stake_usd, hold_bars, fee_pct):
    df_target = all_dfs.get(target_name)
    if df_target is None or len(df_target) < 300:
        return pd.DataFrame()

    features_df = build_full_integrated_features(target_name, all_dfs)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
    scaler = StandardScaler()

    init_end = int(len(df_target) * 0.25)
    X_init, y_init = [], []
    
    for j in range(30, init_end):
        feat_row = features_df.iloc[j]
        if j + 1 < len(df_target) and not feat_row.isna().any():
            curr_c = float(df_target["Close"].iloc[j])
            next_c = float(df_target["Close"].iloc[j + 1])
            label = 1 if next_c > curr_c else 0
            X_init.append(feat_row.values)
            y_init.append(label)

    if len(X_init) > 20:
        X_init = np.array(X_init)
        y_init = np.array(y_init)
        scaler.fit(X_init)
        X_init_scaled = scaler.transform(X_init)
        model.partial_fit(X_init_scaled, y_init, classes=np.array([0, 1]))

    records = []
    start_i = init_end
    progress_bar = st.progress(0)
    total_steps = (len(df_target) - hold_bars - 1) - start_i

    for idx, i in enumerate(range(start_i, len(df_target) - hold_bars - 1)):
        if total_steps > 0 and idx % 50 == 0:
            progress_bar.progress(min(1.0, idx / total_steps))

        feat_row = features_df.iloc[i]
        if feat_row.isna().any():
            continue

        X_pred = scaler.transform(np.array([feat_row.values]))
        
        try:
            if hasattr(model, "classes_") and len(model.classes_) == 2:
                probs = model.predict_proba(X_pred)[0]
                classes_list = list(model.classes_)
                p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
                p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
                model_status = "学習済み・正常"
            else:
                p_low, p_high = 0.5, 0.5
                model_status = "未学習・初期状態"
        except Exception:
            p_low, p_high = 0.5, 0.5
            model_status = "予測エラー"

        signal = predict_direction(p_high, p_low)
        trade_decision = assess_trade(signal)

        entry_index = i + 1
        entry_price = float(df_target["Open"].iloc[entry_index])
        future_window = df_target.iloc[entry_index : entry_index + hold_bars]

        result, reason, category, settle_time = evaluate_outcome(
            entry_price, signal["direction"], future_window, target_pct, fee_pct
        )
        
        status = "SETTLED" if result != "PENDING" else "PENDING"
        pnl = 0.0
        if status == "SETTLED" and trade_decision == "PAPER_TRADE_CANDIDATE":
            pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

        actual_next_c = float(df_target["Close"].iloc[entry_index])
        actual_curr_c = float(df_target["Close"].iloc[entry_index - 1])
        direction_label = 1 if actual_next_c > actual_curr_c else 0
        
        try:
            model.partial_fit(X_pred, np.array([direction_label]), classes=np.array([0, 1]))
        except ValueError:
            model.partial_fit(X_pred, np.array([direction_label]))

        records.append({
            "time": str(df_target.index[i]),
            "symbol": target_name,
            "direction": signal["direction"],
            "p_high": signal["p_high"],
            "p_low": signal["p_low"],
            "confidence": signal["confidence"],
            "probability_gap": signal["probability_gap"],
            "model_status": model_status,
            "trade_decision": trade_decision,
            "status": status,
            "result": result,
            "reason": reason,
            "category": category,
            "settle_time": str(settle_time),
            "entry_price": entry_price,
            "pnl_usd": pnl,
        })

    progress_bar.empty()
    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")

    return res_df


# -------------------- ⑤ ダッシュボード描画 --------------------

def show_dashboard(res_df):
    if res_df.empty:
        st.warning("検証データがありません。")
        return

    settled_df = res_df[res_df["status"] == "SETTLED"].copy()
    executed_df = settled_df[settled_df["trade_decision"] == "PAPER_TRADE_CANDIDATE"].copy()
    track_only_df = settled_df[settled_df["trade_decision"] == "TRACK_ONLY"].copy()
    pending_df = res_df[res_df["status"] == "PENDING"].copy()

    total_exec = len(executed_df)
    total_track = len(track_only_df)
    total_pending = len(pending_df)
    
    wins = int((executed_df["result"] == "WIN").sum())
    losses = int((executed_df["result"] == "LOSS").sum())
    win_rate = wins / total_exec * 100 if total_exec > 0 else 0.0
    total_pnl = float(executed_df["pnl_usd"].sum())
    be_rate = (1 / (1 + PAYOUT_RATE)) * 100

    gross_profit = float(executed_df[executed_df["pnl_usd"] > 0]["pnl_usd"].sum())
    gross_loss = abs(float(executed_df[executed_df["pnl_usd"] < 0]["pnl_usd"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    st.markdown(
        f"""
        <div class="genius-banner">
            <h2>🌌 UTAYA プロ検証レポート (全件追跡・動的判定)</h2>
            <p>確定済み実取引: <strong>{total_exec}回</strong> (勝率: <strong>{win_rate:.2f}%</strong> / 損益分岐: {be_rate:.2f}%) ｜ 見送り追跡: <strong>{total_track}回</strong> ｜ 判定待ち: <strong>{total_pending}件</strong></p>
            <p>累積損益: <strong>${total_pnl:+.2f}</strong> ｜ プロフィットファクター (PF): <strong>{profit_factor:.2f}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if win_rate >= be_rate and total_exec > 10:
        st.success(f"🌟 実取引勝率 ({win_rate:.2f}%) は損益分岐点 ({be_rate:.2f}%) を突破しています。")
    else:
        st.info(f"📊 実取引勝率: {win_rate:.2f}% (目標: {be_rate:.2f}%以上)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧭 方向予測の分布")
        dir_counts = res_df["direction"].value_counts().reindex(["HIGH", "LOW"], fill_value=0)
        st.bar_chart(dir_counts)
    with col2:
        st.subheader("📊 要因分類 (確定済み実取引)")
        if not executed_df.empty and "category" in executed_df.columns:
            cat_counts = executed_df["category"].value_counts()
            st.bar_chart(cat_counts)

    st.subheader("📈 累積損益の推移（成長曲線）")
    chart_df = executed_df[["time", "pnl_usd"]].copy()
    if not chart_df.empty:
        chart_df["time"] = pd.to_datetime(chart_df["time"])
        chart_df["累積損益"] = chart_df["pnl_usd"].cumsum()
        chart_df = chart_df.set_index("time")
        st.line_chart(chart_df[["累積損益"]])
    else:
        st.info("確定した取引データがありません。")

    st.subheader("📝 詳細監査ログ (直近50件)")
    st.dataframe(
        res_df.sort_values("time", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )


# -------------------- ⑥ サイドバー設定 --------------------

st.sidebar.header("🌌 プロシステム設定")
selected_symbol = st.sidebar.selectbox("🎯 分析対象銘柄・市場", options=list(SYMBOLS.keys()), index=0)

timeframe = st.sidebar.selectbox("⏱️ 分析時間軸", options=["1分足 (短期)", "1時間足 (中期)"], index=1)
interval_str = "1m" if "1分足" in timeframe else "1h"
period_str = "7d" if "1分足" in timeframe else "180d"

target_pct = st.sidebar.slider("🎯 目標変動率 (±%)", 0.002, 0.05, 0.01, 0.001, format="%.3f")
fee_pct = st.sidebar.slider("💸 手数料・スプレッド考慮 (%)", 0.0, 0.005, 0.0005, 0.0001, format="%.4f")
stake_usd = st.sidebar.number_input("💵 仮想賭け金 ($)", 1.0, 10000.0, 5.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (足の本数)", 1, 20, 5)

be_rate_disp = (1 / (1 + PAYOUT_RATE)) * 100
st.sidebar.info(f"📊 損益分岐勝率: **{be_rate_disp:.2f}%** (ペ88%)")

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 データ・学習状態の管理")
if st.sidebar.button("🔄 AI履歴・学習状態をリセット", use_container_width=True):
    st.session_state.v3_monitoring = False
    files_to_delete = [VIRTUAL_TRADES_FILE, ANALYSIS_LOG_FILE]
    for file_path in files_to_delete:
        try:
            if file_path.exists(): file_path.unlink()
        except OSError:
            pass
    st.sidebar.success("リセット完了しました。再読み込みしてください。")
    st.rerun()


# -------------------- ⑦ 連続監視・動的ライフサイクルループ --------------------

if "v3_monitoring" not in st.session_state:
    st.session_state.v3_monitoring = False

st.subheader(f"♾️ 動的ライフサイクル監視 [{selected_symbol}]")
col1, col2 = st.columns(2)
with col1:
    if st.button("▶ 連続監視・学習を開始", use_container_width=True):
        st.session_state.v3_monitoring = True
with col2:
    if st.button("⏹ 連続監視を停止", use_container_width=True):
        st.session_state.v3_monitoring = False

if st.session_state.v3_monitoring:
    st.success(f"連続監視中 [{selected_symbol}]：PENDING予測の自動再評価とオンライン学習を稼働中")
else:
    st.info("停止中です。開始ボタンで連続監視・分析を開始できます。")

@st.fragment(run_every="30s")
def continuous_v3_loop():
    if not st.session_state.v3_monitoring:
        st.caption("監視は停止しています。")
        return

    with st.spinner("全主要市場のデータを同期中..."):
        all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)

    df_target = all_dfs.get(selected_symbol)
    if df_target is None or len(df_target) < 50:
        st.warning("対象データの件数が不足しています。")
        return

    # 1. 既存の PENDING 記録を読み込み、最新データで再評価して更新する
    if ANALYSIS_LOG_FILE.exists():
        try:
            log_df = pd.read_csv(ANALYSIS_LOG_FILE)
            updated_rows = []
            for _, row in log_df.iterrows():
                row_dict = row.to_dict()
                if row_dict["status"] == "PENDING" and row_dict["symbol"] == selected_symbol:
                    row_dict = update_prediction_status(row_dict, df_target, hold_bars, target_pct, fee_pct)
                updated_rows.append(row_dict)
            log_df = pd.DataFrame(updated_rows)
            log_df.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")
        except Exception:
            pass

    # 2. 新規予測の生成と保存
    features_table = build_full_integrated_features(selected_symbol, all_dfs)
    signal_i = len(df_target) - 2
    feat_row = features_table.iloc[signal_i]

    if feat_row.isna().any():
        st.warning("特徴量計算エラー（同期待ち）")
        return

    model, scaler, is_trained = load_or_init_model(selected_symbol)

    try:
        feat_sc = scaler.transform(np.array([feat_row.values]))
    except Exception:
        X_live = features_table.dropna().values
        scaler.fit(X_live)
        feat_sc = scaler.transform(np.array([feat_row.values]))

    try:
        if is_trained and hasattr(model, "classes_") and len(model.classes_) == 2:
            probs = model.predict_proba(feat_sc)[0]
            classes_list = list(model.classes_)
            p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
            p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
            model_status = "学習済み・正常"
        else:
            p_low, p_high = 0.5, 0.5
            model_status = "⚠️ 未学習・初期状態"
    except Exception:
        p_low, p_high = 0.5, 0.5
        model_status = "❌ 予測エラー"

    signal = predict_direction(p_high, p_low)
    trade_decision = assess_trade(signal)

    signal_price = float(df_target["Close"].iloc[signal_i])
    signal_time = df_target.index[signal_i]
    time_str = str(signal_time)

    # 重複登録を防ぎつつ新規作成
    if ANALYSIS_LOG_FILE.exists():
        existing_log = pd.read_csv(ANALYSIS_LOG_FILE)
    else:
        existing_log = pd.DataFrame(columns=["time", "symbol", "direction", "p_high", "p_low", "confidence", "probability_gap", "model_status", "trade_decision", "status", "result", "reason", "category", "settle_time", "entry_price", "pnl_usd"])

    if existing_log.empty or not ((existing_log["time"] == time_str) & (existing_log["symbol"] == selected_symbol)).any():
        future_window = df_target.loc[signal_time:].iloc[1 : hold_bars + 1]
        result, reason, category, settle_time = evaluate_outcome(
            signal_price, signal["direction"], future_window, target_pct, fee_pct
        )
        status = "SETTLED" if result != "PENDING" else "PENDING"
        pnl = 0.0
        if status == "SETTLED" and trade_decision == "PAPER_TRADE_CANDIDATE":
            pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

        new_row = pd.DataFrame([{
            "time": time_str,
            "symbol": selected_symbol,
            "direction": signal["direction"],
            "p_high": signal["p_high"],
            "p_low": signal["p_low"],
            "confidence": signal["confidence"],
            "probability_gap": signal["probability_gap"],
            "model_status": model_status,
            "trade_decision": trade_decision,
            "status": status,
            "result": result,
            "reason": reason,
            "category": category,
            "settle_time": str(settle_time),
            "entry_price": signal_price,
            "pnl_usd": pnl,
        }])
        existing_log = pd.concat([existing_log, new_row])
        existing_log.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")

        # オンライン学習更新
        actual_next_c = float(df_target["Close"].iloc[signal_i + 1])
        actual_curr_c = float(df_target["Close"].iloc[signal_i])
        direction_label = 1 if actual_next_c > actual_curr_c else 0
        try:
            model.partial_fit(feat_sc, np.array([direction_label]), classes=np.array([0, 1]))
            save_model_state(model, scaler, selected_symbol)
        except ValueError:
            model.partial_fit(feat_sc, np.array([direction_label]))
            save_model_state(model, scaler, selected_symbol)

    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 基準価格: ${signal_price:,.2f} | 状態: {model_status}")

    card_class = "scenario-card-high" if signal["direction"] == "HIGH" else "scenario-card-low"
    st.markdown(
        f"""
        <div class="{card_class}">
            <h2>🧭 方向予測: {signal['direction']} [{selected_symbol}]</h2>
            <p>推定確率: HIGH <strong>{signal['p_high']:.1%}</strong> ｜ LOW <strong>{signal['p_low']:.1%}</strong> (ギャップ: {signal['probability_gap']:.2f})</p>
            <p>予測状態: <strong>{model_status}</strong> ｜ 取引判断: <strong>{trade_decision}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not existing_log.empty:
        st.subheader("📋 動的ライフサイクル・全件追跡ログ")
        st.dataframe(existing_log.sort_values("time", ascending=False).head(20), use_container_width=True, hide_index=True)
        
        settled_logs = existing_log[(existing_log["status"] == "SETTLED") & (existing_log["trade_decision"] == "PAPER_TRADE_CANDIDATE") & (existing_log["result"].isin(["WIN", "LOSS"]))]
        s_wins = int((settled_logs["result"] == "WIN").sum())
        s_losses = int((settled_logs["result"] == "LOSS").sum())
        s_rate = s_wins / len(settled_logs) * 100 if len(settled_logs) > 0 else 0.0
        s_pnl = float(settled_logs["pnl_usd"].sum())
        st.caption(f"確定済み実取引実績: {s_wins}勝 {s_losses}敗 (勝率: {s_rate:.1f}%) | 累積損益: ${s_pnl:+.2f} | 記録総数: {len(existing_log)}件")

continuous_v3_loop()


# -------------------- ⑧ バックテスト実行ボタン --------------------

st.markdown("---")
st.subheader("🚀 プロ検証ウォークフォワードの実行")

if st.button("▶️ 公平な検証トレーニングを開始", use_container_width=True):
    with st.spinner("全市場データを一括取得し、動的ライフサイクル検証を実行中..."):
        all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
        df_target = all_dfs.get(selected_symbol)

        if df_target is None or len(df_target) < 300:
            st.error("対象データの件数が不足しています（最低300本必要です）。期間を長くしてください。")
        else:
            res_df = run_pro_walkforward(
                selected_symbol,
                all_dfs,
                target_pct=target_pct,
                stake_usd=stake_usd,
                hold_bars=hold_bars,
                fee_pct=fee_pct,
            )
            show_dashboard(res_df)
elif ANALYSIS_LOG_FILE.exists():
    try:
        saved_df = pd.read_csv(ANALYSIS_LOG_FILE)
        if not saved_df.empty:
            show_dashboard(saved_df)
    except Exception:
        pass
