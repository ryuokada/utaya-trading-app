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
# UTAYA グローバル・インベストメンツ (v8 大口危機察知・プロ防御版)
# 思想: 大口衝突・価格摩擦の検知、緊急警報（EMERGENCY WARNING）、リスク強制遮断
# =========================================================

st.set_page_config(
    page_title="UTAYA Global Investments - v8 Emergency Risk Engine",
    page_icon="🚨",
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
TARGET_ROI = 0.20   # 目標ROI 20% (10$ -> 12$)
DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(exist_ok=True)

GLOBAL_HISTORY_DIR = DATA_DIR / "full_market_history_v8"
GLOBAL_HISTORY_DIR.mkdir(exist_ok=True)
ANALYSIS_LOG_FILE = DATA_DIR / "full_market_analysis_log_v8.csv"

# -------------------- CSS --------------------
st.markdown(
    """
    <style>
    .genius-banner {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
        padding: 20px;
        border-radius: 15px;
        color: #000;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .emergency-alert-card {
        background: rgba(255, 0, 0, 0.2);
        border-left: 6px solid #ff0000;
        padding: 20px;
        border-radius: 8px;
        color: #fff;
        margin-bottom: 20px;
        animation: pulse 1.5s infinite;
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

st.title("🚨 UTAYA v8 大口危機察知・緊急防御エンジン")
st.caption(
    "大口衝突・価格摩擦の自動検知 | 危険シグナル強制シャットダウン | 10$➔12$ 達成管理"
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
                    df_new["Volume"] = 1000.0
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


# -------------------- ② 危機察知・価格摩擦 特徴量エンジン (v8) --------------------

def build_v8_emergency_features(target_name, all_dfs):
    """
    大口の衝突（出来高急増かつ実体が小さい＝価格摩擦）や、急激なボラティリティ跳ね上がりを検出
    """
    if target_name not in all_dfs:
        return pd.DataFrame()

    df_target = all_dfs[target_name].copy()
    df_target.index = pd.to_datetime(df_target.index)
    if df_target.index.tz is not None:
        df_target.index = df_target.index.tz_localize(None)

    open_p = df_target["Open"].astype(float)
    high = df_target["High"].astype(float)
    low = df_target["Low"].astype(float)
    close = df_target["Close"].astype(float)
    volume = df_target["Volume"].astype(float)

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
    volume_z = (volume - vol_mean) / vol_std

    body_size = (close - open_p).abs()
    candle_range = (high - low).replace(0, 0.0001)
    body_ratio = body_size / candle_range

    features = pd.DataFrame(index=df_target.index)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["sma_ratio_10"] = (close - close.rolling(10).mean()) / close.rolling(10).mean()
    features["rsi"] = rsi
    features["volume_z"] = volume_z.fillna(0.0)
    features["body_ratio"] = body_ratio
    features["volatility"] = returns.rolling(14).std()

    # --- 🚨 危機察知フラグ（大口の衝突・価格摩擦インジケータ） ---
    # 出来高が平均から大きく乖離（Z > 2.0）しているのに、ローソク足の実体が極端に小さい（body_ratio < 0.2）状態
    # ＝「大口同士が激しくぶつかり合い、どちらに抜けるか分からない極限の危険ゾーン」
    friction_signal = ((volume_z > 2.0) & (body_ratio < 0.25)).astype(float)
    features["emergency_friction"] = friction_signal

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

    return features.replace([np.inf, -np.inf], np.nan)


def predict_direction(p_high, p_low):
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


def assess_trade(signal, is_emergency=False, min_gap=0.035):
    """
    🚨 緊急警報（is_emergency=True）が点灯している場合は、
    確率ギャップに関わらず「強制見送り（TRACK_ONLY / EMERGENCY_HOLD）」にする。
    """
    if is_emergency:
        return "EMERGENCY_HOLD (大口衝突・危険回避の為強制見送り)"
    if signal["probability_gap"] >= min_gap:
        return "PAPER_TRADE_CANDIDATE"
    return "TRACK_ONLY"


def evaluate_outcome(entry_price, direction, future_candles, target_pct=0.01, fee_pct=0.0005, hold_bars=5):
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
            return "LOSS", "同一足で上下両方到達（大口の乱高下・ノイズ）", "VOLATILITY_NOISE", str(idx)
        if hit_up:
            if direction == "HIGH":
                return "WIN", "上目標先着達成", "TREND_SUCCESS", str(idx)
            else:
                return "LOSS", "上方向逆行失敗", "REVERSE_FAILURE_HIGH", str(idx)
        if hit_down:
            if direction == "LOW":
                return "WIN", "下目標先着達成", "TREND_SUCCESS", str(idx)
            else:
                return "LOSS", "下方向逆行失敗", "REVERSE_FAILURE_LOW", str(idx)

    if len(future_candles) >= hold_bars:
        return "TIMEOUT", "判定期間内未到達（膠着・タイムアウト）", "TIMEOUT_RANGE", str(future_candles.index[-1])

    return "PENDING", "判定期間内未到達・データ待機中", "PENDING", None


def update_prediction_status(prediction, df_target, hold_bars, target_pct, fee_pct, stake_usd):
    if prediction["status"] != "PENDING":
        return prediction

    signal_time = pd.to_datetime(prediction["time"])
    future_candles = df_target.loc[signal_time:].iloc[1 : hold_bars + 1]

    result, reason, category, settle_time = evaluate_outcome(
        float(prediction["entry_price"]),
        prediction["direction"],
        future_candles,
        target_pct,
        fee_pct,
        hold_bars,
    )

    if result == "PENDING":
        return prediction

    prediction["status"] = "SETTLED"
    prediction["result"] = result
    prediction["reason"] = reason
    prediction["category"] = category
    prediction["settle_time"] = str(settle_time)
    
    # 緊急見送りの場合は損益発生ナシ（0ドル）
    if "EMERGENCY_HOLD" in str(prediction["trade_decision"]):
        prediction["pnl_usd"] = 0.0
    elif prediction["trade_decision"] == "PAPER_TRADE_CANDIDATE":
        prediction["pnl_usd"] = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd
    else:
        prediction["pnl_usd"] = 0.0

    return prediction


# -------------------- ③ 検証・ウォークフォワードエンジン --------------------

def run_v8_walkforward(target_name, all_dfs, target_pct, stake_usd, hold_bars, fee_pct):
    df_target = all_dfs.get(target_name)
    if df_target is None or len(df_target) < 300:
        return pd.DataFrame()

    features_df = build_v8_emergency_features(target_name, all_dfs)
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
        scaler.fit(np.array(X_init))
        model.partial_fit(scaler.transform(np.array(X_init)), np.array(y_init), classes=np.array([0, 1]))

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

        is_emergency = bool(feat_row.get("emergency_friction", 0) > 0)
        X_pred = scaler.transform(np.array([feat_row.values]))
        
        try:
            if hasattr(model, "classes_") and len(model.classes_) == 2:
                probs = model.predict_proba(X_pred)[0]
                classes_list = list(model.classes_)
                p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
                p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
                model_status = "v8緊急防御・稼働中"
            else:
                p_low, p_high = 0.5, 0.5
                model_status = "初期化中"
        except Exception:
            p_low, p_high = 0.5, 0.5
            model_status = "エラー"

        signal = predict_direction(p_high, p_low)
        trade_decision = assess_trade(signal, is_emergency=is_emergency)

        entry_index = i + 1
        entry_price = float(df_target["Open"].iloc[entry_index])
        future_window = df_target.iloc[entry_index : entry_index + hold_bars]

        result, reason, category, settle_time = evaluate_outcome(
            entry_price, signal["direction"], future_window, target_pct, fee_pct, hold_bars
        )
        
        status = "SETTLED" if result != "PENDING" else "PENDING"
        pnl = 0.0
        if status == "SETTLED":
            if "EMERGENCY_HOLD" in str(trade_decision):
                pnl = 0.0
            elif trade_decision == "PAPER_TRADE_CANDIDATE":
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


# -------------------- ④ 全市場スキャナー (v8) --------------------

def run_v8_multi_market_scanner(all_dfs, target_pct, hold_bars, fee_pct):
    scanner_results = []
    for name, df in all_dfs.items():
        if df is None or len(df) < 50:
            continue
        try:
            features_table = build_v8_emergency_features(name, all_dfs)
            signal_i = len(df) - 2
            feat_row = features_table.iloc[signal_i]
            if feat_row.isna().any():
                continue

            is_emergency = bool(feat_row.get("emergency_friction", 0) > 0)
            X_live = features_table.dropna().values
            scaler = StandardScaler().fit(X_live)
            model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
            
            y_live = [1 if df["Close"].iloc[k+1] > df["Close"].iloc[k] else 0 for k in range(len(features_table.dropna()))]
            if len(X_live) == len(y_live) and len(X_live) > 10:
                model.partial_fit(scaler.transform(X_live), np.array(y_live), classes=np.array([0, 1]))

            feat_sc = scaler.transform(np.array([feat_row.values]))
            probs = model.predict_proba(feat_sc)[0]
            classes_list = list(model.classes_)
            p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
            p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]

            signal = predict_direction(p_high, p_low)
            trade_decision = assess_trade(signal, is_emergency=is_emergency)
            volatility = float(features_table["volatility"].iloc[signal_i])

            status_text = "🚨 【緊急警報】大口衝突・危険" if is_emergency else ("🔥 勝負候補" if trade_decision == "PAPER_TRADE_CANDIDATE" else "👀 様子見")

            scanner_results.append({
                "銘柄": name,
                "方向予測": signal["direction"],
                "HIGH確率": f"{signal['p_high']:.1%}",
                "LOW確率": f"{signal['p_low']:.1%}",
                "確率ギャップ": round(signal["probability_gap"], 3),
                "危機状態": status_text,
                "基準価格": round(float(df["Close"].iloc[signal_i]), 2)
            })
        except Exception:
            pass
    return pd.DataFrame(scanner_results)


# -------------------- ⑤ ダッシュボード描画 --------------------

def show_dashboard(res_df, stake_usd):
    if res_df.empty:
        st.warning("検証データがありません。")
        return

    settled_df = res_df[res_df["status"] == "SETTLED"].copy()
    executed_df = settled_df[settled_df["trade_decision"] == "PAPER_TRADE_CANDIDATE"].copy()
    emergency_held_df = settled_df[settled_df["trade_decision"].str.contains("EMERGENCY_HOLD", na=False)].copy()
    track_only_df = settled_df[settled_df["trade_decision"] == "TRACK_ONLY"].copy()
    pending_df = res_df[res_df["status"] == "PENDING"].copy()

    total_exec = len(executed_df)
    total_emergency = len(emergency_held_df)
    total_track = len(track_only_df)
    total_pending = len(pending_df)
    
    wins = int((executed_df["result"] == "WIN").sum())
    losses = int(executed_df["result"].isin(["LOSS", "TIMEOUT"]).sum())
    win_rate = wins / total_exec * 100 if total_exec > 0 else 0.0
    total_pnl = float(executed_df["pnl_usd"].sum())
    be_rate = (1 / (1 + PAYOUT_RATE)) * 100

    current_capital = stake_usd + total_pnl
    actual_roi = (total_pnl / stake_usd) * 100 if stake_usd > 0 else 0.0

    gross_profit = float(executed_df[executed_df["pnl_usd"] > 0]["pnl_usd"].sum())
    gross_loss = abs(float(executed_df[executed_df["pnl_usd"] < 0]["pnl_usd"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    st.markdown(
        f"""
        <div class="genius-banner">
            <h2>💰 資金達成状況: 初期 ${stake_usd:.2f} ➔ 現在 ${current_capital:.2f} (ROI: {actual_roi:+.1f}%)</h2>
            <p>確定勝負: <strong>{total_exec}回</strong> (勝率: <strong>{win_rate:.2f}%</strong>) ｜ 🚨 **緊急回避（大口衝突防衛）: {total_emergency}回** ｜ PF: <strong>{profit_factor:.2f}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if actual_roi >= (TARGET_ROI * 100):
        st.success(f"🏆 目標達成！ROI {actual_roi:.1f}%（10$ $\rightarrow$ 12$ 以上の利益創出に成功！）")
    else:
        st.info(f"📊 目標ROI 20%（${stake_usd * 1.2:.2f}）に向けて検証中... (現在: ${current_capital:.2f})")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧭 方向予測の分布")
        dir_counts = res_df["direction"].value_counts().reindex(["HIGH", "LOW"], fill_value=0)
        st.bar_chart(dir_counts)
    with col2:
        st.subheader("📊 要因分類 (実戦勝負)")
        if not executed_df.empty and "category" in executed_df.columns:
            cat_counts = executed_df["category"].value_counts()
            st.bar_chart(cat_counts)

    st.subheader("📈 資金成長曲線（累積損益）")
    chart_df = executed_df[["time", "pnl_usd"]].copy()
    if not chart_df.empty:
        chart_df["time"] = pd.to_datetime(chart_df["time"])
        chart_df["資産推移"] = stake_usd + chart_df["pnl_usd"].cumsum()
        chart_df = chart_df.set_index("time")
        st.line_chart(chart_df[["資産推移"]])
    else:
        st.info("確定した取引データがありません。")

    st.subheader("📝 詳細プロ監査ログ (直近50件)")
    st.dataframe(
        res_df.sort_values("time", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )


# -------------------- ⑥ サイドバー設定 --------------------

st.sidebar.header("🚨 v8 大口危機防御設定")
selected_symbol = st.sidebar.selectbox("🎯 分析対象銘柄・市場 (個別)", options=list(SYMBOLS.keys()), index=0)

timeframe = st.sidebar.selectbox("⏱️ 分析時間軸", options=["1分足 (短期)", "1時間足 (中期)"], index=1)
interval_str = "1m" if "1分足" in timeframe else "1h"
period_str = "7d" if "1分足" in timeframe else "180d"

target_pct = st.sidebar.slider("🎯 目標変動率 (±%)", 0.002, 0.05, 0.01, 0.001, format="%.3f")
fee_pct = st.sidebar.slider("💸 手数料・スプレッド (%)", 0.0, 0.005, 0.0005, 0.0001, format="%.4f")
stake_usd = st.sidebar.number_input("💵 初期投資資金 ($)", 1.0, 10000.0, 10.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (足の本数)", 1, 20, 5)

be_rate_disp = (1 / (1 + PAYOUT_RATE)) * 100
st.sidebar.info(f"📊 損益分岐勝率: **{be_rate_disp:.2f}%** (ペ88%)")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 AI学習データ・履歴をリセット", use_container_width=True):
    st.session_state.v8_monitoring = False
    if ANALYSIS_LOG_FILE.exists():
        try:
            ANALYSIS_LOG_FILE.unlink()
        except OSError:
            pass
    st.sidebar.success("リセット完了しました。")
    st.rerun()

tab1, tab2 = st.tabs(["📊 個別銘柄・v8緊急防御検証", "🌌 全市場危機スキャナー"])


# -------------------- ⑦ タブ1: 個別実戦検証 --------------------

with tab1:
    if "v8_monitoring" not in st.session_state:
        st.session_state.v8_monitoring = False

    st.subheader(f"♾️ v8 危機防御リアルタイム監視 [{selected_symbol}]")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ v8緊急防御監視を開始", use_container_width=True):
            st.session_state.v8_monitoring = True
    with c2:
        if st.button("⏹ 監視を停止", use_container_width=True):
            st.session_state.v8_monitoring = False

    if st.session_state.v8_monitoring:
        st.success(f"v8監視中 [{selected_symbol}]：大口衝突・価格摩擦の緊急警報が稼働中")
    else:
        st.info("停止中です。開始ボタンを押してください。")

    @st.fragment(run_every="30s")
    def continuous_v8_loop():
        if not st.session_state.v8_monitoring:
            st.caption("監視は停止しています。")
            return

        with st.spinner("市場データを同期中..."):
            all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)

        df_target = all_dfs.get(selected_symbol)
        if df_target is None or len(df_target) < 50:
            st.warning("データ不足です。")
            return

        if ANALYSIS_LOG_FILE.exists():
            try:
                log_df = pd.read_csv(ANALYSIS_LOG_FILE)
                updated_rows = []
                for _, row in log_df.iterrows():
                    row_dict = row.to_dict()
                    if row_dict["status"] == "PENDING" and row_dict["symbol"] == selected_symbol:
                        row_dict = update_prediction_status(row_dict, df_target, hold_bars, target_pct, fee_pct, stake_usd)
                    updated_rows.append(row_dict)
                log_df = pd.DataFrame(updated_rows)
                log_df.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")
            except Exception:
                pass

        features_table = build_v8_emergency_features(selected_symbol, all_dfs)
        signal_i = len(df_target) - 2
        feat_row = features_table.iloc[signal_i]
        if feat_row.isna().any():
            return

        is_emergency = bool(feat_row.get("emergency_friction", 0) > 0)
        X_live = features_table.dropna().values
        scaler = StandardScaler().fit(X_live)
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
        y_live = [1 if df_target["Close"].iloc[k+1] > df_target["Close"].iloc[k] else 0 for k in range(len(features_table.dropna()))]
        if len(X_live) == len(y_live) and len(X_live) > 10:
            model.partial_fit(scaler.transform(X_live), np.array(y_live), classes=np.array([0, 1]))

        feat_sc = scaler.transform(np.array([feat_row.values]))
        probs = model.predict_proba(feat_sc)[0]
        classes_list = list(model.classes_)
        p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
        p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]

        signal = predict_direction(p_high, p_low)
        trade_decision = assess_trade(signal, is_emergency=is_emergency)
        signal_price = float(df_target["Close"].iloc[signal_i])
        signal_time = df_target.index[signal_i]
        time_str = str(signal_time)

        # 🚨 緊急警報の発動表示
        if is_emergency:
            st.markdown(
                """
                <div class="emergency-alert-card">
                    <h2>🚨 【緊急警報】大口衝突・価格摩擦の異常検知！</h2>
                    <p>現在、出来高が爆発しているにもかかわらず価格が膠着（実体が極小）しています。<strong>大口同士の激しい玉のぶつかり合い（あるいは仕掛け前の罠）が発生中</strong>です。すべての新規エントリーを強制シャットダウンします！</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if ANALYSIS_LOG_FILE.exists():
            existing_log = pd.read_csv(ANALYSIS_LOG_FILE)
        else:
            existing_log = pd.DataFrame(columns=["time", "symbol", "direction", "p_high", "p_low", "confidence", "probability_gap", "model_status", "trade_decision", "status", "result", "reason", "category", "settle_time", "entry_price", "pnl_usd"])

        if existing_log.empty or not ((existing_log["time"] == time_str) & (existing_log["symbol"] == selected_symbol)).any():
            future_window = df_target.loc[signal_time:].iloc[1 : hold_bars + 1]
            result, reason, category, settle_time = evaluate_outcome(
                signal_price, signal["direction"], future_window, target_pct, fee_pct, hold_bars
            )
            status = "SETTLED" if result != "PENDING" else "PENDING"
            pnl = 0.0
            if status == "SETTLED":
                if "EMERGENCY_HOLD" in str(trade_decision):
                    pnl = 0.0
                elif trade_decision == "PAPER_TRADE_CANDIDATE":
                    pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

            new_row = pd.DataFrame([{
                "time": time_str,
                "symbol": selected_symbol,
                "direction": signal["direction"],
                "p_high": signal["p_high"],
                "p_low": signal["p_low"],
                "confidence": signal["confidence"],
                "probability_gap": signal["probability_gap"],
                "model_status": "v8緊急防御稼働中",
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

        card_class = "scenario-card-high" if signal["direction"] == "HIGH" else "scenario-card-low"
        st.markdown(
            f"""
            <div class="{card_class}">
                <h2>🧭 v8 方向予測: {signal['direction']} [{selected_symbol}]</h2>
                <p>HIGH: <strong>{signal['p_high']:.1%}</strong> ｜ LOW: <strong>{signal['p_low']:.1%}</strong> (ギャップ: {signal['probability_gap']:.2f})</p>
                <p>プロ防衛判断: <strong>{trade_decision}</strong> (基準価格: ${signal_price:,.2f})</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not existing_log.empty:
            st.subheader("📋 リアルタイム全件追跡ログ（防衛記録含む）")
            st.dataframe(existing_log.sort_values("time", ascending=False).head(15), use_container_width=True, hide_index=True)

    continuous_v8_loop()

    st.markdown("---")
    st.subheader("🚀 公平なウォークフォワード検証（大口防御バックテスト）")
    if st.button("▶️ 会社資金10$倍増 v8緊急防御検証開始", use_container_width=True):
        with st.spinner("大口衝突フィルターを適用したウォークフォワード検証を実行中..."):
            all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
            df_target = all_dfs.get(selected_symbol)
            if df_target is None or len(df_target) < 300:
                st.error("データ件数が不足しています（最低300本必要です）。")
            else:
                res_df = run_v8_walkforward(
                    selected_symbol, all_dfs, target_pct=target_pct, stake_usd=stake_usd, hold_bars=hold_bars, fee_pct=fee_pct
                )
                show_dashboard(res_df, stake_usd)
    elif ANALYSIS_LOG_FILE.exists():
        try:
            saved_df = pd.read_csv(ANALYSIS_LOG_FILE)
            if not saved_df.empty:
                show_dashboard(saved_df, stake_usd)
        except Exception:
            pass


# -------------------- ⑧ タブ2: 全市場危機スキャナー --------------------

with tab2:
    st.subheader("🌌 v8 全市場危機察知スキャナー")
    st.caption("全9市場を同時に監視し、現在「大口同士の衝突（価格摩擦）」が発生している危険な銘柄を赤色の緊急警報としてあぶり出します。")

    if st.button("🔍 全市場の危機・大口衝突を一斉スキャン", use_container_width=True):
        with st.spinner("全9市場の摩擦・危機特徴量を解析中..."):
            all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
            scanner_df = run_v8_multi_market_scanner(all_dfs, target_pct, hold_bars, fee_pct)
            if not scanner_df.empty:
                st.success("スキャン完了：市場の危機状況を一覧表示します。")
                st.dataframe(scanner_df, use_container_width=True, hide_index=True)
            else:
                st.warning("スキャン可能なデータが不足しています。")
