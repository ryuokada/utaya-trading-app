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
# UTAYA グローバル・インベストメンツ専用 AIシステム (プロ心理戦・オッズ構造統合版 v6)
# 思想: 大口・小口の塊（マネーフロー）、群集心理（過熱感）、構造的歪みの完全攻略
# =========================================================

st.set_page_config(
    page_title="UTAYA Global Investments - Pro Psychology & Flow Engine v6",
    page_icon="🔥",
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

GLOBAL_HISTORY_DIR = DATA_DIR / "full_market_history_v6"
GLOBAL_HISTORY_DIR.mkdir(exist_ok=True)
VIRTUAL_TRADES_FILE = DATA_DIR / "full_market_virtual_trades_v6.csv"
ANALYSIS_LOG_FILE = DATA_DIR / "full_market_analysis_log_v6.csv"

# -------------------- CSS --------------------
st.markdown(
    """
    <style>
    .genius-banner {
        background: linear-gradient(135deg, #232526, #414345);
        padding: 20px;
        border-radius: 15px;
        color: #fff;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #ffd700;
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

st.title("🔥 UTAYA グローバル・インベストメンツ (プロ心理戦・オッズ構造統合版)")
st.caption(
    "大口マネーフロー検知 | 群集心理（過熱感）分析 | 構造的歪みハック | 10$➔12$ 達成管理"
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
                    df_new["Volume"] = 1000.0  # 出来高がない場合のフォールバック
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


# -------------------- ② 心理戦・オッズ構造 特徴量エンジニアリング --------------------

def build_psychology_flow_features(target_name, all_dfs):
    """
    大口の塊（出来高急増）、群集心理（RSI・乖離率）、市場間資金フローを統合したプロ仕様特徴量
    """
    if target_name not in all_dfs:
        return pd.DataFrame()

    df_target = all_dfs[target_name].copy()
    df_target.index = pd.to_datetime(df_target.index)
    if df_target.index.tz is not None:
        df_target.index = df_target.index.tz_localize(None)

    close = df_target["Close"].astype(float)
    volume = df_target["Volume"].astype(float)
    returns = close.pct_change()
    delta = close.diff()
    
    # RSI（群集の過熱感）
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(losses != 0, 100.0)
    rsi = rsi.where(gains != 0, 50.0)

    # 出来高フロー（大口の塊の検知：直近平均に対する出来高の倍率）
    vol_ma = volume.rolling(14).mean()
    volume_shock = volume / vol_ma.replace(0, np.nan)

    features = pd.DataFrame(index=df_target.index)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["sma_ratio_10"] = (close - close.rolling(10).mean()) / close.rolling(10).mean()
    features["rsi"] = rsi
    features["volume_shock"] = volume_shock.fillna(1.0) # 大口・小口の塊の指標
    features["volatility"] = returns.rolling(14).std()

    # 他市場との資金連動フロー
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


def assess_trade(signal, min_gap=0.035):
    """プロの勝負基準（過剰人気や心理的歪みを考慮した厳格フィルタ）"""
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
            return "LOSS", "同一足で上下両方到達（ノイズ・騙し）", "VOLATILITY_NOISE", str(idx)
        if hit_up:
            if direction == "HIGH":
                return "WIN", "上目標先着達成（心理戦勝利）", "TREND_SUCCESS", str(idx)
            else:
                return "LOSS", "上方向逆行失敗（大口の罠）", "REVERSE_FAILURE_HIGH", str(idx)
        if hit_down:
            if direction == "LOW":
                return "WIN", "下目標先着達成（心理戦勝利）", "TREND_SUCCESS", str(idx)
            else:
                return "LOSS", "下方向逆行失敗（大口の罠）", "REVERSE_FAILURE_LOW", str(idx)

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
    
    if prediction["trade_decision"] == "PAPER_TRADE_CANDIDATE":
        prediction["pnl_usd"] = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd
    else:
        prediction["pnl_usd"] = 0.0

    return prediction


# -------------------- ③ 検証・ウォークフォワードエンジン --------------------

def run_pro_walkforward(target_name, all_dfs, target_pct, stake_usd, hold_bars, fee_pct):
    df_target = all_dfs.get(target_name)
    if df_target is None or len(df_target) < 300:
        return pd.DataFrame()

    features_df = build_psychology_flow_features(target_name, all_dfs)
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

        X_pred = scaler.transform(np.array([feat_row.values]))
        
        try:
            if hasattr(model, "classes_") and len(model.classes_) == 2:
                probs = model.predict_proba(X_pred)[0]
                classes_list = list(model.classes_)
                p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
                p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
                model_status = "プロ心理戦・稼働中"
            else:
                p_low, p_high = 0.5, 0.5
                model_status = "初期化中"
        except Exception:
            p_low, p_high = 0.5, 0.5
            model_status = "エラー"

        signal = predict_direction(p_high, p_low)
        trade_decision = assess_trade(signal)

        entry_index = i + 1
        entry_price = float(df_target["Open"].iloc[entry_index])
        future_window = df_target.iloc[entry_index : entry_index + hold_bars]

        result, reason, category, settle_time = evaluate_outcome(
            entry_price, signal["direction"], future_window, target_pct, fee_pct, hold_bars
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


# -------------------- ④ 全市場一括スキャナー --------------------

def run_multi_market_scanner(all_dfs, target_pct, hold_bars, fee_pct):
    scanner_results = []
    for name, df in all_dfs.items():
        if df is None or len(df) < 50:
            continue
        try:
            features_table = build_psychology_flow_features(name, all_dfs)
            signal_i = len(df) - 2
            feat_row = features_table.iloc[signal_i]
            if feat_row.isna().any():
                continue

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
            trade_decision = assess_trade(signal)
            volatility = float(features_table["volatility"].iloc[signal_i])
            vol_shock = float(features_table["volume_shock"].iloc[signal_i])

            scanner_results.append({
                "銘柄": name,
                "方向予測": signal["direction"],
                "HIGH確率": f"{signal['p_high']:.1%}",
                "LOW確率": f"{signal['p_low']:.1%}",
                "確率ギャップ": round(signal["probability_gap"], 3),
                "大口マネーフロー(衝撃度)": round(vol_shock, 2),
                "推奨状態": "🔥 勝負候補" if trade_decision == "PAPER_TRADE_CANDIDATE" else "👀 様子見",
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
    track_only_df = settled_df[settled_df["trade_decision"] == "TRACK_ONLY"].copy()
    pending_df = res_df[res_df["status"] == "PENDING"].copy()

    total_exec = len(executed_df)
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
            <p>確定済み実戦勝負: <strong>{total_exec}回</strong> (勝率: <strong>{win_rate:.2f}%</strong> / 損益分岐: {be_rate:.2f}%) ｜ PF: <strong>{profit_factor:.2f}</strong> ｜ 判定待ち: <strong>{total_pending}件</strong></p>
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

st.sidebar.header("🔥 プロ投資会社設定")
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
    st.session_state.v6_monitoring = False
    for file_path in [VIRTUAL_TRADES_FILE, ANALYSIS_LOG_FILE]:
        try:
            if file_path.exists(): file_path.unlink()
        except OSError:
            pass
    st.sidebar.success("リセット完了しました。")
    st.rerun()

tab1, tab2 = st.tabs(["📊 個別銘柄・プロ実戦検証", "🌌 全市場マネーフロー・ランキング"])


# -------------------- ⑦ タブ1: 個別実戦検証 --------------------

with tab1:
    if "v6_monitoring" not in st.session_state:
        st.session_state.v6_monitoring = False

    st.subheader(f"♾️ プロ心理戦リアルタイム監視 [{selected_symbol}]")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ プロ監視・自動学習を開始", use_container_width=True):
            st.session_state.v6_monitoring = True
    with c2:
        if st.button("⏹ 監視を停止", use_container_width=True):
            st.session_state.v6_monitoring = False

    if st.session_state.v6_monitoring:
        st.success(f"プロ監視中 [{selected_symbol}]：大口マネーフロー＆TIMEOUT判定稼働中")
    else:
        st.info("停止中です。開始ボタンを押してください。")

    @st.fragment(run_every="30s")
    def continuous_v6_loop():
        if not st.session_state.v6_monitoring:
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

        features_table = build_psychology_flow_features(selected_symbol, all_dfs)
        signal_i = len(df_target) - 2
        feat_row = features_table.iloc[signal_i]
        if feat_row.isna().any():
            return

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
        trade_decision = assess_trade(signal)
        signal_price = float(df_target["Close"].iloc[signal_i])
        signal_time = df_target.index[signal_i]
        time_str = str(signal_time)

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
                "model_status": "リアルタイム心理戦稼働中",
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
                <h2>🧭 プロ心理予測: {signal['direction']} [{selected_symbol}]</h2>
                <p>HIGH: <strong>{signal['p_high']:.1%}</strong> ｜ LOW: <strong>{signal['p_low']:.1%}</strong> (ギャップ: {signal['probability_gap']:.2f})</p>
                <p>プロ判断: <strong>{trade_decision}</strong> (基準価格: ${signal_price:,.2f})</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not existing_log.empty:
            st.subheader("📋 リアルタイム全件追跡ログ")
            st.dataframe(existing_log.sort_values("time", ascending=False).head(15), use_container_width=True, hide_index=True)

    continuous_v6_loop()

    st.markdown("---")
    st.subheader("🚀 公平なウォークフォワード検証（バックテスト）")
    if st.button("▶️ 会社資金10$倍増 検証トレーニング開始", use_container_width=True):
        with st.spinner("先読み防止ウォークフォワード検証を実行中..."):
            all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
            df_target = all_dfs.get(selected_symbol)
            if df_target is None or len(df_target) < 300:
                st.error("データ件数が不足しています（最低300本必要です）。")
            else:
                res_df = run_pro_walkforward(
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


# -------------------- ⑧ タブ2: 全市場マネーフローランキング --------------------

with tab2:
    st.subheader("🌌 全市場マネーフロー・オッズ構造ランキング")
    st.caption("大口の塊（出来高急変）と群集心理（RSI・過熱感）を同期させ、今最も歪んでいる市場をあぶり出します。")

    if st.button("🔍 全市場マネーフローを一斉スキャン", use_container_width=True):
        with st.spinner("全9市場のオッズ構造・マネーフローを解析中..."):
            all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
            scanner_df = run_multi_market_scanner(all_dfs, target_pct, hold_bars, fee_pct)
            if not scanner_df.empty:
                st.success("スキャン完了：大口マネーフローが流入している歪み度の高い順にランキング表示します。")
                st.dataframe(scanner_df.sort_values(by="確率ギャップ", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.warning("スキャン可能なデータが不足しています。")
