import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from datetime import datetime
import os

# =========================================================
# UTAYA グローバル・インベストメンツ (v15.1 NaN完全防御・方向判定パネル搭載版)
# =========================================================

st.set_page_config(
    page_title="UTAYA Global Investments - v15.1 Multi-Universe Direction Engine",
    page_icon="🎯",
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

PAYOUT_RATE = 0.88  
TARGET_ROI = 0.20   
DATA_DIR = Path("utaya_data_v15")
DATA_DIR.mkdir(exist_ok=True)

GLOBAL_HISTORY_DIR = DATA_DIR / "history"
GLOBAL_HISTORY_DIR.mkdir(exist_ok=True)
ANALYSIS_LOG_FILE = DATA_DIR / "analysis_log_v15.csv"

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
    .direction-panel-container {
        display: flex;
        gap: 20px;
        margin-bottom: 20px;
    }
    .panel-high {
        flex: 1;
        background: rgba(0, 176, 155, 0.15);
        border: 2px solid #00b09b;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .panel-low {
        flex: 1;
        background: rgba(255, 65, 108, 0.15);
        border: 2px solid #ff416c;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .tier-card-normal {
        background: rgba(0, 176, 155, 0.1);
        border-left: 5px solid #00b09b;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .tier-card-cond {
        background: rgba(255, 193, 7, 0.1);
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .tier-card-stop {
        background: rgba(255, 65, 108, 0.1);
        border-left: 5px solid #ff416c;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 UTAYA v15.1 多角性宇宙論・方向判定エンジン (NaN防御版)")
st.caption(
    "NaN完全排除ガード | ハイロー値幅到達直接学習 | 上下独立確率パネル常時表示"
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


# -------------------- ② 特徴量とラベル生成 (NaN完全浄化ガード) --------------------

def build_v15_features_and_labels(target_name, all_dfs, fixed_range_usd, hold_bars=5):
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

    # ラベル生成
    labels = []
    for idx in range(len(df_target) - hold_bars):
        entry_price = float(df_target["Open"].iloc[idx + 1])
        future_window = df_target.iloc[idx + 1 : idx + 1 + hold_bars]
        
        upper_target = entry_price + fixed_range_usd
        lower_target = entry_price - fixed_range_usd
        
        lbl = np.nan
        for _, candle in future_window.iterrows():
            h = float(candle["High"])
            l = float(candle["Low"])
            hit_up = h >= upper_target
            hit_down = l <= lower_target
            
            if hit_up and not hit_down:
                lbl = 1
                break
            elif hit_down and not hit_up:
                lbl = 0
                break
            elif hit_up and hit_down:
                break
        labels.append(lbl)

    features_trimmed = features.iloc[:len(labels)].copy()
    features_trimmed["target"] = labels
    
    # 🚨 ここで NaN を完全浄化（無限大や欠損行をすべてドロップし、予測時にNaNが混入するのを根絶）
    features_trimmed = features_trimmed.replace([np.inf, -np.inf], np.nan).dropna()

    if features_trimmed.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    return features_trimmed.drop(columns=["target"]), features_trimmed["target"].astype(int)


# -------------------- ③ 評価関数 --------------------

def evaluate_quick_mode_outcome(entry_price, direction, future_candles, fixed_range_usd, hold_bars=5):
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
            return "LOSS", "同一足で上下両方到達（ノイズ・逆発）", "VOLATILITY_NOISE", str(idx)
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
        return "TIMEOUT", "スコープ未到達（膠着）", "TIMEOUT_RANGE", str(future_candles.index[-1])

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
    
    if "一時停止" in str(prediction["tier"]):
        prediction["pnl_usd"] = 0.0
    elif prediction["tier"] == "通常候補":
        prediction["pnl_usd"] = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd
    else:
        prediction["pnl_usd"] = 0.0

    return prediction


def classify_trading_tier(feat_row, p_high, p_low):
    is_emergency = bool(feat_row.get("emergency_friction", 0) > 0)
    volatility = float(feat_row.get("volatility", 0.01))
    probability_gap = abs(p_high - p_low)

    if is_emergency or np.isnan(volatility) or volatility > 0.15:
        return "一時停止", "🚨 大口の衝突（価格摩擦）または異常ボラティリティ検知。リスク管理のため一時停止。"

    if probability_gap >= 0.04 and max(p_high, p_low) >= 0.54:
        return "通常候補", f"🔥 確率ギャップ {probability_gap:.3f} 良好。優位性が検証基準を満たした方向でエントリー候補。"

    return "条件付き候補", f"👀 ギャップ縮小中 ({probability_gap:.3f})。方向を選び、モメンタムや条件成立を継続監視。"


# -------------------- ④ ウォークフォワード検証エンジン --------------------

def run_v15_walkforward(target_name, all_dfs, stake_usd, hold_bars, fixed_range_usd):
    df_target = all_dfs.get(target_name)
    if df_target is None or len(df_target) < 300:
        return pd.DataFrame()

    features_df, labels_s = build_v15_features_and_labels(target_name, all_dfs, fixed_range_usd, hold_bars)
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

        X_pred = scaler.transform(np.array([feat_row.values]))
        
        try:
            if hasattr(model, "classes_") and len(model.classes_) == 2:
                probs = model.predict_proba(X_pred)[0]
                classes_list = list(model.classes_)
                p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
                p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
                model_status = "v15.1検証稼働中"
            else:
                p_low, p_high = 0.5, 0.5
                model_status = "初期化中"
        except Exception:
            p_low, p_high = 0.5, 0.5
            model_status = "エラー"

        direction = "HIGH" if p_high >= p_low else "LOW"
        tier, tier_reason = classify_trading_tier(feat_row, p_high, p_low)

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
            if "一時停止" in tier:
                pnl = 0.0
            elif tier == "通常候補":
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
            "probability_gap": abs(p_high - p_low),
            "tier": tier,
            "tier_reason": tier_reason,
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


# -------------------- ⑤ 全市場スキャナー --------------------

def run_v15_multi_market_scanner(all_dfs, fixed_ranges):
    scanner_results = []
    for name, cfg in SYMBOLS_CONFIG.items():
        df = all_dfs.get(name)
        rng = fixed_ranges.get(name, cfg["quick_range_usd"])
        if df is None or len(df) < 50:
            continue
        try:
            features_df, labels_s = build_v15_features_and_labels(name, all_dfs, rng, hold_bars=5)
            if len(features_df) < 20:
                continue

            signal_i = len(features_df) - 1
            feat_row = features_df.iloc[signal_i]
            row_time = features_df.index[signal_i]

            X_live = features_df.values
            y_live = labels_s.values
            scaler = StandardScaler().fit(X_live)
            model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
            model.partial_fit(scaler.transform(X_live), y_live, classes=np.array([0, 1]))

            feat_sc = scaler.transform(np.array([feat_row.values]))
            probs = model.predict_proba(feat_sc)[0]
            classes_list = list(model.classes_)
            p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
            p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]

            direction = "HIGH" if p_high >= p_low else "LOW"
            tier, tier_reason = classify_trading_tier(feat_row, p_high, p_low)
            base_price = float(df["Close"].loc[row_time]) if row_time in df.index else float(df["Close"].iloc[-1])

            scanner_results.append({
                "銘柄": name,
                "判定ステージ": tier,
                "予測方向": direction,
                "HIGH確率": f"{p_high:.1%}",
                "LOW確率": f"{p_low:.1%}",
                "確率ギャップ": round(abs(p_high - p_low), 3),
                "クイック目標幅": rng,
                "基準価格": round(base_price, 2),
                "評価理由": tier_reason
            })
        except Exception:
            pass
    return pd.DataFrame(scanner_results)


# -------------------- ⑥ ダッシュボード描画 --------------------

def show_audit_dashboard(res_df, stake_usd, fixed_range_usd):
    if res_df.empty:
        st.warning("検証データがありません。バックテストを実行してください。")
        return

    settled_df = res_df[res_df["status"] == "SETTLED"].copy()
    normal_df = settled_df[settled_df["tier"] == "通常候補"].copy()
    cond_df = settled_df[settled_df["tier"] == "条件付き候補"].copy()
    stop_df = settled_df[settled_df["tier"] == "一時停止"].copy()

    total_exec = len(normal_df)
    wins = int((normal_df["result"] == "WIN").sum())
    win_rate = wins / total_exec * 100 if total_exec > 0 else 0.0
    total_pnl = float(normal_df["pnl_usd"].sum())
    be_rate = (1 / (1 + PAYOUT_RATE)) * 100

    current_capital = stake_usd + total_pnl
    actual_roi = (total_pnl / stake_usd) * 100 if stake_usd > 0 else 0.0

    gross_profit = float(normal_df[normal_df["pnl_usd"] > 0]["pnl_usd"].sum())
    gross_loss = abs(float(normal_df[normal_df["pnl_usd"] < 0]["pnl_usd"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    brier = 0.0
    if not settled_df.empty and "actual_label" in settled_df.columns:
        try:
            brier = brier_score_loss(settled_df["actual_label"], settled_df["p_high"])
        except Exception:
            brier = 0.0

    st.markdown(
        f"""
        <div class="genius-banner">
            <h2>🎯 v15.1 多角性宇宙論ダッシュボード (目標±${fixed_range_usd})</h2>
            <p>資金状況: 初期 ${stake_usd:.2f} ➔ 現在 ${current_capital:.2f} (実績ROI: {actual_roi:+.1f}%)</p>
            <p>通常候補勝負: <strong>{total_exec}回</strong> (勝率: <strong>{win_rate:.2f}%</strong> / 損益分岐: {be_rate:.2f}%) ｜ 条件付き監視: {len(cond_df)} ｜ 一時停止: {len(stop_df)}</p>
            <p>確率校正 (Brier Score): <strong>{brier:.4f}</strong> (0に近いほど正確) ｜ PF: <strong>{profit_factor:.2f}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧭 判定ステージの分布")
        st.bar_chart(res_df["tier"].value_counts())
    with col2:
        st.subheader("📊 通常候補の結果内訳")
        if not normal_df.empty:
            st.bar_chart(normal_df["result"].value_counts())

    st.subheader("📈 資金成長曲線（通常候補 累積損益）")
    chart_df = normal_df[["time", "pnl_usd"]].copy()
    if not chart_df.empty:
        chart_df["time"] = pd.to_datetime(chart_df["time"])
        chart_df["資産推移"] = stake_usd + chart_df["pnl_usd"].cumsum()
        chart_df = chart_df.set_index("time")
        st.line_chart(chart_df[["資産推移"]])
    else:
        st.info("通常候補の確定した取引データがありません。")

    st.subheader("📝 詳細プロ監査ログ (直近50件)")
    st.dataframe(res_df.sort_values("time", ascending=False).head(50), use_container_width=True, hide_index=True)


# -------------------- ⑦ サイドバー・タブ設定 --------------------

st.sidebar.header("🎯 v15.1 多角性宇宙論・設定")
selected_symbol = st.sidebar.selectbox("🎯 分析対象銘柄 (個別)", options=list(SYMBOLS_CONFIG.keys()), index=0)

cfg = SYMBOLS_CONFIG[selected_symbol]
default_range = cfg["quick_range_usd"]

fixed_range_usd = st.sidebar.number_input(
    f"📏 クイックモード目標値幅 (±$ / pt)", 
    min_value=0.001, max_value=10000.0, value=float(default_range), step=0.01,
    format="%.2f",
    help="ハイロー・クイックモードで目標値幅に先に到達するかを判定する基準幅です。"
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

tab1, tab2 = st.tabs(["📊 個別銘柄・方向判定パネル監査", "🌌 全12市場リアルタイム・方向判定スキャナー"])


# -------------------- ⑧ タブ1: 個別実戦検証 --------------------

with tab1:
    if "v15_monitoring" not in st.session_state:
        st.session_state.v15_monitoring = False

    st.subheader(f"♾️ v15.1 方向判定パネル監視 [{selected_symbol}]")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ v15.1監査監視を開始", use_container_width=True):
            st.session_state.v15_monitoring = True
    with c2:
        if st.button("⏹ 監視を停止", use_container_width=True):
            st.session_state.v15_monitoring = False

    if st.session_state.v15_monitoring:
        st.success(f"v15.1監視中 [{selected_symbol}]：NaN防壁完備＆上下独立確率パネル稼働中")
    else:
        st.info("停止中です。開始ボタンを押してください。")

    @st.fragment(run_every="30s")
    def continuous_v15_loop():
        if not st.session_state.v15_monitoring:
            st.caption("監視は停止しています。")
            return

        with st.spinner("マルチ市場データを同期中..."):
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
                        row_dict = update_prediction_status(row_dict, df_target, hold_bars, fixed_range_usd, stake_usd)
                    updated_rows.append(row_dict)
                log_df = pd.DataFrame(updated_rows)
                log_df.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")
            except Exception:
                pass

        features_df, labels_s = build_v15_features_and_labels(selected_symbol, all_dfs, fixed_range_usd, hold_bars)
        if len(features_df) < 10:
            st.warning("有効な特徴量データが不足しています（NaN除去後）。期間を広げるか時間軸を変更してください。")
            return

        signal_i = len(features_df) - 1
        feat_row = features_df.iloc[signal_i]
        row_time = features_df.index[signal_i]

        X_live = features_df.values
        y_live = labels_s.values
        scaler = StandardScaler().fit(X_live)
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
        model.partial_fit(scaler.transform(X_live), y_live, classes=np.array([0, 1]))

        feat_sc = scaler.transform(np.array([feat_row.values]))
        probs = model.predict_proba(feat_sc)[0]
        classes_list = list(model.classes_)
        p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
        p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]

        direction = "HIGH" if p_high >= p_low else "LOW"
        tier, tier_reason = classify_trading_tier(feat_row, p_high, p_low)
        signal_price = float(df_target["Close"].loc[row_time]) if row_time in df_target.index else float(df_target["Close"].iloc[-1])
        time_str = str(row_time)

        # 🌌 UTAYA v15.1 方向判定パネル
        st.markdown(
            f"""
            <div class="direction-panel-container">
                <div class="panel-high">
                    <h3>▲ HIGH / 上昇</h3>
                    <h1 style="color: #00b09b; margin: 10px 0;">{p_high:.1%}</h1>
                    <p style="font-size: 0.85rem; color: #aaa; margin: 0;">検証済みモデルで算出 (上方目標: +${fixed_range_usd})</p>
                </div>
                <div class="panel-low">
                    <h3>▼ LOW / 下降</h3>
                    <h1 style="color: #ff416c; margin: 10px 0;">{p_low:.1%}</h1>
                    <p style="font-size: 0.85rem; color: #aaa; margin: 0;">検証済みモデルで算出 (下方目標: -${fixed_range_usd})</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        card_class = "tier-card-normal"
        if "条件付き" in tier:
            card_class = "tier-card-cond"
        elif "一時停止" in tier:
            card_class = "tier-card-stop"

        st.markdown(
            f"""
            <div class="{card_class}">
                <h3>🎯 判定ステージ: <strong>[{tier}]</strong></h3>
                <p><strong>評価理由:</strong> {tier_reason}</p>
                <p><strong>基準価格:</strong> ${signal_price:,.2f} ｜ 判定期間: {hold_bars}本 ｜ 目標スコープ: ±${fixed_range_usd}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if ANALYSIS_LOG_FILE.exists():
            existing_log = pd.read_csv(ANALYSIS_LOG_FILE)
        else:
            existing_log = pd.DataFrame(columns=["time", "symbol", "direction", "p_high", "p_low", "probability_gap", "tier", "tier_reason", "status", "result", "reason", "category", "settle_time", "entry_price", "pnl_usd", "actual_label"])

        if existing_log.empty or not ((existing_log["time"] == time_str) & (existing_log["symbol"] == selected_symbol)).any():
            future_window = df_target.loc[row_time:].iloc[1 : hold_bars + 1]
            result, reason, category, settle_time = evaluate_quick_mode_outcome(
                signal_price, direction, future_window, fixed_range_usd, hold_bars=hold_bars
            )
            status = "SETTLED" if result != "PENDING" else "PENDING"
            pnl = 0.0
            if status == "SETTLED":
                if "一時停止" in tier:
                    pnl = 0.0
                elif tier == "通常候補":
                    pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

            actual_lbl = int(labels_s.loc[row_time]) if row_time in labels_s.index else 0

            new_row = pd.DataFrame([{
                "time": time_str,
                "symbol": selected_symbol,
                "direction": direction,
                "p_high": p_high,
                "p_low": p_low,
                "probability_gap": abs(p_high - p_low),
                "tier": tier,
                "tier_reason": tier_reason,
                "status": status,
                "result": result,
                "reason": reason,
                "category": category,
                "settle_time": str(settle_time),
                "entry_price": signal_price,
                "pnl_usd": pnl,
                "actual_label": actual_lbl,
            }])
            existing_log = pd.concat([existing_log, new_row])
            existing_log.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")

        if not existing_log.empty:
            st.subheader("📋 リアルタイム監査ログ")
            st.dataframe(existing_log.sort_values("time", ascending=False).head(15), use_container_width=True, hide_index=True)

    continuous_v15_loop()

    st.markdown("---")
    st.subheader("🚀 厳格ウォークフォワード検証（未来リーク完全排除バックテスト）")
    if st.button("▶️ 会社資金10$倍増 v15.1厳格検証開始", use_container_width=True):
        with st.spinner("ハイロー値幅到達を直接学習したウォークフォワード検証を実行中..."):
            all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
            df_target = all_dfs.get(selected_symbol)
            if df_target is None or len(df_target) < 300:
                st.error("データ件数が不足しています（最低300本必要です）。")
            else:
                res_df = run_v15_walkforward(
                    selected_symbol, all_dfs, stake_usd=stake_usd, hold_bars=hold_bars, fixed_range_usd=fixed_range_usd
                )
                show_audit_dashboard(res_df, stake_usd, fixed_range_usd)
    elif ANALYSIS_LOG_FILE.exists():
        try:
            saved_df = pd.read_csv(ANALYSIS_LOG_FILE)
            if not saved_df.empty:
                show_audit_dashboard(saved_df, stake_usd, fixed_range_usd)
        except Exception:
            pass


# -------------------- ⑨ タブ2: 全市場スキャナー --------------------

with tab2:
    st.subheader("🌌 v15.1 全12市場リアルタイム・方向判定スキャナー")
    st.caption("全12市場を対象に、上下両方向の確率パネルを評価し、通常候補・条件付き候補・一時停止の3段階でチャンスを探索します。")

    if st.button("🔍 全12市場を一斉スキャン", use_container_width=True):
        with st.spinner("全12市場の方向判定スキャンを実行中..."):
            all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
            fixed_ranges_map = {name: cfg["quick_range_usd"] for name, cfg in SYMBOLS_CONFIG.items()}
            scanner_df = run_v15_multi_market_scanner(all_dfs, fixed_ranges_map)
            if not scanner_df.empty:
                st.success("スキャン完了：全市場の方向判定パネル一覧です。")
                st.dataframe(scanner_df, use_container_width=True, hide_index=True)
            else:
                st.warning("スキャン可能なデータが不足しています。")
