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
# UTAYA グローバル・インベストメンツ (v10 プロフェッショナル・オーディット版)
# 思想: データリーク完全排除、厳格インデックス整合、実測検証、確率校正
# =========================================================

st.set_page_config(
    page_title="UTAYA Global Investments - v10 Professional Audit Engine",
    page_icon="🛡️",
    layout="wide",
)

SYMBOLS = {
    "BTC (暗号資産)": "BTC-USD",
    "ETH (暗号資産)": "ETH-USD",
    "SOL (暗号資産)": "SOL-USD",
    "XRP (暗号資産)": "XRP-USD",
    "BNB (暗号資産)": "BNB-USD",
    "WTI原油 (エネルギー)": "CL=F",
    "金スポット (貴金属)": "GC=F",
    "日経平均 (日本株)": "^N225",
    "S&P 500 (米国株)": "^GSPC",
    "Nasdaq 100 (米国株)": "^NDX",
    "USD/JPY (為替)": "USDJPY=X",
    "米10年債利回り (金利)": "^TNX",
}

PAYOUT_RATE = 0.88  # ペイアウト率88% (損益分岐勝率 約53.19%)
TARGET_ROI = 0.20   # 目標ROI 20% (10$ -> 12$)
DATA_DIR = Path("utaya_data_v10")
DATA_DIR.mkdir(exist_ok=True)

GLOBAL_HISTORY_DIR = DATA_DIR / "history"
GLOBAL_HISTORY_DIR.mkdir(exist_ok=True)
ANALYSIS_LOG_FILE = DATA_DIR / "analysis_log_v10.csv"

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

st.title("🛡️ UTAYA v10 プロフェッショナル・オーディットエンジン")
st.caption(
    "データリーク完全排除 | 厳格インデックス整合 | Brierスコア校正 | 10$➔12$ 統計検証"
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
                # 出来高がない場合はNaNとして扱い、架空の数値を埋めない（データリーク・歪み防止）
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
    for name, ticker in SYMBOLS.items():
        df = fetch_single_market_data(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            all_dfs[name] = df
    return all_dfs


# -------------------- ② 厳格インデックス整合・特徴量エンジン (v10) --------------------

def build_v10_features_and_labels(target_name, all_dfs):
    """
    特徴量と正解ラベルのインデックスを完全に一致させ、未来情報の混入（データリーク）をシャットアウトする
    """
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

    # 出来高が存在する場合のみZスコア計算、ない場合は0
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

    # 危機察知（大口衝突・価格摩擦）フラグ
    friction_signal = ((volume_z > 2.0) & (body_ratio < 0.25)).astype(float)
    features["emergency_friction"] = friction_signal

    # 他市場との連動フロー (未来情報を参照しないよう ffill は直近のみ許容、または安全に結合)
    for name, df_other in all_dfs.items():
        if name == target_name:
            continue
        df_other = df_other.copy()
        df_other.index = pd.to_datetime(df_other.index)
        if df_other.index.tz is not None:
            df_other.index = df_other.index.tz_localize(None)

        other_close = df_other["Close"].reindex(df_target.index, method="ffill", limit=3).astype(float)
        short_tag = name.split(" ")[0].lower()
        features[f"{short_tag}_ret1"] = other_close.pct_change(1)

    # 正解ラベル：次足の終値が当足の終値より高ければ1、低ければ0
    # インデックスを完全に整合させるため、シフトしたラベルを作成
    labels = (close.shift(-1) > close).astype(int)

    # 結合して欠損値を一括削除し、インデックスのズレを完全に防止
    combined = features.copy()
    combined["target"] = labels
    combined = combined.dropna()

    clean_features = combined.drop(columns=["target"])
    clean_labels = combined["target"].astype(int)

    return clean_features, clean_labels


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
    if is_emergency:
        return "EMERGENCY_HOLD (大口衝突・価格急変の危険回避の為強制見送り)"
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
            return "LOSS", "同一足で上下両方到達（価格急変・ノイズ）", "VOLATILITY_NOISE", str(idx)
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
    
    if "EMERGENCY_HOLD" in str(prediction["trade_decision"]):
        prediction["pnl_usd"] = 0.0
    elif prediction["trade_decision"] == "PAPER_TRADE_CANDIDATE":
        prediction["pnl_usd"] = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd
    else:
        prediction["pnl_usd"] = 0.0

    return prediction


# -------------------- ③ 厳格ウォークフォワード・バックテストエンジン (v10) --------------------

def run_v10_walkforward(target_name, all_dfs, target_pct, stake_usd, hold_bars, fee_pct):
    df_target = all_dfs.get(target_name)
    if df_target is None or len(df_target) < 300:
        return pd.DataFrame()

    features_df, labels_s = build_v10_features_and_labels(target_name, all_dfs)
    if len(features_df) < 100:
        return pd.DataFrame()

    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
    scaler = StandardScaler()

    init_end = int(len(features_df) * 0.3)
    
    # 初期学習データのみでスケーラーとモデルを学習（テスト期間の情報混入を完全防止）
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

        # 予測時点までに学習したscalerを用いて変換（未来のscaler情報リーク防止）
        X_pred = scaler.transform(np.array([feat_row.values]))

        try:
            if hasattr(model, "classes_") and len(model.classes_) == 2:
                probs = model.predict_proba(X_pred)[0]
                classes_list = list(model.classes_)
                p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
                p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
                model_status = "v10厳格検証稼働中"
            else:
                p_low, p_high = 0.5, 0.5
                model_status = "初期化中"
        except Exception:
            p_low, p_high = 0.5, 0.5
            model_status = "エラー"

        signal = predict_direction(p_high, p_low)
        trade_decision = assess_trade(signal, is_emergency=is_emergency)

        # 実エントリー価格と未来足の検証
        try:
            entry_idx_loc = df_target.index.get_loc(row_time) + 1
            entry_price = float(df_target["Open"].iloc[entry_idx_loc])
            future_window = df_target.iloc[entry_idx_loc : entry_idx_loc + hold_bars]
        except Exception:
            continue

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

        # 予測完了後、この瞬間に確定した正解ラベルでモデルをオンライン学習（厳格に未来リークなし）
        actual_label = int(labels_s.loc[row_time])
        try:
            model.partial_fit(X_pred, np.array([actual_label]), classes=np.array([0, 1]))
        except ValueError:
            model.partial_fit(X_pred, np.array([actual_label]))

        records.append({
            "time": str(row_time),
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
            "actual_label": actual_label,
        })

    progress_bar.empty()
    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")

    return res_df


# -------------------- ④ 全市場スキャナー (v10) --------------------

def run_v10_multi_market_scanner(all_dfs, target_pct, hold_bars, fee_pct):
    scanner_results = []
    for name, df in all_dfs.items():
        if df is None or len(df) < 50:
            continue
        try:
            features_df, labels_s = build_v10_features_and_labels(name, all_dfs)
            if len(features_df) < 20:
                continue

            signal_i = len(features_df) - 1
            feat_row = features_df.iloc[signal_i]
            row_time = features_df.index[signal_i]
            is_emergency = bool(feat_row.get("emergency_friction", 0) > 0)

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

            signal = predict_direction(p_high, p_low)
            trade_decision = assess_trade(signal, is_emergency=is_emergency)
            base_price = float(df["Close"].loc[row_time]) if row_time in df.index else float(df["Close"].iloc[-1])

            status_text = "🚨 【緊急警報】大口衝突・危険" if is_emergency else ("🔥 勝負候補" if trade_decision == "PAPER_TRADE_CANDIDATE" else "👀 様子見")

            scanner_results.append({
                "銘柄": name,
                "方向予測": signal["direction"],
                "モデル推定HIGH確率": f"{signal['p_high']:.1%}",
                "モデル推定LOW確率": f"{signal['p_low']:.1%}",
                "確率ギャップ": round(signal["probability_gap"], 3),
                "危機状態": status_text,
                "基準価格": round(base_price, 2)
            })
        except Exception:
            pass
    return pd.DataFrame(scanner_results)


# -------------------- ⑤ ダッシュボード・監査指標描画 --------------------

def show_audit_dashboard(res_df, stake_usd):
    if res_df.empty:
        st.warning("検証データがありません。バックテストを実行してください。")
        return

    settled_df = res_df[res_df["status"] == "SETTLED"].copy()
    executed_df = settled_df[settled_df["trade_decision"] == "PAPER_TRADE_CANDIDATE"].copy()
    emergency_held_df = settled_df[settled_df["trade_decision"].str.contains("EMERGENCY_HOLD", na=False)].copy()
    pending_df = res_df[res_df["status"] == "PENDING"].copy()

    total_exec = len(executed_df)
    total_emergency = len(emergency_held_df)
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

    # 最大ドローダウンの計算
    if not executed_df.empty:
        exec_sorted = executed_df.sort_values("time").copy()
        exec_sorted["cum_pnl"] = exec_sorted["pnl_usd"].cumsum()
        exec_sorted["peak"] = exec_sorted["cum_pnl"].cummax()
        exec_sorted["drawdown"] = exec_sorted["cum_pnl"] - exec_sorted["peak"]
        max_drawdown = float(exec_sorted["drawdown"].min())
    else:
        max_drawdown = 0.0

    # Brierスコア（確率校正の評価: 0に近いほど優秀）
    brier = 0.0
    if not settled_df.empty and "actual_label" in settled_df.columns:
        try:
            brier = brier_score_loss(settled_df["actual_label"], settled_df["p_high"])
        except Exception:
            brier = 0.0

    st.markdown(
        f"""
        <div class="genius-banner">
            <h2>💰 資金オーディット: 初期 ${stake_usd:.2f} ➔ 現在 ${current_capital:.2f} (実績ROI: {actual_roi:+.1f}%)</h2>
            <p>確定勝負: <strong>{total_exec}回</strong> (勝率: <strong>{win_rate:.2f}%</strong> / 損益分岐: {be_rate:.2f}%) ｜ 🚨 **緊急防御: {total_emergency}回** ｜ PF: <strong>{profit_factor:.2f}</strong> ｜ 最大DD: <strong>${max_drawdown:.2f}</strong></p>
            <p>確率校正 (Brier Score): <strong>{brier:.4f}</strong> (※0に近いほど確率予測が正確)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if total_exec < 30:
        st.warning("⚠️ 注意: 統計的に十分な取引回数（30回以上）に達していません。過剰な信頼は禁物です。")

    if actual_roi >= (TARGET_ROI * 100):
        st.success(f"🏆 目標達成！実績ROI {actual_roi:.1f}%（10$ $\rightarrow$ 12$ 達成証明）")
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
        st.info("確定した実戦データがありません。")

    st.subheader("📝 詳細プロ監査ログ (直近50件)")
    st.dataframe(
        res_df.sort_values("time", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )


# -------------------- ⑥ サイドバー設定 --------------------

st.sidebar.header("🛡️ v10 プロ監査設定")
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
if st.sidebar.button("🔄 学習ログ・キャッシュをリセット", use_container_width=True):
    st.session_state.v10_monitoring = False
    if ANALYSIS_LOG_FILE.exists():
        try:
            ANALYSIS_LOG_FILE.unlink()
        except OSError:
            pass
    st.sidebar.success("リセット完了しました。")
    st.rerun()

tab1, tab2 = st.tabs(["📊 個別銘柄・v10監査検証", "🌌 全市場マルチアセット・オーディット"])


# -------------------- ⑦ タブ1: 個別実戦検証 --------------------

with tab1:
    if "v10_monitoring" not in st.session_state:
        st.session_state.v10_monitoring = False

    st.subheader(f"♾️ v10 厳格監査リアルタイム監視 [{selected_symbol}]")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ v10監査監視を開始", use_container_width=True):
            st.session_state.v10_monitoring = True
    with c2:
        if st.button("⏹ 監視を停止", use_container_width=True):
            st.session_state.v10_monitoring = False

    if st.session_state.v10_monitoring:
        st.success(f"v10監視中 [{selected_symbol}]：データリーク防止・厳格インデックス整合稼働中")
    else:
        st.info("停止中です。開始ボタンを押してください。")

    @st.fragment(run_every="30s")
    def continuous_v10_loop():
        if not st.session_state.v10_monitoring:
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
                        row_dict = update_prediction_status(row_dict, df_target, hold_bars, target_pct, fee_pct, stake_usd)
                    updated_rows.append(row_dict)
                log_df = pd.DataFrame(updated_rows)
                log_df.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")
            except Exception:
                pass

        features_df, labels_s = build_v10_features_and_labels(selected_symbol, all_dfs)
        if len(features_df) < 10:
            return

        signal_i = len(features_df) - 1
        feat_row = features_df.iloc[signal_i]
        row_time = features_df.index[signal_i]
        is_emergency = bool(feat_row.get("emergency_friction", 0) > 0)

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

        signal = predict_direction(p_high, p_low)
        trade_decision = assess_trade(signal, is_emergency=is_emergency)
        signal_price = float(df_target["Close"].loc[row_time]) if row_time in df_target.index else float(df_target["Close"].iloc[-1])
        time_str = str(row_time)

        if is_emergency:
            st.markdown(
                """
                <div class="emergency-alert-card">
                    <h2>🚨 【緊急警報】大口衝突・価格摩擦の異常検知！</h2>
                    <p>厳格な需給・摩擦特徴量により、相場の危険領域を検知しました。<strong>すべての新規エントリーを強制シャットダウンします！</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if ANALYSIS_LOG_FILE.exists():
            existing_log = pd.read_csv(ANALYSIS_LOG_FILE)
        else:
            existing_log = pd.DataFrame(columns=["time", "symbol", "direction", "p_high", "p_low", "confidence", "probability_gap", "model_status", "trade_decision", "status", "result", "reason", "category", "settle_time", "entry_price", "pnl_usd", "actual_label"])

        if existing_log.empty or not ((existing_log["time"] == time_str) & (existing_log["symbol"] == selected_symbol)).any():
            future_window = df_target.loc[row_time:].iloc[1 : hold_bars + 1]
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

            actual_lbl = int(labels_s.loc[row_time]) if row_time in labels_s.index else 0

            new_row = pd.DataFrame([{
                "time": time_str,
                "symbol": selected_symbol,
                "direction": signal["direction"],
                "p_high": signal["p_high"],
                "p_low": signal["p_low"],
                "confidence": signal["confidence"],
                "probability_gap": signal["probability_gap"],
                "model_status": "v10リアルタイム監査稼働中",
                "trade_decision": trade_decision,
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

        card_class = "scenario-card-high" if signal["direction"] == "HIGH" else "scenario-card-low"
        st.markdown(
            f"""
            <div class="{card_class}">
                <h2>🧭 v10 方向予測: {signal['direction']} [{selected_symbol}]</h2>
                <p>HIGH推定確率: <strong>{signal['p_high']:.1%}</strong> ｜ LOW推定確率: <strong>{signal['p_low']:.1%}</strong> (ギャップ: {signal['probability_gap']:.2f})</p>
                <p>監査判断: <strong>{trade_decision}</strong> (基準価格: ${signal_price:,.2f})</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not existing_log.empty:
            st.subheader("📋 リアルタイム監査ログ")
            st.dataframe(existing_log.sort_values("time", ascending=False).head(15), use_container_width=True, hide_index=True)

    continuous_v10_loop()

    st.markdown("---")
    st.subheader("🚀 厳格ウォークフォワード検証（データリーク完全防止バックテスト）")
    if st.button("▶️ 会社資金10$倍増 v10厳格検証開始", use_container_width=True):
        with st.spinner("未来情報リークを完全に排除したウォークフォワード検証を実行中..."):
            all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
            df_target = all_dfs.get(selected_symbol)
            if df_target is None or len(df_target) < 300:
                st.error("データ件数が不足しています（最低300本必要です）。")
            else:
                res_df = run_v10_walkforward(
                    selected_symbol, all_dfs, target_pct=target_pct, stake_usd=stake_usd, hold_bars=hold_bars, fee_pct=fee_pct
                )
                show_audit_dashboard(res_df, stake_usd)
    elif ANALYSIS_LOG_FILE.exists():
        try:
            saved_df = pd.read_csv(ANALYSIS_LOG_FILE)
            if not saved_df.empty:
                show_audit_dashboard(saved_df, stake_usd)
        except Exception:
            pass


# -------------------- ⑧ タブ2: 全市場マルチアセット・オーディット --------------------

with tab2:
    st.subheader("🌌 v10 マルチアセット・オーディットスキャナー")
    st.caption("全12市場を対象に、厳格にインデックスを整合させた特徴量とモデル推定確率、危機状態を一覧表示します。")

    if st.button("🔍 全12市場を監査スキャン", use_container_width=True):
        with st.spinner("全12市場の厳格スキャンを実行中..."):
            all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
            scanner_df = run_v10_multi_market_scanner(all_dfs, target_pct, hold_bars, fee_pct)
            if not scanner_df.empty:
                st.success("スキャン完了：監査結果をランキング表示します。")
                st.dataframe(scanner_df, use_container_width=True, hide_index=True)
            else:
                st.warning("スキャン可能なデータが不足しています。")
