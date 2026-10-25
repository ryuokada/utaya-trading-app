
# ============================================================
# UTAYA BTC HILO - Auto Learning & Cause Analysis Edition
# 長期1時間足モデルと短期1分足モデルを分離した完全版
#
# 主な機能:
# 1. 1年以上を目標とした1時間足データ取得
# 2. 短期1分足データの別取得・別評価
# 3. ウォークフォワード・バックテスト
# 4. HIGH / LOW の方向予測import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

# =========================================================
# UTAYA 仮想平行宇宙・未来シナリオ分析AI (完全自律仮想トレード・修正版)
# =========================================================

st.set_page_config(
    page_title="UTAYA Multiverse Scenario AI - Autonomous Trading",
    page_icon="🌌",
    layout="wide",
)

SYMBOL = "BTC-USD"
PAYOUT_RATE = 0.88
DEFAULT_TARGET_USD = 50.0
DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(exist_ok=True)

MARKET_HISTORY = DATA_DIR / "btc_1m_history.csv"
VIRTUAL_TRADES_FILE = DATA_DIR / "virtual_trades_history.csv"
GENIUS_HISTORY = DATA_DIR / "genius_ai_history.csv"
MODEL_FILE = DATA_DIR / "genius_model.joblib"
SCALER_FILE = DATA_DIR / "genius_scaler.joblib"

# -------------------- CSS --------------------

st.markdown(
    """
    <style>
    .genius-banner {
        background: linear-gradient(135deg, #1f4068, #e43f5a);
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

st.title("🌌 UTAYA 自律仮想トレード・敗因分析システム")
st.caption(
    "自動仮想エントリー | 敗因詳細記録 | 4連敗安全停止 & 手動解除 | 継続学習"
)


# -------------------- ① 永続データ保存・取得エンジン --------------------

@st.cache_data(ttl=30, show_spinner=False)
def fetch_latest_market_data():
    try:
        df_new = yf.download(
            SYMBOL,
            period="1d",
            interval="1m",
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
                df_new = df_new[(df_new["Close"] > 0) & (df_new["High"] >= df_new["Low"])]
                df_new = df_new[~df_new.index.duplicated(keep="last")].sort_index()

                if MARKET_HISTORY.exists():
                    try:
                        df_old = pd.read_csv(MARKET_HISTORY, index_col=0, parse_dates=True)
                        df_combined = pd.concat([df_old, df_new])
                        df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
                        df_combined.tail(10000).to_csv(MARKET_HISTORY, encoding="utf-8-sig")
                    except Exception:
                        df_new.to_csv(MARKET_HISTORY, encoding="utf-8-sig")
                else:
                    df_new.to_csv(MARKET_HISTORY, encoding="utf-8-sig")

        if MARKET_HISTORY.exists():
            df_stored = pd.read_csv(MARKET_HISTORY, index_col=0, parse_dates=True)
            return df_stored.sort_index()
        return df_new
    except Exception:
        if MARKET_HISTORY.exists():
            return pd.read_csv(MARKET_HISTORY, index_col=0, parse_dates=True).sort_index()
        return None


# -------------------- ② 特徴量一括計算エンジン --------------------

def build_feature_table(df):
    close = df["Close"].astype(float)
    returns = close.pct_change()
    delta = close.diff()
    
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(losses != 0, 100.0)
    rsi = rsi.where(gains != 0, 50.0)

    features = pd.DataFrame(index=df.index)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["return_5"] = close.pct_change(5)
    features["sma10_ratio"] = (close - close.rolling(10).mean()) / close.rolling(10).mean()
    features["sma30_ratio"] = (close - close.rolling(30).mean()) / close.rolling(30).mean()
    features["rsi"] = rsi
    features["volatility"] = returns.rolling(14).std()

    return features.replace([np.inf, -np.inf], np.nan)


def judge_first_touch_with_reason(entry, direction, future, target=DEFAULT_TARGET_USD):
    if direction not in ("HIGH", "LOW") or future is None or future.empty:
        return 0, "LOSS", "判定データ不足"

    upper = float(entry) + float(target)
    lower = float(entry) - float(target)

    for _, candle in future.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])

        hit_up = high >= upper
        hit_down = low <= lower

        if hit_up and hit_down:
            return 0, "LOSS", "同一足で上下両方到達 (順序不明)"
        if hit_up:
            if direction == "HIGH":
                return 1, "WIN", "上目標先着 (予測合致)"
            else:
                return 0, "LOSS", "逆方向到達 (上目標先着)"
        if hit_down:
            if direction == "LOW":
                return 1, "WIN", "下目標先着 (予測合致)"
            else:
                return 0, "LOSS", "逆方向到達 (下目標先着)"

    return 0, "LOSS", "判定期間内未到達 (目標未到達)"


# -------------------- ③ モデル永続化管理 --------------------

def load_or_init_model():
    if MODEL_FILE.exists() and SCALER_FILE.exists():
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            return model, scaler
        except Exception:
            pass
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
    scaler = StandardScaler()
    return model, scaler

def save_model_state(model, scaler):
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)


# -------------------- ④ バックテスト・天才育成トレーニング --------------------

def run_genius_simulation(df, target_usd, stake_usd, hold_bars):
    if df is None or len(df) < 300:
        return pd.DataFrame()

    features_df = build_feature_table(df)
    model, scaler = load_or_init_model()

    init_end = int(len(df) * 0.2)
    X_init, y_init = [], []
    
    for j in range(30, init_end):
        feat_row = features_df.iloc[j]
        if j + hold_bars < len(df) and not feat_row.isna().any():
            fut_close = float(df["Close"].iloc[j + hold_bars])
            curr_close = float(df["Close"].iloc[j])
            label = 1 if fut_close > curr_close else 0
            X_init.append(feat_row.values)
            y_init.append(label)

    if len(X_init) > 20:
        X_init = np.array(X_init)
        y_init = np.array(y_init)
        X_init_scaled = scaler.fit_transform(X_init)
        model.partial_fit(X_init_scaled, y_init, classes=np.array([0, 1]))

    records = []
    start_i = init_end
    progress_bar = st.progress(0)
    total_steps = (len(df) - hold_bars - 1) - start_i

    for idx, i in enumerate(range(start_i, len(df) - hold_bars - 1)):
        if total_steps > 0 and idx % 50 == 0:
            progress_bar.progress(min(1.0, idx / total_steps))

        feat_row = features_df.iloc[i]
        if feat_row.isna().any():
            continue

        X_pred = scaler.transform(np.array([feat_row.values]))
        
        try:
            prob = model.predict_proba(X_pred)[0]
            p_low, p_high = prob[0], prob[1]
        except Exception:
            p_low, p_high = 0.5, 0.5

        direction = "HIGH" if p_high >= p_low else "LOW"
        confidence = max(p_high, p_low)

        entry_index = i + 1
        entry_price = float(df["Open"].iloc[entry_index])
        future = df.iloc[entry_index : entry_index + hold_bars]

        win_flag, result, reason = judge_first_touch_with_reason(entry_price, direction, future, target_usd)
        pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

        # 初回または未初期化対策として classes を明示
        try:
            model.partial_fit(X_pred, np.array([win_flag]))
        except ValueError:
            model.partial_fit(X_pred, np.array([win_flag]), classes=np.array([0, 1]))

        records.append({
            "time": str(df.index[i]),
            "direction": direction,
            "confidence": confidence,
            "p_high": p_high,
            "p_low": p_low,
            "result": result,
            "reason": reason,
            "pnl_usd": pnl,
        })

    progress_bar.empty()
    save_model_state(model, scaler)

    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df.to_csv(GENIUS_HISTORY, index=False, encoding="utf-8-sig")

    return res_df


# -------------------- ⑤ ダッシュボード描画 --------------------

def show_genius_dashboard(res_df):
    if res_df.empty:
        st.warning("検証データがありません。")
        return

    total = len(res_df)
    wins = int((res_df["result"] == "WIN").sum())
    losses = int((res_df["result"] == "LOSS").sum())
    win_rate = wins / total * 100 if total else 0.0
    total_pnl = float(res_df["pnl_usd"].sum())

    st.markdown(
        f"""
        <div class="genius-banner">
            <h2>🏆 天才AI 育成実績レポート（総レッスン数: {total}回）</h2>
            <p>実践勝率: <strong>{win_rate:.2f}%</strong> ｜ 獲得仮想損益: <strong>${total_pnl:+.2f}</strong> (損益分岐基準: 53.19%)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if win_rate >= 53.19:
        st.success(f"🌟 現在の勝率 ({win_rate:.2f}%) は損益分岐点 (53.19%) を超えています！")
    else:
        st.warning(f"⚠️ 現在の勝率 ({win_rate:.2f}%) は損益分岐点を下回っています。")

    st.subheader("🧭 選択方向の偏り診断 (HIGH / LOW)")
    pred_counts = res_df["direction"].value_counts().reindex(["HIGH", "LOW"], fill_value=0)
    st.bar_chart(pred_counts)

    st.subheader("📊 敗因詳細の分析")
    if "reason" in res_df.columns:
        reason_counts = res_df[res_df["result"] == "LOSS"]["reason"].value_counts()
        if not reason_counts.empty:
            st.bar_chart(reason_counts)
        else:
            st.info("敗因データがまだありません。")

    st.subheader("📈 天才AIの成長曲線（累積損益の推移）")
    chart_df = res_df[["time", "pnl_usd"]].copy()
    chart_df["time"] = pd.to_datetime(chart_df["time"])
    chart_df["累積損益"] = chart_df["pnl_usd"].cumsum()
    chart_df = chart_df.set_index("time")
    st.line_chart(chart_df[["累積損益"]])

    st.subheader("📝 AIの反省・学習ログ (直近50件)")
    st.dataframe(
        res_df.sort_values("time", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )


# -------------------- ⑥ サイドバー設定 & リセット機能 --------------------

st.sidebar.header("🌌 天才育成パラメータ")
target_usd = st.sidebar.slider("🎯 目標値幅 (±USD)", 10.0, 100.0, DEFAULT_TARGET_USD, 5.0)
stake_usd = st.sidebar.number_input("💵 仮想賭け金 ($)", 1.0, 10000.0, 5.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (分足の本数)", 1, 5, 1)
period = st.sidebar.selectbox("📅 学習データ期間", ["1d", "2d", "5d", "7d"], index=3)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ 安全・停止管理")
if "force_stopped" not in st.session_state:
    st.session_state.force_stopped = False

if st.session_state.force_stopped:
    st.sidebar.warning("⚠️ 4連敗検知により自動停止中です。")
    if st.sidebar.button("🔓 停止を手動解除して再開", use_container_width=True):
        st.session_state.force_stopped = False
        st.sidebar.success("停止状態を解除しました！")
        st.rerun()
else:
    st.sidebar.success("正常稼働中（安全装置有効）")

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 データ・学習状態の管理")
reset_market_data = st.sidebar.checkbox("蓄積した市場データも削除する", value=False)
if st.sidebar.button("🔄 AI履歴・学習状態をリセット", use_container_width=True):
    st.session_state.utaya_monitoring = False
    files_to_delete = [VIRTUAL_TRADES_FILE, GENIUS_HISTORY, MODEL_FILE, SCALER_FILE]
    if reset_market_data:
        files_to_delete.append(MARKET_HISTORY)
    for file_path in files_to_delete:
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError as e:
            st.error(f"削除できませんでした: {file_path.name}: {e}")
    st.sidebar.success("リセット完了しました。ページを再読み込みしてください。")
    st.rerun()


# -------------------- ⑦ 連続監視・自動仮想トレードエンジン --------------------

if "utaya_monitoring" not in st.session_state:
    st.session_state.utaya_monitoring = False

st.subheader("♾️ UTAYA AI 連続監視・自動仮想トレードモード")
col1, col2 = st.columns(2)
with col1:
    if st.button("▶ 連続監視を開始", use_container_width=True):
        st.session_state.utaya_monitoring = True
with col2:
    if st.button("⏹ 連続監視を停止", use_container_width=True):
        st.session_state.utaya_monitoring = False

if st.session_state.utaya_monitoring and not st.session_state.force_stopped:
    st.success("連続監視・自動仮想トレード中：30秒ごとにシグナルを判定し、仮想トレードを実行します")
elif st.session_state.force_stopped:
    st.error("🚨 4連敗安全停止発動中。サイドバーの「停止を手動解除して再開」を押してください。")
else:
    st.info("停止中です。開始ボタンで連続監視を開始できます。")

@st.fragment(run_every="30s")
def continuous_virtual_trading():
    if not st.session_state.utaya_monitoring or st.session_state.force_stopped:
        st.caption("監視は停止しているか、安全装置によって一時中断されています。")
        return

    with st.spinner("BTC市場データを更新し、自動仮想トレードを判定中..."):
        live_df = fetch_latest_market_data()

    if live_df is None or len(live_df) < 100:
        st.warning("データが不足しています。")
        return

    signal_i = len(live_df) - 2
    features_table = build_feature_table(live_df)
    feat_row = features_table.iloc[signal_i]

    if feat_row.isna().any():
        st.warning("特徴量計算エラー")
        return

    model, scaler = load_or_init_model()

    X_live = features_table.dropna().values
    if len(X_live) > 30:
        scaler.partial_fit(X_live)

    feat_sc = scaler.transform(np.array([feat_row.values]))
    try:
        probs = model.predict_proba(feat_sc)[0]
        p_low, p_high = probs[0], probs[1]
    except Exception:
        p_low, p_high = 0.5, 0.5

    live_direction = "HIGH" if p_high >= p_low else "LOW"
    signal_price = float(live_df["Close"].iloc[signal_i])
    signal_time = live_df.index[signal_i]

    st.caption(f"最終更新時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 基準価格: ${signal_price:,.2f}")

    if live_direction == "HIGH":
        st.markdown(
            f"""
            <div class="scenario-card-high">
                <h2>🚀 連続監視 予測判定: HIGH</h2>
                <p>HIGH確率：<strong>{p_high:.1%}</strong> ｜ LOW確率：{p_low:.1%}</p>
                <p>上方向目標価格: ${signal_price + target_usd:,.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="scenario-card-low">
                <h2>📉 連続監視 予測判定: LOW</h2>
                <p>LOW確率：<strong>{p_low:.1%}</strong> ｜ HIGH確率：{p_high:.1%}</p>
                <p>下方向目標価格: ${signal_price - target_usd:,.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if VIRTUAL_TRADES_FILE.exists():
        try:
            trades_df = pd.read_csv(VIRTUAL_TRADES_FILE)
        except Exception:
            trades_df = pd.DataFrame(columns=["time", "price", "direction", "result", "reason", "pnl_usd"])
    else:
        trades_df = pd.DataFrame(columns=["time", "price", "direction", "result", "reason", "pnl_usd"])

    time_str = str(signal_time)
    if trades_df.empty or time_str not in trades_df["time"].values:
        future_window = live_df.loc[signal_time:].iloc[1 : hold_bars + 1]
        if len(future_window) >= hold_bars:
            win_flag, result, reason = judge_first_touch_with_reason(signal_price, live_direction, future_window, target_usd)
            pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

            try:
                model.partial_fit(feat_sc, np.array([win_flag]))
            except ValueError:
                model.partial_fit(feat_sc, np.array([win_flag]), classes=np.array([0, 1]))

            save_model_state(model, scaler)

            new_trade = pd.DataFrame([{
                "time": time_str,
                "price": signal_price,
                "direction": live_direction,
                "result": result,
                "reason": reason,
                "pnl_usd": pnl
            }])
            trades_df = pd.concat([trades_df, new_trade]).tail(500)
            trades_df.to_csv(VIRTUAL_TRADES_FILE, index=False, encoding="utf-8-sig")

            recent_results = trades_df["result"].tail(4).tolist()
            if len(recent_results) == 4 and all(r == "LOSS" for r in recent_results):
                st.session_state.force_stopped = True
                st.warning("⚠️ 4連続で敗北を検知したため、安全のために自動仮想トレードを停止しました。")
                st.rerun()

    if not trades_df.empty:
        st.subheader("📋 自律仮想トレード履歴・敗因ログ")
        st.dataframe(trades_df.sort_values("time", ascending=False).head(20), use_container_width=True, hide_index=True)
        
        v_wins = int((trades_df["result"] == "WIN").sum())
        v_losses = int((trades_df["result"] == "LOSS").sum())
        v_rate = v_wins / len(trades_df) * 100 if len(trades_df) > 0 else 0.0
        v_pnl = float(trades_df["pnl_usd"].sum())
        st.caption(f"仮想実績: {v_wins}勝 {v_losses}敗 (勝率: {v_rate:.1f}%) | 累積損益: ${v_pnl:+.2f}")

continuous_virtual_trading()


# -------------------- ⑧ バックテスト実行ボタン --------------------

st.markdown("---")
st.subheader("🚀 天才育成シミュレーション（一括検証）の実行")

if st.button("▶️ 天才育成トレーニングを開始", use_container_width=True):
    with st.spinner("市場データを取得し、永続モデルによる適応学習トレーニングを実行中..."):
        full_df = fetch_latest_market_data()

        if full_df is None or len(full_df) < 300:
            st.error("データ件数が不足しています（最低300本必要です）。しばらく待ってから再試行してください。")
        else:
            res_df = run_genius_simulation(
                full_df,
                target_usd=target_usd,
                stake_usd=stake_usd,
                hold_bars=hold_bars,
            )
            show_genius_dashboard(res_df)
elif GENIUS_HISTORY.exists():
    try:
        saved_df = pd.read_csv(GENIUS_HISTORY)
        if not saved_df.empty:
            show_genius_dashboard(saved_df)
    except Exception:
        pass
# 5. ±目標値幅の先着判定
# 6. 勝敗原因・相場環境別分析
# 7. 履歴のCSV保存
# 8. 自動監視・連敗時の安全停止
# 9. モデルの保存・リセット
#
# 注意: シミュレーターであり、実注文は行いません。
# ============================================================

import time
import warnings
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- 基本設定 --------------------

st.set_page_config(
    page_title="UTAYA BTC HILO - Cause Analysis AI",
    page_icon="⚡",
    layout="wide",
)

APP_NAME = "UTAYA BTC HILO"
SYMBOL = "BTC-USD"
PAYOUT_RATE = 0.88
DEFAULT_TARGET_USD = 50.0

DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

HOURLY_FILE = DATA_DIR / "btc_hourly_history.csv"
MINUTE_FILE = DATA_DIR / "btc_minute_history.csv"
HISTORY_FILE = DATA_DIR / "learning_history.csv"
HOURLY_MODEL_FILE = DATA_DIR / "hourly_model.joblib"
MINUTE_MODEL_FILE = DATA_DIR / "minute_model.joblib"

FEATURES = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "sma10_gap",
    "sma30_gap",
    "sma50_gap",
    "rsi14",
    "volatility10",
    "volatility24",
    "range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "volume_change",
    "volume_z",
]

# -------------------- デザイン --------------------

st.markdown(
    """
    <style>
    .main-banner-high {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 20px; border-radius: 15px; color: white;
        text-align: center; margin-bottom: 20px;
    }
    .main-banner-low {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 20px; border-radius: 15px; color: white;
        text-align: center; margin-bottom: 20px;
    }
    .main-banner-neutral {
        background: linear-gradient(135deg, #355c7d, #6c5b7b);
        padding: 20px; border-radius: 15px; color: white;
        text-align: center; margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ UTAYA BTC HILO - 自動学習・敗因分析AI")
st.caption(
    "長期1時間足モデルと短期1分足モデルを分離 | "
    "ウォークフォワード検証 | 仮想取引専用"
)

# -------------------- データ整形 --------------------

def normalize_ohlc(df):
    """Yahoo Finance形式を統一して、OHLCVを検査する。"""
    if df is None or df.empty:
        return None

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).title() for c in df.columns]

    required = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in required):
        return None

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df = df[
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
        & (df["High"] >= df["Low"])
        & (df["High"] >= df["Open"])
        & (df["High"] >= df["Close"])
        & (df["Low"] <= df["Open"])
        & (df["Low"] <= df["Close"])
    ]

    return df if not df.empty else None


def merge_and_save_history(new_df, file_path, max_rows=None):
    """取得データを既存CSVと統合し、時刻重複を除いて保存する。"""
    if new_df is None or new_df.empty:
        if file_path.exists():
            try:
                return normalize_ohlc(
                    pd.read_csv(file_path, index_col=0, parse_dates=True)
                )
            except Exception:
                return None
        return None

    frames = [new_df]

    if file_path.exists():
        try:
            old_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            old_df = normalize_ohlc(old_df)
            if old_df is not None:
                frames.insert(0, old_df)
        except Exception:
            pass

    merged = pd.concat(frames)
    merged = normalize_ohlc(merged)

    if merged is None:
        return None

    if max_rows is not None:
        merged = merged.tail(max_rows)

    merged.to_csv(file_path, encoding="utf-8-sig")
    return merged


@st.cache_data(ttl=300, show_spinner=False)
def download_market_data(interval, period):
    """Yahoo Financeから指定時間足のデータを取得する。"""
    try:
        raw = yf.download(
            SYMBOL,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return normalize_ohlc(raw)
    except Exception as exc:
        return None


def get_market_data(mode="hourly", refresh=False):
    """
    長期モデルと短期モデルのデータを分離。
    1時間足は最大2年を要求し、1年以上の取得を目指す。
    1分足は取得元の制限により短期間となる。
    """
    if mode == "hourly":
        interval = "1h"
        period = "2y"
        file_path = HOURLY_FILE
        max_rows = None
    else:
        interval = "1m"
        period = "7d"
        file_path = MINUTE_FILE
        max_rows = 20000

    if refresh:
        download_market_data.clear()

    downloaded = download_market_data(interval, period)
    merged = merge_and_save_history(downloaded, file_path, max_rows)

    # ダウンロード失敗時でも、保存済みCSVがあれば利用する。
    if merged is None and file_path.exists():
        try:
            merged = normalize_ohlc(
                pd.read_csv(file_path, index_col=0, parse_dates=True)
            )
        except Exception:
            merged = None

    return merged


# -------------------- 特徴量作成 --------------------

def build_features(df):
    """未来情報を使わず、現在までのOHLCVから特徴量を作成する。"""
    if df is None or len(df) < 60:
        return None

    x = df.copy()
    close = x["Close"].astype(float)
    open_price = x["Open"].astype(float)
    high = x["High"].astype(float)
    low = x["Low"].astype(float)
    volume = x["Volume"].astype(float)

    returns = close.pct_change()

    for n in [1, 3, 6, 12, 24]:
        x[f"ret_{n}"] = close.pct_change(n)

    for n in [10, 30, 50]:
        sma = close.rolling(n).mean()
        x[f"sma{n}_gap"] = close / sma - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi14"] = (100 - 100 / (1 + rs)) / 100

    x["volatility10"] = returns.rolling(10).std()
    x["volatility24"] = returns.rolling(24).std()
    x["range_pct"] = (high - low) / close
    x["body_pct"] = (close - open_price) / open_price
    x["upper_wick_pct"] = (high - np.maximum(open_price, close)) / close
    x["lower_wick_pct"] = (np.minimum(open_price, close) - low) / close

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    x["volume_change"] = volume.pct_change().replace(
        [np.inf, -np.inf], np.nan
    )
    x["volume_z"] = (
        (volume - volume_mean) / volume_std.replace(0, np.nan)
    )

    x[FEATURES] = x[FEATURES].replace([np.inf, -np.inf], np.nan)
    x[FEATURES] = x[FEATURES].fillna(0.0)

    return x


# -------------------- 方向モデル --------------------

def make_model():
    """特徴量の標準化と分類器を一体化して保存する。"""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=7,
                    min_samples_leaf=8,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_direction_model(feature_df, end_index, horizon):
    """
    end_indexより前のデータだけで学習する。
    ラベルはhorizon本先の終値が現在より高いか低いか。
    """
    if end_index <= 100:
        return None

    train_end = end_index - horizon
    if train_end < 80:
        return None

    close = feature_df["Close"].astype(float)
    future_return = close.shift(-horizon) / close - 1

    train = feature_df.iloc[:train_end].copy()
    labels = (future_return.iloc[:train_end] > 0).astype(int)

    valid = (
        train[FEATURES].notna().all(axis=1)
        & future_return.iloc[:train_end].notna()
    )
    X = train.loc[valid, FEATURES]
    y = labels.loc[valid]

    if len(X) < 80 or y.nunique() < 2:
        return None

    model = make_model()
    model.fit(X, y)
    return model


def predict_direction(model, row):
    """HIGH / LOWの二択を返す。"""
    if model is None:
        return "HIGH", 0.5, 0.5

    probabilities = model.predict_proba(
        row[FEATURES].to_frame().T
    )[0]
    classes = list(model.named_steps["classifier"].classes_)
    probs = dict(zip(classes, probabilities))

    p_low = float(probs.get(0, 0.5))
    p_high = float(probs.get(1, 0.5))
    direction = "HIGH" if p_high >= p_low else "LOW"

    return direction, p_high, p_low


def get_model_path(mode):
    return HOURLY_MODEL_FILE if mode == "hourly" else MINUTE_MODEL_FILE


def save_model(model, mode):
    """長期モデルと短期モデルを別々のファイルに保存する。"""
    if model is not None:
        joblib.dump(model, get_model_path(mode))


def load_model(mode):
    path = get_model_path(mode)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


# -------------------- 先着判定 --------------------

def judge_first_touch(entry, direction, future, target):
    """
    目標価格への先着判定。
    同じ足で上下両方に到達した場合は順序不明のため保守的にLOSS。
    """
    if future is None or future.empty:
        return "LOSS", "判定データ不足"

    upper = float(entry) + float(target)
    lower = float(entry) - float(target)

    for _, candle in future.iterrows():
        hit_up = float(candle["High"]) >= upper
        hit_down = float(candle["Low"]) <= lower

        if hit_up and hit_down:
            return "LOSS", "同一足で上下両方到達・順序不明"

        if hit_up:
            if direction == "HIGH":
                return "WIN", "上目標に先着"
            return "LOSS", "上目標が先着・LOW予測と逆"

        if hit_down:
            if direction == "LOW":
                return "WIN", "下目標に先着"
            return "LOSS", "下目標が先着・HIGH予測と逆"

    return "LOSS", "判定期間内に目標未到達"


# -------------------- 相場環境・原因分類 --------------------

def classify_market_regime(row):
    """予測時点で確認できるトレンドとボラティリティを分類する。"""
    trend = float(row.get("sma30_gap", 0.0))
    vol = float(row.get("volatility24", 0.0))

    if trend > 0.003:
        trend_name = "上昇トレンド"
    elif trend < -0.003:
        trend_name = "下降トレンド"
    else:
        trend_name = "横ばい"

    if vol >= 0.008:
        vol_name = "高ボラ"
    elif vol <= 0.002:
        vol_name = "低ボラ"
    else:
        vol_name = "中ボラ"

    return trend_name, vol_name


def explain_result(direction, result, reason, trend, volatility, p_high, p_low):
    """結果の原因をルールベースで説明する。原因は確定診断ではない。"""
    notes = []

    if result == "WIN":
        notes.append("予測方向の目標価格が先に到達")
    else:
        if "両方" in reason:
            notes.append("同一足内の値動きが大きく、先着順を確定できない")
        elif "未到達" in reason:
            notes.append("設定した判定期間内に目標値幅へ届かなかった")
        elif "逆" in reason:
            notes.append("予測と反対方向の目標価格が先に到達")
        else:
            notes.append(reason)

    if trend == "上昇トレンド" and direction == "LOW":
        notes.append("上昇トレンドに逆らうLOW予測")
    elif trend == "下降トレンド" and direction == "HIGH":
        notes.append("下降トレンドに逆らうHIGH予測")

    if volatility == "高ボラ":
        notes.append("高ボラ環境で目標・逆方向に到達しやすい可能性")
    elif volatility == "低ボラ" and result == "LOSS":
        notes.append("値動き不足により目標未到達となった可能性")

    confidence = max(p_high, p_low)
    if confidence < 0.55:
        notes.append("方向確率が拮抗しており、予測確度が低い")
    elif confidence >= 0.70 and result == "LOSS":
        notes.append("高い予測確率でも敗北。確率の校正を再検証する必要")

    return " / ".join(notes)


# -------------------- ウォークフォワード・バックテスト --------------------

def run_backtest(
    raw_df,
    mode,
    target_usd,
    stake_usd,
    hold_bars,
    retrain_every=24,
    min_train_rows=300,
):
    """
    時系列順に検証する。
    各時点で未来を学習に含めない。
    学習対象のラベルも、判定期間が完了した過去データだけを使用する。
    """
    if raw_df is None or len(raw_df) < min_train_rows + hold_bars + 70:
        return pd.DataFrame(), {}

    feat = build_features(raw_df)
    if feat is None:
        return pd.DataFrame(), {}

    records = []
    model = None
    last_fit = -10**9

    start = min_train_rows
    stop = len(feat) - hold_bars - 1

    progress = st.progress(0, text=f"{mode}モデルを時系列検証中...")

    total_steps = max(1, stop - start)

    for i in range(start, stop):
        # 学習は毎回ではなく、指定間隔で更新する。
        # train_direction_model内部で、未来ラベルが完成した行だけを使う。
        if model is None or i - last_fit >= retrain_every:
            model = train_direction_model(feat, i, hold_bars)
            last_fit = i

        if model is None:
            continue

        row = feat.iloc[i]
        direction, p_high, p_low = predict_direction(model, row)

        # 次の足の始値でエントリーする
        entry_i = i + 1
        entry_candle = feat.iloc[entry_i]
        entry_price = float(entry_candle["Open"])
        future = feat.iloc[entry_i : entry_i + hold_bars]

        result, reason = judge_first_touch(
            entry=entry_price,
            direction=direction,
            future=future,
            target=target_usd,
        )

        pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd
        trend, volatility = classify_market_regime(row)
        explanation = explain_result(
            direction, result, reason, trend, volatility, p_high, p_low
        )

        records.append(
            {
                "model": mode,
                "signal_time": str(feat.index[i]),
                "entry_time": str(feat.index[entry_i]),
                "entry_price": entry_price,
                "direction": direction,
                "p_high": p_high,
                "p_low": p_low,
                "confidence": max(p_high, p_low),
                "trend": trend,
                "volatility": volatility,
                "upper_target": entry_price + target_usd,
                "lower_target": entry_price - target_usd,
                "result": result,
                "reason": reason,
                "cause_analysis": explanation,
                "pnl_usd": pnl,
            }
        )

        if (i - start) % 10 == 0:
            progress.progress(
                min(1.0, (i - start) / total_steps),
                text=f"{mode}モデルを時系列検証中... {i-start}/{total_steps}",
            )

    progress.empty()

    result_df = pd.DataFrame(records)
    if result_df.empty:
        return result_df, {}

    # 方向正解率は終値方向を予測する別指標。
    # 先着判定の取引勝率とは異なる。
    close = feat["Close"]
    direction_labels = (close.shift(-hold_bars) > close).astype(int)

    actual_direction = []
    predicted_direction = []

    for _, r in result_df.iterrows():
        try:
            idx = feat.index.get_loc(pd.Timestamp(r["signal_time"]))
            actual = int(direction_labels.iloc[idx])
            if pd.isna(close.shift(-hold_bars).iloc[idx]):
                continue
            actual_direction.append(actual)
            predicted_direction.append(1 if r["direction"] == "HIGH" else 0)
        except Exception:
            continue

    metrics = {}
    if actual_direction:
        metrics["direction_accuracy"] = accuracy_score(
            actual_direction, predicted_direction
        ) * 100
        metrics["balanced_accuracy"] = balanced_accuracy_score(
            actual_direction, predicted_direction
        ) * 100

    # 履歴はモデルごとに分けて保存する。
    history_path = (
        DATA_DIR / f"{mode}_backtest_history.csv"
    )

    if history_path.exists():
        try:
            old = pd.read_csv(history_path)
            result_df = pd.concat([old, result_df], ignore_index=True)
            result_df = result_df.drop_duplicates(
                subset=["model", "entry_time"]
            )
            result_df = result_df.sort_values("entry_time")
        except Exception:
            pass

    result_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    # 互換用の統合履歴も保存
    if HISTORY_FILE.exists():
        try:
            old_all = pd.read_csv(HISTORY_FILE)
            all_history = pd.concat(
                [old_all, result_df], ignore_index=True
            )
            all_history = all_history.drop_duplicates(
                subset=["model", "entry_time"]
            )
        except Exception:
            all_history = result_df.copy()
    else:
        all_history = result_df.copy()

    all_history.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
    return result_df, metrics


# -------------------- 結果集計 --------------------

def max_losing_streak(results):
    longest = 0
    current = 0
    for result in results:
        if result == "LOSS":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize_by_column(df, col):
    if df.empty or col not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby(col, dropna=False)
        .agg(
            検証数=("result", "size"),
            勝数=("result", lambda s: int((s == "WIN").sum())),
            平均損益=("pnl_usd", "mean"),
            累積損益=("pnl_usd", "sum"),
        )
        .assign(
            勝率=lambda x: x["勝数"] / x["検証数"] * 100
        )
        .reset_index()
    )


def show_results(df, metrics=None):
    if df is None or df.empty:
        st.warning("バックテスト結果がありません。データ期間や取得状況を確認してください。")
        return

    metrics = metrics or {}
    wins = int((df["result"] == "WIN").sum())
    losses = int((df["result"] == "LOSS").sum())
    total = len(df)
    win_rate = wins / total * 100 if total else 0.0
    pnl = float(df["pnl_usd"].sum())
    avg_pnl = pnl / total if total else 0.0
    streak = max_losing_streak(df["result"].tolist())

    # 88%払戻し・負け100%損失の理論損益分岐点
    breakeven = 100 / (1 + PAYOUT_RATE)

    st.subheader("📊 総合バックテスト評価")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("検証数", f"{total:,}回")
    c2.metric("取引勝率", f"{win_rate:.2f}%", f"{wins}勝 {losses}敗")
    c3.metric("累積損益", f"${pnl:+,.2f}")
    c4.metric("最大連敗", f"{streak}回")

    c5, c6, c7 = st.columns(3)
    c5.metric("平均損益/回", f"${avg_pnl:+.2f}")
    c6.metric("理論損益分岐勝率", f"{breakeven:.2f}%")
    if "direction_accuracy" in metrics:
        c7.metric("終値方向正解率", f"{metrics['direction_accuracy']:.2f}%")
    else:
        c7.metric("終値方向正解率", "算出不可")

    if win_rate >= breakeven:
        st.success(
            "検証区間では、取引勝率が理論損益分岐点以上でした。"
            "別期間での再検証が必要です。"
        )
    else:
        st.warning(
            "検証区間では理論損益分岐点を下回っています。"
            "現状の条件で利益が出るとは評価できません。"
        )

    if "balanced_accuracy" in metrics:
        st.caption(
            f"方向予測のBalanced Accuracy: "
            f"{metrics['balanced_accuracy']:.2f}% "
            "（HIGH/LOWの偏りを考慮した方向評価）"
        )

    st.subheader("📈 累積損益")
    chart = df.copy()
    chart["entry_time"] = pd.to_datetime(
        chart["entry_time"], errors="coerce"
    )
    chart = chart.dropna(subset=["entry_time"]).sort_values("entry_time")
    chart["累積損益"] = chart["pnl_usd"].cumsum()
    st.line_chart(chart.set_index("entry_time")[["累積損益"]])

    st.subheader("🧭 HIGH / LOW別の勝率")
    direction_summary = summarize_by_column(df, "direction")
    if not direction_summary.empty:
        st.dataframe(direction_summary, use_container_width=True, hide_index=True)

    st.subheader("🔍 勝敗原因の分析")

    reason_summary = summarize_by_column(df, "reason")
    if not reason_summary.empty:
        st.markdown("**判定理由別**")
        st.dataframe(
            reason_summary.sort_values("検証数", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    for col, title in [
        ("trend", "トレンド別"),
        ("volatility", "ボラティリティ別"),
    ]:
        summary = summarize_by_column(df, col)
        if not summary.empty:
            st.markdown(f"**{title}**")
            st.dataframe(summary, use_container_width=True, hide_index=True)

    if "cause_analysis" in df.columns:
        st.markdown("**直近の具体的な反省記録**")
        st.dataframe(
            df[
                [
                    "entry_time", "direction", "confidence", "trend",
                    "volatility", "result", "reason", "cause_analysis",
                    "pnl_usd",
                ]
            ].sort_values("entry_time", ascending=False).head(100),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("📋 検証履歴")
    st.dataframe(
        df.sort_values("entry_time", ascending=False).head(200),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "📥 この結果をCSVでダウンロード",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"utaya_{df['model'].iloc[-1]}_backtest.csv",
        mime="text/csv",
    )


# -------------------- サイドバー --------------------

st.sidebar.header("⚙️ UTAYA AI設定")

model_mode = st.sidebar.radio(
    "分析モデル",
    options=["hourly", "minute"],
    format_func=lambda x: (
        "長期モデル（1時間足）" if x == "hourly"
        else "短期モデル（1分足）"
    ),
)

target_usd = st.sidebar.slider(
    "🎯 目標値幅（±USD）",
    min_value=10.0,
    max_value=500.0 if model_mode == "hourly" else 100.0,
    value=DEFAULT_TARGET_USD,
    step=5.0,
)

stake_usd = st.sidebar.number_input(
    "💵 1回あたりの仮想賭け金（USD）",
    min_value=1.0,
    max_value=10000.0,
    value=5.0,
    step=1.0,
)

hold_bars = st.sidebar.slider(
    "⏱️ 判定期間（選択中の足の本数）",
    min_value=1,
    max_value=48 if model_mode == "hourly" else 5,
    value=3 if model_mode == "hourly" else 1,
)

retrain_every = st.sidebar.slider(
    "学習モデルの更新間隔（本）",
    min_value=6,
    max_value=100,
    value=24,
)

st.sidebar.caption(
    "長期モデルの判定期間は時間足単位です。"
    "短期モデルの判定期間は1分足単位です。"
)

# -------------------- 市場監視 --------------------

st.subheader("⚡ BTC市場監視")

refresh_now = st.button("🔄 最新データを取得")

with st.spinner("市場データを取得しています..."):
    live_df = get_market_data(model_mode, refresh=refresh_now)

if live_df is None or len(live_df) < 100:
    st.error(
        "十分な市場データを取得できませんでした。"
        "ネット接続、Yahoo Financeの制限、CSVファイルを確認してください。"
    )
else:
    st.caption(
        f"データ期間: {live_df.index.min()} ～ {live_df.index.max()} | "
        f"件数: {len(live_df):,} | モデル: {model_mode}"
    )

    features = build_features(live_df)

    # 最後の足は形成中の可能性があるため、原則として直前の確定足を使う
    signal_i = len(features) - 2
    row = features.iloc[signal_i]

    model = load_model(model_mode)

    # 保存モデルがなければ、過去データから一度学習する
    if model is None and signal_i >= 300:
        with st.spinner("初回モデルを過去データから学習しています..."):
            model = train_direction_model(
                features, signal_i, hold_bars
            )
            if model is not None:
                save_model(model, model_mode)

    if model is not None:
        direction, p_high, p_low = predict_direction(model, row)
    else:
        direction, p_high, p_low = "HIGH", 0.5, 0.5
        st.warning(
            "学習データが不足しているため、まだ信頼できる予測を出せません。"
            "バックテストを先に実行してください。"
        )

    signal_price = float(row["Close"])
    trend, volatility = classify_market_regime(row)

    # 連敗停止は監視上の安全策。バックテスト自体の結果は変更しない。
    safety_stop = False
    if HISTORY_FILE.exists():
        try:
            hist = pd.read_csv(HISTORY_FILE)
            hist = hist[hist["model"] == model_mode]
            hist = hist.sort_values("entry_time")
            if not hist.empty:
                safety_stop = (
                    max_losing_streak(hist["result"].tolist()) >= 4
                )
        except Exception:
            pass

    if safety_stop:
        st.markdown(
            """
            <div class="main-banner-neutral">
                <h2>🛑 直近の連敗による安全停止</h2>
                <p>監視シグナルを一時停止しています。履歴と敗因を確認してください。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif direction == "HIGH":
        st.markdown(
            f"""
            <div class="main-banner-high">
                <h2>🚀 HIGH（上方向予測）</h2>
                <p>基準価格: ${signal_price:,.2f}</p>
                <p>上目標: ${signal_price + target_usd:,.2f}</p>
                <p>HIGH確率: {p_high:.1%} / LOW確率: {p_low:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="main-banner-low">
                <h2>📉 LOW（下方向予測）</h2>
                <p>基準価格: ${signal_price:,.2f}</p>
                <p>下目標: ${signal_price - target_usd:,.2f}</p>
                <p>HIGH確率: {p_high:.1%} / LOW確率: {p_low:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("基準価格", f"${signal_price:,.2f}")
    m2.metric("トレンド", trend)
    m3.metric("ボラティリティ", volatility)
    m4.metric("予測確度", f"{max(p_high, p_low):.1%}")

    st.caption(
        f"予測時刻: {features.index[signal_i]} | "
        "このシグナルは予測であり、将来の値動きを保証しません。"
    )

# -------------------- バックテスト --------------------

st.markdown("---")
st.subheader("🧪 過去データ学習・ウォークフォワード検証")

if model_mode == "hourly":
    st.info(
        "長期モデルは1時間足を使用します。取得できるデータ量は"
        "データ提供元の制限に左右されます。1年以上のデータが"
        "保存されていれば、それも統合して検証します。"
    )
else:
    st.info(
        "短期モデルは1分足専用です。一般に1分足の取得可能期間は短く、"
        "1年以上の1分足バックテストには別途取得・保存したデータが必要です。"
    )

if st.button(
    "🚀 過去データ学習・バックテスト・敗因分析を実行",
    use_container_width=True,
):
    with st.spinner("データを取得し、時系列順に検証しています..."):
        backtest_df = get_market_data(model_mode, refresh=True)

    if backtest_df is None:
        st.error(
            "データ取得に失敗しました。保存済みCSVが存在するか確認してください。"
        )
    else:
        minimum_rows = 400 if model_mode == "hourly" else 300

        if len(backtest_df) < minimum_rows:
            st.error(
                f"データ件数が不足しています（{len(backtest_df)}本）。"
                f"このモデルでは少なくとも{minimum_rows}本を目安としています。"
            )
        else:
            result_df, metrics = run_backtest(
                raw_df=backtest_df,
                mode=model_mode,
                target_usd=target_usd,
                stake_usd=stake_usd,
                hold_bars=hold_bars,
                retrain_every=retrain_every,
                min_train_rows=300,
            )

            if result_df.empty:
                st.error(
                    "検証結果を作成できませんでした。"
                    "データ量または学習対象の方向ラベルを確認してください。"
                )
            else:
                # 検証で使った最後の過去モデルを保存する。
                # 次の監視では、選択モデルだけをロードする。
                final_model = train_direction_model(
                    build_features(backtest_df),
                    len(backtest_df) - hold_bars - 1,
                    hold_bars,
                )
                if final_model is not None:
                    save_model(final_model, model_mode)

                st.success(
                    f"{model_mode}モデルの検証完了。"
                    f"検証件数: {len(result_df):,}件"
                )
                show_results(result_df, metrics)

# -------------------- 保存済み結果の表示 --------------------

st.markdown("---")
st.subheader("📚 保存済みの検証履歴")

saved_file = (
    DATA_DIR / f"{model_mode}_backtest_history.csv"
)

if saved_file.exists():
    try:
        saved_df = pd.read_csv(saved_file)
        if not saved_df.empty:
            show_results(saved_df)
        else:
            st.caption("このモデルの保存済み履歴はまだありません。")
    except Exception as exc:
        st.warning(f"履歴を読み込めませんでした: {exc}")
else:
    st.caption("このモデルの保存済み履歴はまだありません。")

# -------------------- 自動更新 --------------------

st.markdown("---")
st.subheader("🔁 自動監視の更新")

auto_refresh = st.checkbox(
    "30秒ごとに画面を自動更新する",
    value=False,
)

if auto_refresh:
    st.caption(
        "自動更新はブラウザで画面を開いている間の機能です。"
        "PCの電源断やアプリ停止中も稼働する常駐サービスではありません。"
    )
    time.sleep(30)
    st.rerun()

# -------------------- 学習リセット --------------------

st.markdown("---")
st.subheader("🧹 モデル・履歴の管理")

st.warning(
    "リセットすると、選択したモデルの保存済みモデルと履歴が削除されます。"
    "CSVを必要に応じて先にダウンロードしてください。"
)

confirm_reset = st.checkbox(
    "削除することを確認しました",
    key="confirm_reset",
)

if st.button("🗑️ 選択モデルの学習・履歴をリセット"):
    if not confirm_reset:
        st.error("先に確認チェックを入れてください。")
    else:
        paths = [
            get_model_path(model_mode),
            DATA_DIR / f"{model_mode}_backtest_history.csv",
        ]
        for path in paths:
            if path.exists():
                path.unlink()

        # 統合履歴から該当モデルだけを除く
        if HISTORY_FILE.exists():
            try:
                all_df = pd.read_csv(HISTORY_FILE)
                if "model" in all_df.columns:
                    all_df = all_df[all_df["model"] != model_mode]
                    all_df.to_csv(
                        HISTORY_FILE, index=False, encoding="utf-8-sig"
                    )
            except Exception:
                pass

        st.success(
            f"{model_mode}モデルの保存済みモデルと履歴を削除しました。"
        )
        st.rerun()

st.caption(
    "UTAYA AIは学習・分析用シミュレーターです。"
    "バックテスト結果は将来の成績を保証せず、"
    "データ取得元、価格差、手数料、約定遅延などによって実績は変わります。"
)
