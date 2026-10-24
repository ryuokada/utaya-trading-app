import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# =========================================================
# UTAYA 仮想平行宇宙・未来シナリオ分析AI (天才育成完全版)
# 統合仕様: 高速特徴量 + 二択強制 + オンライン自己学習(反省ループ)
# =========================================================

st.set_page_config(
    page_title="UTAYA Multiverse Scenario AI - Genius Edition",
    page_icon="🌌",
    layout="wide",
)

SYMBOL = "BTC-USD"
PAYOUT_RATE = 0.88
DEFAULT_TARGET_USD = 50.0
DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(exist_ok=True)

GENIUS_HISTORY = DATA_DIR / "genius_ai_history.csv"

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

st.title("🌌 UTAYA AI 天才育成統合システム")
st.caption(
    "高速一括演算エンジン | 二択強制決定 | オンライン連続学習 (1%のひらめきと99%の努力)"
)


# -------------------- ① データ取得 --------------------

@st.cache_data(ttl=60, show_spinner=False)
def get_web_data(period="7d"):
    try:
        df = yf.download(
            SYMBOL,
            period=period,
            interval="1m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required):
            return None

        df = df[required].copy()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        df = df[
            (df["Open"] > 0)
            & (df["High"] > 0)
            & (df["Low"] > 0)
            & (df["Close"] > 0)
            & (df["High"] >= df["Low"])
        ]

        return df if not df.empty else None

    except Exception:
        return None


# -------------------- ② 特徴量一括計算エンジン (高速化) --------------------

@st.cache_data(ttl=60, show_spinner=False)
def build_feature_table(df_len, df):
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


def judge_first_touch(entry, direction, future, target=DEFAULT_TARGET_USD):
    if direction not in ("HIGH", "LOW") or future is None or future.empty:
        return "LOSS", "判定データ不足"

    upper = float(entry) + float(target)
    lower = float(entry) - float(target)

    for _, candle in future.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])

        hit_up = high >= upper
        hit_down = low <= lower

        if hit_up and hit_down:
            return "LOSS", "同一足で上下両方到達"
        if hit_up:
            return ("WIN", "上目標先着") if direction == "HIGH" else ("LOSS", "逆方向")
        if hit_down:
            return ("WIN", "下目標先着") if direction == "LOW" else ("LOSS", "逆方向")

    return "LOSS", "未到達タイムアウト"


# -------------------- ③ 天才育成・オンライン学習エンジン --------------------

def run_genius_simulation(df, target_usd, stake_usd, hold_bars):
    if df is None or len(df) < 300:
        return pd.DataFrame()

    features_df = build_feature_table(len(df), df)
    
    # 逐次学習モデル (SGDClassifier) とスケーラー
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
    scaler = StandardScaler()

    # 初期基礎教育
    init_end = int(len(df) * 0.2)
    X_init, y_init = [], []
    
    for j in range(30, init_end):
        feat_row = features_df.iloc[j]
        if j + hold_bars < len(df) and not feat_row.isna().any():
            curr = float(df["Close"].iloc[j])
            fut = float(df["Close"].iloc[j + hold_bars])
            label = 1 if fut > curr else 0
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

        # 🎯 二択の強制決定
        direction = "HIGH" if p_high >= p_low else "LOW"
        confidence = max(p_high, p_low)

        # 実績の判定
        entry_index = i + 1
        entry_candle = df.iloc[entry_index]
        entry_price = float(entry_candle["Open"])
        future = df.iloc[entry_index : entry_index + hold_bars]

        result, reason = judge_first_touch(entry_price, direction, future, target_usd)
        pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

        # 🧠 間違いからの即時反省・自己学習 (オンライン学習)
        actual_fut = float(df["Close"].iloc[entry_index + hold_bars - 1]) if entry_index + hold_bars - 1 < len(df) else entry_price
        actual_label = 1 if actual_fut > entry_price else 0
        model.partial_fit(X_pred, np.array([actual_label]))

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
    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df.to_csv(GENIUS_HISTORY, index=False, encoding="utf-8-sig")

    return res_df


# -------------------- ④ ダッシュボード描画 --------------------

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
        st.success(f"🌟 現在の勝率 ({win_rate:.2f}%) は損益分岐点 (53.19%) を超えています！天才への第一歩です。")
    else:
        st.warning(f"⚠️ 現在の勝率 ({win_rate:.2f}%) は損益分岐点を下回っています。さらなる反省とデータ学習が必要です。")

    st.subheader("🧭 選択方向の偏り診断 (HIGH / LOW)")
    pred_counts = res_df["direction"].value_counts().reindex(["HIGH", "LOW"], fill_value=0)
    st.bar_chart(pred_counts)

    st.subheader("📊 HIGH／LOW別の勝率・成績")
    dir_stats = []
    for d in ["HIGH", "LOW"]:
        sub = res_df[res_df["direction"] == d]
        sub_total = len(sub)
        sub_wins = int((sub["result"] == "WIN").sum())
        sub_rate = sub_wins / sub_total * 100 if sub_total else 0.0
        sub_pnl = float(sub["pnl_usd"].sum())
        dir_stats.append({
            "方向": d,
            "取引数": sub_total,
            "勝率": f"{sub_rate:.2f}%",
            "損益": f"${sub_pnl:+.2f}"
        })
    st.dataframe(pd.DataFrame(dir_stats), use_container_width=True, hide_index=True)

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


# -------------------- ⑤ サイドバー設定 --------------------

st.sidebar.header("🌌 天才育成パラメータ")
target_usd = st.sidebar.slider("🎯 目標値幅 (±USD)", 10.0, 100.0, DEFAULT_TARGET_USD, 5.0)
stake_usd = st.sidebar.number_input("💵 仮想賭け金 ($)", 1.0, 10000.0, 5.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (分足の本数)", 1, 5, 1)
period = st.sidebar.selectbox("📅 学習データ期間", ["1d", "2d", "5d", "7d"], index=3)


# -------------------- ⑥ リアルタイム・二択予測監視 --------------------

st.subheader("⚡ 現在の市場状態と天才AIの二択予測")

with st.spinner("市場データを解析し、AIがリアルタイム予測中..."):
    live_df = get_web_data("1d")

if live_df is not None and len(live_df) >= 50:
    signal_i = len(live_df) - 2
    features_table = build_feature_table(len(live_df), live_df)
    feat_row = features_table.iloc[signal_i]
    
    if not feat_row.isna().any():
        X_live, y_live = [], []
        for j in range(30, signal_i - 1):
            f_row = features_table.iloc[j]
            curr = float(live_df["Close"].iloc[j])
            fut = float(live_df["Close"].iloc[j + hold_bars]) if j + hold_bars < len(live_df) else curr
            l_val = 1 if fut > curr else 0
            if not f_row.isna().any():
                X_live.append(f_row.values)
                y_live.append(l_val)
        
        if len(X_live) > 50:
            clf = SGDClassifier(loss="log_loss", random_state=42)
            scaler_live = StandardScaler()
            X_live_sc = scaler_live.fit_transform(np.array(X_live))
            clf.fit(X_live_sc, np.array(y_live))
            
            feat_sc = scaler_live.transform(np.array([feat_row.values]))
            probs = clf.predict_proba(feat_sc)[0]
            p_low, p_high = probs[0], probs[1]
        else:
            p_low, p_high = 0.5, 0.5

        live_direction = "HIGH" if p_high >= p_low else "LOW"
        signal_price = float(live_df["Close"].iloc[signal_i])
        signal_time = live_df.index[signal_i]

        st.caption(f"観測基準時刻: {signal_time} | 現在基準価格: ${signal_price:,.2f}")

        if live_direction == "HIGH":
            st.markdown(
                f"""
                <div class="scenario-card-high">
                    <h2>🚀 天才AI 予測判定: HIGH</h2>
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
                    <h2>📉 天才AI 予測判定: LOW</h2>
                    <p>LOW確率：<strong>{p_low:.1%}</strong> ｜ HIGH確率：{p_high:.1%}</p>
                    <p>下方向目標価格: ${signal_price - target_usd:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("特徴量計算用のデータが不足しています。")
else:
    st.warning("市場データを取得できませんでした。")


# -------------------- ⑦ バックテスト実行ボタン --------------------

st.markdown("---")
st.subheader("🚀 天才育成シミュレーション（一括検証）の実行")

if st.button("▶️ 天才育成トレーニングを開始", use_container_width=True):
    with st.spinner("特徴量一括計算およびオンライン自己学習トレーニングを実行中..."):
        full_df = get_web_data(period)

        if full_df is None or len(full_df) < 300:
            st.error("データ件数が不足しています（最低300本必要です）。期間を長くしてください。")
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
