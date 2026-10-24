import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# =========================================================
# UTAYA 成長型AI「子供」 (Adaptive Online Learning System)
# 思想: 予測 → 結果の観測 → 誤差からの即時学習（反省と成長）
# =========================================================

st.set_page_config(
    page_title="UTAYA AI Child - Growing System",
    page_icon="🌱",
    layout="wide",
)

SYMBOL = "BTC-USD"
PAYOUT_RATE = 0.88
DEFAULT_TARGET_USD = 50.0
DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(exist_ok=True)

MEMORY_FILE = DATA_DIR / "ai_child_memory.csv"

st.markdown(
    """
    <style>
    .child-banner {
        background: linear-gradient(135deg, #f6d365, #fda085);
        padding: 20px;
        border-radius: 15px;
        color: #2d3748;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌱 UTAYA 成長型AI「子供」の育成ダッシュボード")
st.caption(
    "オンライン学習エンジン搭載 | 間違いからの反省と自己修正のプロセスを可視化"
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


# -------------------- ② 特徴量生成 --------------------

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


# -------------------- ③ オンライン学習（AIの成長システム） --------------------

def run_growing_ai_simulation(df, target_usd, stake_usd, hold_bars):
    """
    AIが過去から現在へ時系列順に歩みを進めながら、
    「予測する ⇒ 結果を知る ⇒ 間違いから即座に学ぶ（部分適応）」を繰り返すシミュレーション。
    """
    if df is None or len(df) < 300:
        return pd.DataFrame()

    features_df = build_feature_table(len(df), df)
    
    # 逐次学習が可能な確率的勾配降下法（SGDClassifier）を使用
    # これにより、新しい経験（間違え）をするたびにモデルの神経回路が少しずつ書き換わる
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
    scaler = StandardScaler()

    # 初期教育（最初の50本でベースを作る）
    init_end = int(len(df) * 0.2)
    X_init, y_init = [], []
    
    for j in range(30, init_end):
        feat = features_df.iloc[j]
        if j + hold_bars < len(df) and not feat.isna().any():
            curr = float(df["Close"].iloc[j])
            fut = float(df["Close"].iloc[j + hold_bars])
            label = 1 if fut > curr else 0
            X_init.append(feat.values)
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
        
        # 予測確率の取得 (0: LOW, 1: HIGH)
        try:
            prob = model.predict_proba(X_pred)[0]
            p_low, p_high = prob[0], prob[1]
        except Exception:
            p_low, p_high = 0.5, 0.5

        direction = "HIGH" if p_high >= p_low else "LOW"
        confidence = max(p_high, p_low)

        # 実績の判定
        entry_index = i + 1
        entry_candle = df.iloc[entry_index]
        entry_price = float(entry_candle["Open"])
        future = df.iloc[entry_index : entry_index + hold_bars]

        result, reason = judge_first_touch(entry_price, direction, future, target_usd)
        pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

        # 🎯 【超重要：AIの反省と学習（成長プロセス）】
        # 実際にどうだったかの正解ラベルを作る
        actual_fut = float(df["Close"].iloc[entry_index + hold_bars - 1]) if entry_index + hold_bars - 1 < len(df) else entry_price
        actual_label = 1 if actual_fut > entry_price else 0

        # 間違えたとき（あるいは正解したとき）、その場でモデルを1ステップだけ更新（反省して賢くなる）
        model.partial_fit(X_pred, np.array([actual_label]))

        records.append({
            "time": str(df.index[i]),
            "direction": direction,
            "confidence": confidence,
            "result": result,
            "reason": reason,
            "pnl_usd": pnl,
        })

    progress_bar.empty()
    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df.to_csv(MEMORY_FILE, index=False, encoding="utf-8-sig")

    return res_df


# -------------------- ④ ダッシュボード描画 --------------------

def show_child_growth(res_df):
    if res_df.empty:
        st.warning("学習データがありません。")
        return

    total = len(res_df)
    wins = int((res_df["result"] == "WIN").sum())
    losses = int((res_df["result"] == "LOSS").sum())
    win_rate = wins / total * 100 if total else 0.0
    total_pnl = float(res_df["pnl_usd"].sum())

    st.markdown(
        f"""
        <div class="child-banner">
            <h2>🧸 AIの成長レポート（経験した総レッスン数: {total}回）</h2>
            <p>現在の累積勝率: <strong>{win_rate:.2f}%</strong> ｜ 獲得仮想損益: <strong>${total_pnl:+.2f}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📈 AIの成長曲線（累積損益の推移）")
    chart_df = res_df[["time", "pnl_usd"]].copy()
    chart_df["time"] = pd.to_datetime(chart_df["time"])
    chart_df["累積損益"] = chart_df["pnl_usd"].cumsum()
    chart_df = chart_df.set_index("time")
    st.line_chart(chart_df[["累積損益"]])

    st.subheader("📝 AIの直近の反省ノート（学習ログ）")
    st.dataframe(
        res_df.sort_values("time", ascending=False).head(30),
        use_container_width=True,
        hide_index=True,
    )


# -------------------- ⑤ サイドバー --------------------

st.sidebar.header("🌱 育成パラメータ")
target_usd = st.sidebar.slider("🎯 目標値幅 (±USD)", 10.0, 100.0, DEFAULT_TARGET_USD, 5.0)
stake_usd = st.sidebar.number_input("💵 仮想賭け金 ($)", 1.0, 10000.0, 5.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (分足の本数)", 1, 5, 1)
period = st.sidebar.selectbox("📅 育成データ期間", ["1d", "2d", "5d", "7d"], index=3)


# -------------------- ⑥ メイン実行 --------------------

st.subheader("⚡ 成長型AIの育成セッション")
st.write(
    "ボタンを押すと、AIが過去の市場の波にもまれながら、"
    "「予測 ➔ 間違いの発見 ➔ 即座の修正（オンライン学習）」を体験し、成長していく過程をシミュレートします。"
)

if st.button("🚀 AI（子供）を学校に通わせる（育成を開始）", use_container_width=True):
    with st.spinner("AIが市場の授業を受けて、失敗から学んでいます..."):
        full_df = get_web_data(period)
        if full_df is None or len(full_df) < 300:
            st.error("データが不足しています。期間を延ばしてください。")
        else:
            growth_df = run_growing_ai_simulation(full_df, target_usd, stake_usd, hold_bars)
            show_child_growth(growth_df)
elif MEMORY_FILE.exists():
    try:
        saved_memory = pd.read_csv(MEMORY_FILE)
        if not saved_memory.empty:
            show_child_growth(saved_memory)
    except Exception:
        pass
