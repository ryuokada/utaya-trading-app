import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# =========================================================
# UTAYA 仮想平行宇宙・未来シナリオ分析AI (二択型 Pro Turbo)
# 仕様: 必ず HIGH / LOW のどちらかを選択し検証・学習する
# =========================================================

st.set_page_config(
    page_title="UTAYA Multiverse Scenario AI - Binary",
    page_icon="🌌",
    layout="wide",
)

SYMBOL = "BTC-USD"
PAYOUT_RATE = 0.88
DEFAULT_TARGET_USD = 50.0
DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(exist_ok=True)

SCENARIO_HISTORY = DATA_DIR / "multiverse_binary_history.csv"

# -------------------- CSS --------------------

st.markdown(
    """
    <style>
    .multiverse-banner {
        background: linear-gradient(135deg, #1f4068, #162447);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #e43f5a;
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

st.title("🌌 UTAYA 仮想平行宇宙・未来シナリオ分析AI (二択型)")
st.caption(
    "常時HIGH/LOW二択強制エンジン | 確率比較による決定論的選択 | 損益・勝率徹底検証"
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


# -------------------- ② 特徴量一括計算エンジン --------------------

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


def make_label(df, i, horizon=1):
    """将来の終値変化をHIGH／LOWの二択で分類する（同値はLOWに割り当て）"""
    if i + horizon >= len(df):
        return None

    current_price = float(df["Close"].iloc[i])
    future_price = float(df["Close"].iloc[i + horizon])

    if current_price <= 0:
        return None

    if future_price > current_price:
        return "HIGH"
    return "LOW"


# -------------------- ③ 先着判定エンジン --------------------

def judge_first_touch(entry, direction, future, target=DEFAULT_TARGET_USD):
    if direction not in ("HIGH", "LOW"):
        return "LOSS", "無効な方向"

    if future is None or future.empty:
        return "LOSS", "判定データ不足"

    upper = float(entry) + float(target)
    lower = float(entry) - float(target)

    for _, candle in future.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])

        hit_up = high >= upper
        hit_down = low <= lower

        if hit_up and hit_down:
            return "LOSS", "同一足で上下両方到達（順序不明・保守的処理）"

        if hit_up:
            if direction == "HIGH":
                return "WIN", "上目標先着 (合致)"
            return "LOSS", "上目標先着 (不一致)"

        if hit_down:
            if direction == "LOW":
                return "WIN", "下目標先着 (合致)"
            return "LOSS", "下目標先着 (不一致)"

    return "LOSS", "判定期間内未到達 (タイムアウト)"


# -------------------- ④ 二択強制ウォークフォワード検証エンジン --------------------

def run_multiverse_backtest(
    df,
    target_usd,
    stake_usd,
    hold_bars,
    retrain_every=120,
    min_train_samples=200,
):
    if df is None or len(df) < 300:
        return pd.DataFrame()

    features_df = build_feature_table(len(df), df)
    records = []
    model = None
    last_train_i = -10**9

    label_cache = {}
    for j in range(30, len(df)):
        label_cache[j] = make_label(df, j, horizon=hold_bars)

    start_i = max(50, int(len(df) * 0.25))
    progress_bar = st.progress(0)
    total_steps = (len(df) - hold_bars - 1) - start_i

    for idx, i in enumerate(range(start_i, len(df) - hold_bars - 1)):
        if total_steps > 0 and idx % 50 == 0:
            progress_bar.progress(min(1.0, idx / total_steps))

        train_end = i - hold_bars - 1
        if train_end < 30:
            continue

        if model is None or (i - last_train_i >= retrain_every):
            X_train, y_train = [], []
            for j in range(30, train_end + 1):
                feat_row = features_df.iloc[j]
                lbl = label_cache.get(j)
                if not feat_row.isna().any() and lbl is not None:
                    X_train.append(feat_row.values)
                    y_train.append(lbl)

            if len(X_train) >= min_train_samples:
                X_train = np.array(X_train)
                y_train = np.array(y_train)

                classes_present = np.unique(y_train)
                if len(classes_present) >= 2:
                    try:
                        weights = compute_class_weight("balanced", classes=classes_present, y=y_train)
                        class_weight_dict = dict(zip(classes_present, weights))
                    except Exception:
                        class_weight_dict = "balanced"

                    model = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=5,
                        class_weight=class_weight_dict,
                        random_state=42,
                        n_jobs=-1,
                    )
                    model.fit(X_train, y_train)
                    last_train_i = i

        feat_row = features_df.iloc[i]
        if feat_row.isna().any() or model is None:
            continue

        X_pred = np.array([feat_row.values])
        probs = model.predict_proba(X_pred)[0]
        classes = model.classes_

        prob_dict = dict(zip(classes, probs))
        p_high = float(prob_dict.get("HIGH", 0.0))
        p_low = float(prob_dict.get("LOW", 0.0))

        # 常に必ず HIGH か LOW の二択を強制決定
        direction = "HIGH" if p_high >= p_low else "LOW"
        max_prob = max(p_high, p_low)

        entry_index = i + 1
        entry_candle = df.iloc[entry_index]
        entry_price = float(entry_candle["Open"])
        future = df.iloc[entry_index : entry_index + hold_bars]

        result, reason = judge_first_touch(
            entry=entry_price,
            direction=direction,
            future=future,
            target=target_usd,
        )
        pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

        records.append({
            "signal_time": str(df.index[i]),
            "entry_time": str(df.index[entry_index]),
            "entry_price": entry_price,
            "direction": direction,
            "confidence": max_prob,
            "prob_high": p_high,
            "prob_low": p_low,
            "result": result,
            "reason": reason,
            "pnl_usd": pnl,
        })

    progress_bar.empty()
    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df.to_csv(SCENARIO_HISTORY, index=False, encoding="utf-8-sig")

    return res_df


# -------------------- ⑤ ダッシュボード描画 --------------------

def display_dashboard(res_df):
    if res_df.empty:
        st.warning("検証結果がありません。")
        return

    total = len(res_df)
    wins = int((res_df["result"] == "WIN").sum())
    losses = int((res_df["result"] == "LOSS").sum())
    win_rate = wins / total * 100 if total else 0.0
    total_pnl = float(res_df["pnl_usd"].sum())

    st.subheader("📊 二択型シナリオ検証・パフォーマンス評価")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総取引件数", f"{total}")
    c2.metric("勝率", f"{win_rate:.2f}%", f"{wins}勝 {losses}敗")
    c3.metric("仮想損益合計", f"${total_pnl:+.2f}")
    c4.metric("損益分岐比較", f"理論値: 53.19%")

    if win_rate >= 53.19:
        st.success(f"🌟 実測勝率 ({win_rate:.2f}%) は理論損益分岐点 (53.19%) を上回っています。")
    else:
        st.warning(f"⚠️ 実測勝率 ({win_rate:.2f}%) は理論損益分岐点 (53.19%) を下回っています。")

    st.subheader("🧭 選択方向の偏り診断 (HIGH / LOW)")
    pred_counts = res_df["direction"].value_counts().reindex(["HIGH", "LOW"], fill_value=0)
    st.bar_chart(pred_counts)

    st.subheader("📊 HIGH／LOW別の実測勝率・成績")
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

    st.subheader("📈 仮想平行宇宙の累積損益推移")
    chart_df = res_df[["entry_time", "pnl_usd"]].copy()
    chart_df["entry_time"] = pd.to_datetime(chart_df["entry_time"])
    chart_df["累積損益"] = chart_df["pnl_usd"].cumsum()
    chart_df = chart_df.set_index("entry_time")
    st.line_chart(chart_df[["累積損益"]])

    st.subheader("📋 予測・検証ログ")
    st.dataframe(
        res_df.sort_values("entry_time", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )


# -------------------- ⑥ サイドバー設定 --------------------

st.sidebar.header("🌌 マルチバース設定 (Binary)")

target_usd = st.sidebar.slider(
    "🎯 目標値幅 (±USD)",
    min_value=10.0,
    max_value=100.0,
    value=DEFAULT_TARGET_USD,
    step=5.0,
)

stake_usd = st.sidebar.number_input(
    "💵 仮想賭け金 ($)",
    min_value=1.0,
    max_value=10000.0,
    value=5.0,
    step=1.0,
)

hold_bars = st.sidebar.slider(
    "⏱️ 判定期間 (分足の本数)",
    min_value=1,
    max_value=5,
    value=1,
)

period = st.sidebar.selectbox(
    "📅 分析データ期間",
    options=["1d", "2d", "5d", "7d"],
    index=3,
)


# -------------------- ⑦ リアルタイム・二択予測監視 --------------------

st.subheader("⚡ 現在の市場状態と二択予測")

with st.spinner("市場データを解析し、MLモデルで二択予測中..."):
    live_df = get_web_data("1d")

if live_df is not None and len(live_df) >= 50:
    signal_i = len(live_df) - 2
    features_table = build_feature_table(len(live_df), live_df)
    feat_row = features_table.iloc[signal_i]
    
    if not feat_row.isna().any():
        X_live, y_live = [], []
        for j in range(30, signal_i - 1):
            f_row = features_table.iloc[j]
            l_val = make_label(live_df, j, horizon=hold_bars)
            if not f_row.isna().any() and l_val is not None:
                X_live.append(f_row.values)
                y_live.append(l_val)
        
        if len(X_live) > 50:
            clf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42, n_jobs=-1)
            clf.fit(np.array(X_live), np.array(y_live))
            probs = clf.predict_proba(np.array([feat_row.values]))[0]
            classes = clf.classes_
            p_dict = dict(zip(classes, probs))
        else:
            p_dict = {"HIGH": 0.5, "LOW": 0.5}

        p_high = float(p_dict.get("HIGH", 0.0))
        p_low = float(p_dict.get("LOW", 0.0))
        live_direction = "HIGH" if p_high >= p_low else "LOW"

        signal_price = float(live_df["Close"].iloc[signal_i])
        signal_time = live_df.index[signal_i]

        st.caption(f"観測基準時刻: {signal_time} | 現在基準価格: ${signal_price:,.2f}")

        # 二択の強制結果表示
        if live_direction == "HIGH":
            st.markdown(
                f"""
                <div class="scenario-card-high">
                    <h2>🚀 UTAYA 予測判定: HIGH</h2>
                    <p>HIGH予測確率：<strong>{p_high:.1%}</strong> ｜ LOW予測確率：{p_low:.1%}</p>
                    <p>上方向目標価格: ${signal_price + target_usd:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="scenario-card-low">
                    <h2>📉 UTAYA 予測判定: LOW</h2>
                    <p>LOW予測確率：<strong>{p_low:.1%}</strong> ｜ HIGH予測確率：{p_high:.1%}</p>
                    <p>下方向目標価格: ${signal_price - target_usd:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("特徴量計算用のデータが不足しています。")
else:
    st.warning("市場データを取得できませんでした。")


# -------------------- ⑧ バックテスト実行ボタン --------------------

st.markdown("---")
st.subheader("🚀 二択型ウォークフォワード検証の実行")

if st.button("▶️ 機械学習バックテストを開始 (Binary)", use_container_width=True):
    with st.spinner("特徴量一括計算および二択型時系列検証を実行中..."):
        full_df = get_web_data(period)

        if full_df is None or len(full_df) < 300:
            st.error("データ件数が不足しています（最低300本必要です）。期間を長くしてください。")
        else:
            res_df = run_multiverse_backtest(
                df=full_df,
                target_usd=target_usd,
                stake_usd=stake_usd,
                hold_bars=hold_bars,
            )
            display_dashboard(res_df)
elif SCENARIO_HISTORY.exists():
    try:
        saved_df = pd.read_csv(SCENARIO_HISTORY)
        if not saved_df.empty:
            display_dashboard(saved_df)
    except Exception:
        pass
