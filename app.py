import streamlit as st
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
