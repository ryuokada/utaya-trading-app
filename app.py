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
# UTAYA 潜在意識×科学的検証統合型・自律成長AIシステム (完全版)
# 理念: 失敗を恐れず全データを資産化し、客観的検証で改善を証明する
# =========================================================

st.set_page_config(
    page_title="UTAYA Murphy-Scientific AI Analyst",
    page_icon="🌌",
    layout="wide",
)

SYMBOL = "BTC-USD"
PAYOUT_RATE = 0.88  # 配当88% -> 損益分岐勝率 約53.19%
DEFAULT_TARGET_USD = 50.0
DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(exist_ok=True)

# ファイルパス定義 (長期・短期分離)
MARKET_HISTORY_1M = DATA_DIR / "btc_1m_history.csv"
MARKET_HISTORY_1H = DATA_DIR / "btc_1h_history.csv"
VIRTUAL_TRADES_FILE = DATA_DIR / "virtual_trades_history.csv"
ANALYSIS_LOG_FILE = DATA_DIR / "deep_analysis_log.csv"

MODEL_1M_FILE = DATA_DIR / "model_1m.joblib"
SCALER_1M_FILE = DATA_DIR / "scaler_1m.joblib"
MODEL_1H_FILE = DATA_DIR / "model_1h.joblib"
SCALER_1H_FILE = DATA_DIR / "scaler_1h.joblib"

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
    .scenario-card-hold {
        background: rgba(108, 117, 125, 0.1);
        border-left: 5px solid #6c757d;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌌 UTAYA 潜在意識×科学的検証 自律成長AIシステム")
st.caption(
    "目標勝率53.19%突破を目指す無制限検証 | 実取引と見送り予測の完全分離 | 全件永久保存"
)


# -------------------- ① 市場環境・外部要因分析 --------------------

def analyze_market_context(df):
    """市場環境（トレンド、ボラティリティ、偏りリスク）を客観評価する"""
    if df is None or len(df) < 30:
        return "データ不足による不確実相場", 0.0, "NORMAL"
    
    close = df["Close"].astype(float)
    sma_short = close.rolling(10).mean().iloc[-1]
    sma_long = close.rolling(30).mean().iloc[-1]
    volatility = close.pct_change().rolling(14).std().iloc[-1]
    
    env_desc = "レンジ・通常変動"
    if sma_short > sma_long * 1.002:
        env_desc = "短期上昇トレンド"
    elif sma_short < sma_long * 0.998:
        env_desc = "短期下落トレンド"
        
    risk_state = "NORMAL"
    if volatility > 0.003:
        env_desc += "（高ボラティリティ・急変警戒）"
        risk_state = "HIGH_VOLATILITY"
        
    return env_desc, float(volatility), risk_state


# -------------------- ② 永続データ保存・取得エンジン (全件保持) --------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_data(interval="1m"):
    file_path = MARKET_HISTORY_1M if interval == "1m" else MARKET_HISTORY_1H
    period_val = "7d" if interval == "1m" else "730d"
    
    try:
        df_new = yf.download(
            SYMBOL,
            period=period_val,
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
                df_new = df_new[(df_new["Close"] > 0) & (df_new["High"] >= df_new["Low"])]
                df_new = df_new[~df_new.index.duplicated(keep="last")].sort_index()

                if file_path.exists():
                    try:
                        df_old = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        df_combined = pd.concat([df_old, df_new])
                        df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
                        # 容量制限なく永久保存
                        df_combined.to_csv(file_path, encoding="utf-8-sig")
                    except Exception:
                        df_new.to_csv(file_path, encoding="utf-8-sig")
                else:
                    df_new.to_csv(file_path, encoding="utf-8-sig")

        if file_path.exists():
            df_stored = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df_stored.sort_index()
        return df_new
    except Exception:
        if file_path.exists():
            return pd.read_csv(file_path, index_col=0, parse_dates=True).sort_index()
        return None


# -------------------- ③ 特徴量と先着判定エンジン --------------------

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


def evaluate_target_touch(entry, direction, future, target=DEFAULT_TARGET_USD):
    """
    ±50$目標値幅への先着順判定（勝敗評価用）
    """
    if direction not in ("HIGH", "LOW") or future is None or future.empty:
        return 0, "LOSS", "判定データ不足", "DATA_ERROR"

    upper = float(entry) + float(target)
    lower = float(entry) - float(target)

    for idx, candle in future.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])

        hit_up = high >= upper
        hit_down = low <= lower

        if hit_up and hit_down:
            return 0, "LOSS", "同一足で上下両方到達（高ボラティリティ・ノイズ交差）", "VOLATILITY_NOISE"
        if hit_up:
            if direction == "HIGH":
                return 1, "WIN", "上目標先着：トレンド順行による的中", "TREND_SUCCESS"
            else:
                return 0, "LOSS", "逆方向50$到達：下落トレンドへの逆行失敗", "REVERSE_FAILURE_HIGH"
        if hit_down:
            if direction == "LOW":
                return 1, "WIN", "下目標先着：下落予測の的中", "TREND_SUCCESS"
            else:
                return 0, "LOSS", "逆方向50$到達：上昇トレンドへの逆行失敗", "REVERSE_FAILURE_LOW"

    return 0, "LOSS", "判定期間内未到達：レンジ相場・モメンタム減衰", "TIMEOUT_RANGE"


# -------------------- ④ モデル永続化管理 --------------------

def load_or_init_model(model_type="1m"):
    m_file = MODEL_1M_FILE if model_type == "1m" else MODEL_1H_FILE
    s_file = SCALER_1M_FILE if model_type == "1m" else SCALER_1H_FILE

    if m_file.exists() and s_file.exists():
        try:
            model = joblib.load(m_file)
            scaler = joblib.load(s_file)
            return model, scaler
        except Exception:
            pass
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
    scaler = StandardScaler()
    return model, scaler

def save_model_state(model, scaler, model_type="1m"):
    m_file = MODEL_1M_FILE if model_type == "1m" else MODEL_1H_FILE
    s_file = SCALER_1M_FILE if model_type == "1m" else SCALER_1H_FILE
    joblib.dump(model, m_file)
    joblib.dump(scaler, s_file)


# -------------------- ⑤ 公平なウォークフォワード検証エンジン --------------------

def run_walkforward_simulation(df, target_usd, stake_usd, hold_bars, model_type="1m"):
    if df is None or len(df) < 300:
        return pd.DataFrame()

    features_df = build_feature_table(df)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
    scaler = StandardScaler()

    init_end = int(len(df) * 0.25)
    X_init, y_init = [], []
    
    # 教師ラベル：次足の終値が上昇=1, 下落=0
    for j in range(30, init_end):
        feat_row = features_df.iloc[j]
        if j + 1 < len(df) and not feat_row.isna().any():
            curr_c = float(df["Close"].iloc[j])
            next_c = float(df["Close"].iloc[j + 1])
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

        confidence = max(p_high, p_low)
        
        if abs(p_high - p_low) < 0.04:
            direction = "HOLD"
        else:
            direction = "HIGH" if p_high >= p_low else "LOW"

        entry_index = i + 1
        entry_price = float(df["Open"].iloc[entry_index])
        future = df.iloc[entry_index : entry_index + hold_bars]

        eval_dir = direction if direction != "HOLD" else ("HIGH" if p_high >= p_low else "LOW")
        win_flag, result, reason, category = evaluate_target_touch(entry_price, eval_dir, future, target_usd)
        
        pnl = 0.0
        if direction != "HOLD":
            pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

        # オンライン学習更新（次足の方向を教師信号とする）
        actual_next_c = float(df["Close"].iloc[entry_index])
        actual_curr_c = float(df["Close"].iloc[entry_index - 1])
        direction_label = 1 if actual_next_c > actual_curr_c else 0
        
        try:
            model.partial_fit(X_pred, np.array([direction_label]))
        except ValueError:
            model.partial_fit(X_pred, np.array([direction_label]), classes=np.array([0, 1]))

        records.append({
            "time": str(df.index[i]),
            "direction": direction,
            "confidence": confidence,
            "p_high": p_high,
            "p_low": p_low,
            "result": result,
            "reason": reason,
            "category": category,
            "pnl_usd": pnl,
        })

    progress_bar.empty()
    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")

    return res_df


# -------------------- ⑥ ダッシュボード描画 (実取引と見送り予測の完全分離表示) --------------------

def show_analyst_dashboard(res_df):
    if res_df.empty:
        st.warning("検証データがありません。")
        return

    # 実取引（Executed）と見送り（HOLD）の分離
    executed_df = res_df[res_df["direction"] != "HOLD"].copy()
    hold_df = res_df[res_df["direction"] == "HOLD"].copy()

    total_exec = len(executed_df)
    holds = len(hold_df)
    
    wins = int((executed_df["result"] == "WIN").sum())
    losses = int((executed_df["result"] == "LOSS").sum())
    win_rate = wins / total_exec * 100 if total_exec > 0 else 0.0
    total_pnl = float(executed_df["pnl_usd"].sum())
    be_rate = (1 / (1 + PAYOUT_RATE)) * 100

    # 見送り予測の仮想成績
    hold_wins = int(hold_df["result"].str.contains("WIN").sum()) if not hold_df.empty else 0
    hold_losses = int(hold_df["result"].str.contains("LOSS").sum()) if not hold_df.empty else 0
    hold_total = hold_wins + hold_losses
    hold_win_rate = hold_wins / hold_total * 100 if hold_total > 0 else 0.0

    st.markdown(
        f"""
        <div class="genius-banner">
            <h2>🧠 潜在意識×科学的検証 AIレポート (無制限全件記録)</h2>
            <p>実取引勝率: <strong>{win_rate:.2f}%</strong> (取引数: {total_exec}回 | 損益分岐: {be_rate:.2f}%) ｜ 実取引損益: <strong>${total_pnl:+.2f}</strong></p>
            <p>見送り(HOLD)予測の参考勝率: <strong>{hold_win_rate:.2f}%</strong> (見送り検証数: {holds}回)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if win_rate >= be_rate:
        st.success(f"🌟 実取引勝率 ({win_rate:.2f}%) は損益分岐点 ({be_rate:.2f}%) を突破しています。")
    else:
        st.warning(f"⚠️ 実取引勝率 ({win_rate:.2f}%) は損益分岐点 ({be_rate:.2f}%) を下回っています。")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🧭 選択方向の分布")
        dir_counts = res_df["direction"].value_counts().reindex(["HIGH", "LOW", "HOLD"], fill_value=0)
        st.bar_chart(dir_counts)
    with col2:
        st.subheader("📊 失敗・成功要因分類 (実取引)")
        if not executed_df.empty and "category" in executed_df.columns:
            cat_counts = executed_df["category"].value_counts()
            st.bar_chart(cat_counts)
    with col3:
        st.subheader("🎯 確信度(Confidence)の分布")
        st.line_chart(res_df["confidence"].tail(50))

    st.subheader("📈 累積損益の推移（実取引の成長曲線）")
    chart_df = executed_df[["time", "pnl_usd"]].copy()
    if not chart_df.empty:
        chart_df["time"] = pd.to_datetime(chart_df["time"])
        chart_df["累積損益"] = chart_df["pnl_usd"].cumsum()
        chart_df = chart_df.set_index("time")
        st.line_chart(chart_df[["累積損益"]])
    else:
        st.info("実行された取引データがありません。")

    st.subheader("📝 詳細な反省・敗因分析ログ (直近50件)")
    st.dataframe(
        res_df.sort_values("time", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )


# -------------------- ⑦ サイドバー設定 --------------------

st.sidebar.header("🌌 分析・モデル設定")
model_selection = st.sidebar.selectbox("🧠 モデル選択", options=["短期モデル (1分足)", "長期モデル (1時間足)"], index=0)
model_type_key = "1m" if "1分足" in model_selection else "1h"

target_usd = st.sidebar.slider("🎯 目標値幅 (±USD)", 10.0, 100.0, DEFAULT_TARGET_USD, 5.0)
stake_usd = st.sidebar.number_input("💵 仮想賭け金 ($)", 1.0, 10000.0, 5.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (足の本数)", 1, 10, 1)

be_rate_disp = (1 / (1 + PAYOUT_RATE)) * 100
st.sidebar.info(f"📊 損益分岐勝率: **{be_rate_disp:.2f}%** (ペ88%)")

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ 安全管理（無制限学習モード）")
st.sidebar.success("✅ 制限完全撤廃：連敗時も停止せず、全予測および見送り予測の追跡・学習を無限に継続します。")

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 データ・学習状態の管理")
reset_market_data = st.sidebar.checkbox("蓄積した市場データも削除する", value=False)
if st.sidebar.button("🔄 AI履歴・学習状態をリセット", use_container_width=True):
    st.session_state.utaya_monitoring = False
    files_to_delete = [VIRTUAL_TRADES_FILE, ANALYSIS_LOG_FILE, MODEL_1M_FILE, SCALER_1M_FILE, MODEL_1H_FILE, SCALER_1H_FILE]
    if reset_market_data:
        files_to_delete.extend([MARKET_HISTORY_1M, MARKET_HISTORY_1H])
    for file_path in files_to_delete:
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass
    st.sidebar.success("リセット完了しました。再読み込みしてください。")
    st.rerun()


# -------------------- ⑧ 連続監視・無制限自動学習エンジン --------------------

if "utaya_monitoring" not in st.session_state:
    st.session_state.utaya_monitoring = False

st.subheader("♾️ 潜在意識×検証 AI 連続監視・無制限自己成長モード")
col1, col2 = st.columns(2)
with col1:
    if st.button("▶ 連続監視・学習を開始", use_container_width=True):
        st.session_state.utaya_monitoring = True
with col2:
    if st.button("⏹ 連続監視を停止", use_container_width=True):
        st.session_state.utaya_monitoring = False

if st.session_state.utaya_monitoring:
    st.success(f"連続監視中 [{model_selection}]：30秒ごとに予測、全件永久保存、オンライン学習を実行中")
else:
    st.info("停止中です。開始ボタンで連続監視・分析を開始できます。")

@st.fragment(run_every="30s")
def continuous_learning_loop():
    if not st.session_state.utaya_monitoring:
        st.caption("監視は停止しています。")
        return

    interval_str = "1m" if model_type_key == "1m" else "1h"
    with st.spinner(f"BTC市場データ({model_selection})を更新し、深層分析中..."):
        live_df = fetch_market_data(interval_str)

    if live_df is None or len(live_df) < 50:
        st.warning("データが不足しています。")
        return

    env_desc, vol, risk_state = analyze_market_context(live_df)

    signal_i = len(live_df) - 2
    features_table = build_feature_table(live_df)
    feat_row = features_table.iloc[signal_i]

    if feat_row.isna().any():
        st.warning("特徴量計算エラー")
        return

    model, scaler = load_or_init_model(model_type_key)

    try:
        feat_sc = scaler.transform(np.array([feat_row.values]))
    except Exception:
        X_live = features_table.dropna().values
        scaler.fit(X_live)
        feat_sc = scaler.transform(np.array([feat_row.values]))

    try:
        probs = model.predict_proba(feat_sc)[0]
        p_low, p_high = probs[0], probs[1]
    except Exception:
        p_low, p_high = 0.5, 0.5

    confidence = max(p_high, p_low)
    
    if abs(p_high - p_low) < 0.04:
        live_direction = "HOLD"
    else:
        live_direction = "HIGH" if p_high >= p_low else "LOW"

    signal_price = float(live_df["Close"].iloc[signal_i])
    signal_time = live_df.index[signal_i]

    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 基準価格: ${signal_price:,.2f} | 相場環境: {env_desc}")

    if live_direction == "HIGH":
        st.markdown(
            f"""
            <div class="scenario-card-high">
                <h2>🚀 連続監視 予測判定: HIGH ({model_selection})</h2>
                <p>確率: HIGH <strong>{p_high:.1%}</strong> ｜ LOW <strong>{p_low:.1%}</strong> (確信度: {confidence:.2f})</p>
                <p>上方向目標価格: ${signal_price + target_usd:,.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif live_direction == "LOW":
        st.markdown(
            f"""
            <div class="scenario-card-low">
                <h2>📉 連続監視 予測判定: LOW ({model_selection})</h2>
                <p>確率: LOW <strong>{p_low:.1%}</strong> ｜ HIGH <strong>{p_high:.1%}</strong> (確信度: {confidence:.2f})</p>
                <p>下方向目標価格: ${signal_price - target_usd:,.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="scenario-card-hold">
                <h2>⏸️ 連続監視 判断保留: HOLD ({model_selection})</h2>
                <p>確率差僅少のため取引を見送りますが、検証用に全件永久保存して追跡します (HIGH: {p_high:.1%} / LOW: {p_low:.1%})</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if VIRTUAL_TRADES_FILE.exists():
        try:
            trades_df = pd.read_csv(VIRTUAL_TRADES_FILE)
        except Exception:
            trades_df = pd.DataFrame(columns=["time", "price", "direction", "result", "reason", "category", "pnl_usd"])
    else:
        trades_df = pd.DataFrame(columns=["time", "price", "direction", "result", "reason", "category", "pnl_usd"])

    time_str = str(signal_time)
    if trades_df.empty or time_str not in trades_df["time"].values:
        future_window = live_df.loc[signal_time:].iloc[1 : hold_bars + 1]
        if len(future_window) >= hold_bars:
            eval_dir = live_direction if live_direction != "HOLD" else ("HIGH" if p_high >= p_low else "LOW")
            win_flag, result, reason, category = evaluate_target_touch(signal_price, eval_dir, future_window, target_usd)
            
            # オンライン学習更新
            actual_next_c = float(live_df["Close"].iloc[signal_i + 1])
            actual_curr_c = float(live_df["Close"].iloc[signal_i])
            direction_label = 1 if actual_next_c > actual_curr_c else 0
            
            try:
                model.partial_fit(feat_sc, np.array([direction_label]))
            except ValueError:
                model.partial_fit(feat_sc, np.array([direction_label]), classes=np.array([0, 1]))
            save_model_state(model, scaler, model_type_key)

            if live_direction != "HOLD":
                pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd
                new_trade = pd.DataFrame([{
                    "time": time_str,
                    "price": signal_price,
                    "direction": live_direction,
                    "result": result,
                    "reason": reason,
                    "category": category,
                    "pnl_usd": pnl
                }])
                trades_df = pd.concat([trades_df, new_trade])
                trades_df.to_csv(VIRTUAL_TRADES_FILE, index=False, encoding="utf-8-sig")
            else:
                hold_pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd
                new_trade = pd.DataFrame([{
                    "time": time_str,
                    "price": signal_price,
                    "direction": "HOLD (見送り追跡)",
                    "result": f"HOLD_{result}",
                    "reason": f"見送り検証: {reason}",
                    "category": f"TRACKED_{category}",
                    "pnl_usd": hold_pnl
                }])
                trades_df = pd.concat([trades_df, new_trade])
                trades_df.to_csv(VIRTUAL_TRADES_FILE, index=False, encoding="utf-8-sig")

    if not trades_df.empty:
        st.subheader("📋 自律学習・全件永久保存ログ")
        st.dataframe(trades_df.sort_values("time", ascending=False).head(20), use_container_width=True, hide_index=True)
        
        executed_logs = trades_df[~trades_df["direction"].str.contains("HOLD", na=False)]
        v_wins = int((executed_logs["result"] == "WIN").sum())
        v_losses = int((executed_logs["result"] == "LOSS").sum())
        v_rate = v_wins / len(executed_logs) * 100 if len(executed_logs) > 0 else 0.0
        v_pnl = float(executed_logs["pnl_usd"].sum())
        st.caption(f"実取引実績: {v_wins}勝 {v_losses}敗 (勝率: {v_rate:.1f}%) | 累積損益: ${v_pnl:+.2f} | 記録総数: {len(trades_df)}件")

continuous_learning_loop()


# -------------------- ⑨ バックテスト実行ボタン --------------------

st.markdown("---")
st.subheader("🚀 大規模時系列ウォークフォワード検証の実行")

if st.button("▶️ 公平な検証トレーニングを開始", use_container_width=True):
    interval_str = "1m" if model_type_key == "1m" else "1h"
    with st.spinner(f"市場データ({model_selection})を取得し、本番モデルを汚染しない独立検証を実行中..."):
        full_df = fetch_market_data(interval_str)

        if full_df is None or len(full_df) < 300:
            st.error("データ件数が不足しています（最低300本必要です）。期間を長くしてください。")
        else:
            res_df = run_walkforward_simulation(
                full_df,
                target_usd=target_usd,
                stake_usd=stake_usd,
                hold_bars=hold_bars,
                model_type=model_type_key,
            )
            show_analyst_dashboard(res_df)
elif ANALYSIS_LOG_FILE.exists():
    try:
        saved_df = pd.read_csv(ANALYSIS_LOG_FILE)
        if not saved_df.empty:
            show_analyst_dashboard(saved_df)
    except Exception:
        pass
