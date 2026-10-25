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
# UTAYA 全銘柄一括統合型・世界市場スキャナーAIシステム
# 思想: すべての主要市場（暗号資産・株・為替・金利）の資金の波及を統合学習する
# =========================================================

st.set_page_config(
    page_title="UTAYA Full-Market Integrated AI Analyst",
    page_icon="🌌",
    layout="wide",
)

# 全主要銘柄の定義
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

PAYOUT_RATE = 0.88  # 配当88% -> 損益分岐勝率 約53.19%
DEFAULT_TARGET_PCT = 0.01 
DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(exist_ok=True)

GLOBAL_HISTORY_DIR = DATA_DIR / "full_market_history"
GLOBAL_HISTORY_DIR.mkdir(exist_ok=True)
VIRTUAL_TRADES_FILE = DATA_DIR / "full_market_virtual_trades.csv"
ANALYSIS_LOG_FILE = DATA_DIR / "full_market_analysis_log.csv"

# -------------------- CSS --------------------
st.markdown(
    """
    <style>
    .genius-banner {
        background: linear-gradient(135deg, #141e30, #243b55);
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

st.title("🌌 UTAYA 全銘柄一括統合型・世界市場スキャナーAIシステム")
st.caption(
    "全主要市場（暗号資産・株・為替・金利）の資金連動マトリクスを一網打尽に学習する完全版"
)


# -------------------- ① 全銘柄データ一括キャッシュ取得エンジン --------------------

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
            cols = ["Open", "High", "Low", "Close"]
            if all(c in df_new.columns for c in cols):
                df_new = df_new[cols].copy()
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
    """動いている全主要銘柄のデータを一括取得して辞書で返す"""
    all_dfs = {}
    for name, ticker in SYMBOLS.items():
        df = fetch_single_market_data(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            all_dfs[name] = df
    return all_dfs


# -------------------- ② 全銘柄統合・高度特徴量エンジニアリング --------------------

def build_full_integrated_features(target_name, all_dfs):
    """
    選択されたターゲット銘柄に対し、自分自身のテクニカル指標に加え、
    他のすべての市場（暗号資産・株・為替・金利）の変動率・MA比率をすべて結合する。
    """
    if target_name not in all_dfs:
        return pd.DataFrame()

    df_target = all_dfs[target_name].copy()
    df_target.index = pd.to_datetime(df_target.index)
    if df_target.index.tz is not None:
        df_target.index = df_target.index.tz_localize(None)

    close = df_target["Close"].astype(float)
    returns = close.pct_change()
    delta = close.diff()
    
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(losses != 0, 100.0)
    rsi = rsi.where(gains != 0, 50.0)

    # 自銘柄の基本特徴量
    features = pd.DataFrame(index=df_target.index)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["return_6"] = close.pct_change(6)
    features["sma_ratio_10"] = (close - close.rolling(10).mean()) / close.rolling(10).mean()
    features["sma_ratio_30"] = (close - close.rolling(30).mean()) / close.rolling(30).mean()
    features["rsi"] = rsi
    features["volatility"] = returns.rolling(14).std()

    # 他の全市場の動きを結合（資金の波及・連動マトリクス）
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
        features[f"{short_tag}_sma10"] = (other_close - other_close.rolling(10).mean()) / other_close.rolling(10).mean()

    return features.replace([np.inf, -np.inf], np.nan)


def evaluate_global_outcome(entry, direction, future, target_pct=0.01):
    if direction not in ("HIGH", "LOW") or future is None or future.empty:
        return 0, "LOSS", "判定データ不足", "DATA_ERROR"

    upper = float(entry) * (1.0 + target_pct)
    lower = float(entry) * (1.0 - target_pct)

    for idx, candle in future.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])

        hit_up = high >= upper
        hit_down = low <= lower

        if hit_up and hit_down:
            return 0, "LOSS", "同一足で上下両方到達（高ボラティリティ・ノイズ交差）", "VOLATILITY_NOISE"
        if hit_up:
            if direction == "HIGH":
                return 1, "WIN", "上目標先着：全市場連動・資金流入成功", "TREND_SUCCESS"
            else:
                return 0, "LOSS", "逆方向到達：リスクオフ・逆行失敗", "REVERSE_FAILURE_HIGH"
        if hit_down:
            if direction == "LOW":
                return 1, "WIN", "下目標先着：下落予測の的中", "TREND_SUCCESS"
            else:
                return 0, "LOSS", "逆方向到達：下落への逆行失敗", "REVERSE_FAILURE_LOW"

    return 0, "LOSS", "判定期間内未到達：レンジ相場・資金停滞", "TIMEOUT_RANGE"


# -------------------- ③ モデル永続化管理 --------------------

def load_or_init_model(symbol_key):
    safe_name = symbol_key.replace(" ", "_").replace("(", "").replace(")", "")
    m_file = DATA_DIR / f"full_model_{safe_name}.joblib"
    s_file = DATA_DIR / f"full_scaler_{safe_name}.joblib"

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

def save_model_state(model, scaler, symbol_key):
    safe_name = symbol_key.replace(" ", "_").replace("(", "").replace(")", "")
    m_file = DATA_DIR / f"full_model_{safe_name}.joblib"
    s_file = DATA_DIR / f"full_scaler_{safe_name}.joblib"
    joblib.dump(model, m_file)
    joblib.dump(scaler, s_file)


# -------------------- ④ ウォークフォワード検証エンジン --------------------

def run_full_walkforward(target_name, all_dfs, target_pct, stake_usd, hold_bars):
    df_target = all_dfs.get(target_name)
    if df_target is None or len(df_target) < 300:
        return pd.DataFrame()

    features_df = build_full_integrated_features(target_name, all_dfs)
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
        X_init = np.array(X_init)
        y_init = np.array(y_init)
        scaler.fit(X_init)
        X_init_scaled = scaler.transform(X_init)
        model.partial_fit(X_init_scaled, y_init, classes=np.array([0, 1]))

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
        entry_price = float(df_target["Open"].iloc[entry_index])
        future = df_target.iloc[entry_index : entry_index + hold_bars]

        eval_dir = direction if direction != "HOLD" else ("HIGH" if p_high >= p_low else "LOW")
        win_flag, result, reason, category = evaluate_global_outcome(entry_price, eval_dir, future, target_pct)
        
        pnl = 0.0
        if direction != "HOLD":
            pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

        actual_next_c = float(df_target["Close"].iloc[entry_index])
        actual_curr_c = float(df_target["Close"].iloc[entry_index - 1])
        direction_label = 1 if actual_next_c > actual_curr_c else 0
        
        try:
            model.partial_fit(X_pred, np.array([direction_label]))
        except ValueError:
            model.partial_fit(X_pred, np.array([direction_label]), classes=np.array([0, 1]))

        records.append({
            "time": str(df_target.index[i]),
            "symbol": target_name,
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


# -------------------- ⑤ ダッシュボード描画 --------------------

def show_dashboard(res_df):
    if res_df.empty:
        st.warning("検証データがありません。")
        return

    executed_df = res_df[res_df["direction"] != "HOLD"].copy()
    total_exec = len(executed_df)
    total_all = len(res_df)
    holds = total_all - total_exec
    
    wins = int((executed_df["result"] == "WIN").sum())
    losses = int((executed_df["result"] == "LOSS").sum())
    win_rate = wins / total_exec * 100 if total_exec > 0 else 0.0
    total_pnl = float(executed_df["pnl_usd"].sum())
    be_rate = (1 / (1 + PAYOUT_RATE)) * 100

    st.markdown(
        f"""
        <div class="genius-banner">
            <h2>🌌 全銘柄統合スキャナー 検証レポート</h2>
            <p>実取引数: <strong>{total_exec}回</strong> (見送り/HOLD: {holds}回) ｜ 実践勝率: <strong>{win_rate:.2f}%</strong> (損益分岐: {be_rate:.2f}%) ｜ 累積損益: <strong>${total_pnl:+.2f}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if win_rate >= be_rate:
        st.success(f"🌟 検証勝率 ({win_rate:.2f}%) は損益分岐点 ({be_rate:.2f}%) を突破しています。")
    else:
        st.warning(f"⚠️ 検証勝率 ({win_rate:.2f}%) は損益分岐点 ({be_rate:.2f}%) を下回っています。")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧭 選択方向の分布")
        dir_counts = res_df["direction"].value_counts().reindex(["HIGH", "LOW", "HOLD"], fill_value=0)
        st.bar_chart(dir_counts)
    with col2:
        st.subheader("📊 要因分類 (実取引)")
        if not executed_df.empty and "category" in executed_df.columns:
            cat_counts = executed_df["category"].value_counts()
            st.bar_chart(cat_counts)

    st.subheader("📈 累積損益の推移（成長曲線）")
    chart_df = executed_df[["time", "pnl_usd"]].copy()
    if not chart_df.empty:
        chart_df["time"] = pd.to_datetime(chart_df["time"])
        chart_df["累積損益"] = chart_df["pnl_usd"].cumsum()
        chart_df = chart_df.set_index("time")
        st.line_chart(chart_df[["累積損益"]])
    else:
        st.info("実行された取引データがありません。")

    st.subheader("📝 詳細検証ログ (直近50件)")
    st.dataframe(
        res_df.sort_values("time", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )


# -------------------- ⑥ サイドバー設定 --------------------

st.sidebar.header("🌌 全市場インテグレーション設定")
selected_symbol = st.sidebar.selectbox("🎯 分析対象銘柄・市場", options=list(SYMBOLS.keys()), index=0)

timeframe = st.sidebar.selectbox("⏱️ 分析時間軸", options=["1分足 (短期)", "1時間足 (中期)"], index=1)
interval_str = "1m" if "1分足" in timeframe else "1h"
period_str = "7d" if "1分足" in timeframe else "180d"

target_pct = st.sidebar.slider("🎯 目標変動率 (±%)", 0.002, 0.05, 0.01, 0.001, format="%.3f")
stake_usd = st.sidebar.number_input("💵 仮想賭け金 ($)", 1.0, 10000.0, 5.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (足の本数)", 1, 20, 5)

be_rate_disp = (1 / (1 + PAYOUT_RATE)) * 100
st.sidebar.info(f"📊 損益分岐勝率: **{be_rate_disp:.2f}%** (ペ88%)")

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 データ・学習状態の管理")
if st.sidebar.button("🔄 AI履歴・学習状態をリセット", use_container_width=True):
    st.session_state.full_monitoring = False
    files_to_delete = [VIRTUAL_TRADES_FILE, ANALYSIS_LOG_FILE]
    for file_path in files_to_delete:
        try:
            if file_path.exists(): file_path.unlink()
        except OSError:
            pass
    st.sidebar.success("リセット完了しました。再読み込みしてください。")
    st.rerun()


# -------------------- ⑦ 連続監視・自動学習エンジン --------------------

if "full_monitoring" not in st.session_state:
    st.session_state.full_monitoring = False

st.subheader(f"♾️ 全銘柄統合スキャナー 連続監視モード [{selected_symbol}]")
col1, col2 = st.columns(2)
with col1:
    if st.button("▶ 連続監視・学習を開始", use_container_width=True):
        st.session_state.full_monitoring = True
with col2:
    if st.button("⏹ 連続監視を停止", use_container_width=True):
        st.session_state.full_monitoring = False

if st.session_state.full_monitoring:
    st.success(f"連続監視中 [{selected_symbol}]：全主要市場の資金連動を解析し、全件永久保存でオンライン学習を実行中")
else:
    st.info("停止中です。開始ボタンで連続監視・分析を開始できます。")

@st.fragment(run_every="30s")
def continuous_loop():
    if not st.session_state.full_monitoring:
        st.caption("監視は停止しています。")
        return

    with st.spinner("全主要市場（暗号資産・株式・為替・金利）のデータを一括同期中..."):
        all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)

    df_target = all_dfs.get(selected_symbol)
    if df_target is None or len(df_target) < 50:
        st.warning("対象データの件数が不足しています。")
        return

    features_table = build_full_integrated_features(selected_symbol, all_dfs)
    signal_i = len(df_target) - 2
    feat_row = features_table.iloc[signal_i]

    if feat_row.isna().any():
        st.warning("特徴量計算エラー（同期待ち）")
        return

    model, scaler = load_or_init_model(selected_symbol)

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

    signal_price = float(df_target["Close"].iloc[signal_i])
    signal_time = df_target.index[signal_i]

    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 基準価格: ${signal_price:,.2f}")

    if live_direction == "HIGH":
        st.markdown(
            f"""
            <div class="scenario-card-high">
                <h2>🚀 全市場統合 予測判定: HIGH [{selected_symbol}]</h2>
                <p>確率: HIGH <strong>{p_high:.1%}</strong> ｜ LOW <strong>{p_low:.1%}</strong> (確信度: {confidence:.2f})</p>
                <p>上方向目標価格: ${signal_price * (1.0 + target_pct):,.2f} (+{target_pct*100}%)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif live_direction == "LOW":
        st.markdown(
            f"""
            <div class="scenario-card-low">
                <h2>📉 全市場統合 予測判定: LOW [{selected_symbol}]</h2>
                <p>確率: LOW <strong>{p_low:.1%}</strong> ｜ HIGH <strong>{p_high:.1%}</strong> (確信度: {confidence:.2f})</p>
                <p>下方向目標価格: ${signal_price * (1.0 - target_pct):,.2f} (-{target_pct*100}%)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="scenario-card-hold">
                <h2>⏸️ 判断保留: HOLD [{selected_symbol}]</h2>
                <p>確率差僅少のため取引を見送りますが、検証用に全件永久保存して追跡します (HIGH: {p_high:.1%} / LOW: {p_low:.1%})</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if VIRTUAL_TRADES_FILE.exists():
        try:
            trades_df = pd.read_csv(VIRTUAL_TRADES_FILE)
        except Exception:
            trades_df = pd.DataFrame(columns=["time", "symbol", "direction", "result", "reason", "category", "pnl_usd"])
    else:
        trades_df = pd.DataFrame(columns=["time", "symbol", "direction", "result", "reason", "category", "pnl_usd"])

    time_str = str(signal_time)
    if trades_df.empty or not ((trades_df["time"] == time_str) & (trades_df["symbol"] == selected_symbol)).any():
        future_window = df_target.loc[signal_time:].iloc[1 : hold_bars + 1]
        if len(future_window) >= hold_bars:
            eval_dir = live_direction if live_direction != "HOLD" else ("HIGH" if p_high >= p_low else "LOW")
            win_flag, result, reason, category = evaluate_global_outcome(signal_price, eval_dir, future_window, target_pct)
            
            actual_next_c = float(df_target["Close"].iloc[signal_i + 1])
            actual_curr_c = float(df_target["Close"].iloc[signal_i])
            direction_label = 1 if actual_next_c > actual_curr_c else 0
            
            try:
                model.partial_fit(feat_sc, np.array([direction_label]))
            except ValueError:
                model.partial_fit(feat_sc, np.array([direction_label]), classes=np.array([0, 1]))
            save_model_state(model, scaler, selected_symbol)

            if live_direction != "HOLD":
                pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd
                new_trade = pd.DataFrame([{
                    "time": time_str,
                    "symbol": selected_symbol,
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
                    "symbol": selected_symbol,
                    "direction": "HOLD (見送り追跡)",
                    "result": f"HOLD_{result}",
                    "reason": f"見送り検証: {reason}",
                    "category": f"TRACKED_{category}",
                    "pnl_usd": hold_pnl
                }])
                trades_df = pd.concat([trades_df, new_trade])
                trades_df.to_csv(VIRTUAL_TRADES_FILE, index=False, encoding="utf-8-sig")

    if not trades_df.empty:
        st.subheader("📋 全市場統合スキャナー・全件追跡ログ")
        st.dataframe(trades_df.sort_values("time", ascending=False).head(20), use_container_width=True, hide_index=True)
        
        executed_logs = trades_df[~trades_df["direction"].str.contains("HOLD", na=False)]
        v_wins = int((executed_logs["result"] == "WIN").sum())
        v_losses = int((executed_logs["result"] == "LOSS").sum())
        v_rate = v_wins / len(executed_logs) * 100 if len(executed_logs) > 0 else 0.0
        v_pnl = float(executed_logs["pnl_usd"].sum())
        st.caption(f"実取引実績: {v_wins}勝 {v_losses}敗 (勝率: {v_rate:.1f}%) | 累積損益: ${v_pnl:+.2f} | 記録総数: {len(trades_df)}件")

continuous_loop()


# -------------------- ⑧ バックテスト実行ボタン --------------------

st.markdown("---")
st.subheader("🚀 全市場統合ウォークフォワード検証の実行")

if st.button("▶️ 公平な検証トレーニングを開始", use_container_width=True):
    with st.spinner("全主要市場（暗号資産・株式・為替・金利）のデータを一括取得し、検証を実行中..."):
        all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
        df_target = all_dfs.get(selected_symbol)

        if df_target is None or len(df_target) < 300:
            st.error("対象データの件数が不足しています（最低300本必要です）。期間を長くしてください。")
        else:
            res_df = run_full_walkforward(
                selected_symbol,
                all_dfs,
                target_pct=target_pct,
                stake_usd=stake_usd,
                hold_bars=hold_bars,
            )
            show_dashboard(res_df)
elif ANALYSIS_LOG_FILE.exists():
    try:
        saved_df = pd.read_csv(ANALYSIS_LOG_FILE)
        if not saved_df.empty:
            show_dashboard(saved_df)
    except Exception:
        pass
