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
# UTAYA 全銘柄統合型・世界市場スキャナーAIシステム (完全監査対応プロ版)
# 思想: 50/50偽装の根絶、方向予測の常時提示、判定待ちライフサイクル管理
# =========================================================

st.set_page_config(
    page_title="UTAYA Full-Market Pro Audit System",
    page_icon="🌌",
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
DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(exist_ok=True)

GLOBAL_HISTORY_DIR = DATA_DIR / "full_market_history_audit"
GLOBAL_HISTORY_DIR.mkdir(exist_ok=True)
VIRTUAL_TRADES_FILE = DATA_DIR / "full_market_virtual_trades_audit.csv"
ANALYSIS_LOG_FILE = DATA_DIR / "full_market_analysis_log_audit.csv"

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

st.title("🌌 UTAYA 世界市場スキャナーAI (監査対応・実戦プロ仕様)")
st.caption(
    "50/50偽装完全排除 | 方向予測の常時提示 | 判定待ち(PENDING)追跡 | PF・期待値検証"
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
    all_dfs = {}
    for name, ticker in SYMBOLS.items():
        df = fetch_single_market_data(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            all_dfs[name] = df
    return all_dfs


# -------------------- ② 特徴量生成・厳格予測エンジン --------------------

def build_full_integrated_features(target_name, all_dfs):
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

    features = pd.DataFrame(index=df_target.index)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["return_6"] = close.pct_change(6)
    features["sma_ratio_10"] = (close - close.rolling(10).mean()) / close.rolling(10).mean()
    features["sma_ratio_30"] = (close - close.rolling(30).mean()) / close.rolling(30).mean()
    features["rsi"] = rsi
    features["volatility"] = returns.rolling(14).std()

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


def evaluate_target_outcome(entry_price, direction, future_candles, target_pct=0.01, fee_pct=0.0005):
    if direction not in ("HIGH", "LOW") or future_candles is None or future_candles.empty:
        return 0, "PENDING", "判定期間内のデータ待機中", "PENDING"

    upper = float(entry_price) * (1.0 + target_pct + fee_pct)
    lower = float(entry_price) * (1.0 - target_pct - fee_pct)

    for idx, candle in future_candles.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])

        hit_up = high >= upper
        hit_down = low <= lower

        if hit_up and hit_down:
            return 0, "LOSS", "同一足で上下両方到達（高ボラティリティ・ノイズ交差）", "VOLATILITY_NOISE"
        if hit_up:
            if direction == "HIGH":
                return 1, "WIN", "上目標先着：コスト考慮後達成", "TREND_SUCCESS"
            else:
                return 0, "LOSS", "逆方向到達：逆行失敗", "REVERSE_FAILURE_HIGH"
        if hit_down:
            if direction == "LOW":
                return 1, "WIN", "下目標先着：コスト考慮後達成", "TREND_SUCCESS"
            else:
                return 0, "LOSS", "逆方向到達：下落逆行失敗", "REVERSE_FAILURE_LOW"

    return 0, "LOSS", "判定期間内未到達：レンジ停滞", "TIMEOUT_RANGE"


# -------------------- ③ モデル管理 --------------------

def load_or_init_model(symbol_key):
    safe_name = symbol_key.replace(" ", "_").replace("(", "").replace(")", "")
    m_file = DATA_DIR / f"audit_model_{safe_name}.joblib"
    s_file = DATA_DIR / f"audit_scaler_{safe_name}.joblib"

    if m_file.exists() and s_file.exists():
        try:
            model = joblib.load(m_file)
            scaler = joblib.load(s_file)
            return model, scaler, True
        except Exception:
            pass
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
    scaler = StandardScaler()
    return model, scaler, False


def save_model_state(model, scaler, symbol_key):
    safe_name = symbol_key.replace(" ", "_").replace("(", "").replace(")", "")
    m_file = DATA_DIR / f"audit_model_{safe_name}.joblib"
    s_file = DATA_DIR / f"audit_scaler_{safe_name}.joblib"
    joblib.dump(model, m_file)
    joblib.dump(scaler, s_file)


# -------------------- ④ バックテスト（ウォークフォワード） --------------------

def run_audit_walkforward(target_name, all_dfs, target_pct, stake_usd, hold_bars, fee_pct):
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
            if hasattr(model, "classes_") and len(model.classes_) == 2:
                probs = model.predict_proba(X_pred)[0]
                classes_list = list(model.classes_)
                p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
                p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
                model_status = "学習済み・正常"
            else:
                p_low, p_high = 0.5, 0.5
                model_status = "未学習・初期状態"
        except Exception:
            p_low, p_high = 0.5, 0.5
            model_status = "予測エラー・不確実"

        confidence = max(p_high, p_low)
        direction = "HIGH" if p_high >= p_low else "LOW"
        trade_decision = "取引実行" if abs(p_high - p_low) >= 0.04 else "見送り推奨 (HOLD)"

        entry_index = i + 1
        entry_price = float(df_target["Open"].iloc[entry_index])
        future_window = df_target.iloc[entry_index : entry_index + hold_bars]

        win_flag, result, reason, category = evaluate_target_outcome(entry_price, direction, future_window, target_pct, fee_pct)
        
        pnl = 0.0
        if trade_decision == "取引実行" and result in ("WIN", "LOSS"):
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
            "direction": direction,
            "p_high": p_high,
            "p_low": p_low,
            "confidence": confidence,
            "model_status": model_status,
            "trade_decision": trade_decision,
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

    executed_df = res_df[(res_df["trade_decision"] == "取引実行") & (res_df["result"].isin(["WIN", "LOSS"]))].copy()
    hold_df = res_df[res_df["trade_decision"].str.contains("見送り")].copy()

    total_exec = len(executed_df)
    holds = len(hold_df)
    
    wins = int((executed_df["result"] == "WIN").sum())
    losses = int((executed_df["result"] == "LOSS").sum())
    win_rate = wins / total_exec * 100 if total_exec > 0 else 0.0
    total_pnl = float(executed_df["pnl_usd"].sum())
    be_rate = (1 / (1 + PAYOUT_RATE)) * 100

    gross_profit = float(executed_df[executed_df["pnl_usd"] > 0]["pnl_usd"].sum())
    gross_loss = abs(float(executed_df[executed_df["pnl_usd"] < 0]["pnl_usd"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    st.markdown(
        f"""
        <div class="genius-banner">
            <h2>🌌 UTAYA 監査対応プロ検証レポート</h2>
            <p>実取引数: <strong>{total_exec}回</strong> (見送り/HOLD: {holds}回) ｜ 実践勝率: <strong>{win_rate:.2f}%</strong> (損益分岐: {be_rate:.2f}%)</p>
            <p>累積損益: <strong>${total_pnl:+.2f}</strong> ｜ プロフィットファクター (PF): <strong>{profit_factor:.2f}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if win_rate >= be_rate:
        st.success(f"🌟 実取引勝率 ({win_rate:.2f}%) は損益分岐点 ({be_rate:.2f}%) を突破しています。")
    else:
        st.warning(f"⚠️ 実取引勝率 ({win_rate:.2f}%) は損益分岐点 ({be_rate:.2f}%) を下回っています。")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧭 方向予測の分布")
        dir_counts = res_df["direction"].value_counts().reindex(["HIGH", "LOW"], fill_value=0)
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

    st.subheader("📝 詳細監査ログ (直近50件)")
    st.dataframe(
        res_df.sort_values("time", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )


# -------------------- ⑥ サイドバー設定 --------------------

st.sidebar.header("🌌 監査プロ設定")
selected_symbol = st.sidebar.selectbox("🎯 分析対象銘柄・市場", options=list(SYMBOLS.keys()), index=0)

timeframe = st.sidebar.selectbox("⏱️ 分析時間軸", options=["1分足 (短期)", "1時間足 (中期)"], index=1)
interval_str = "1m" if "1分足" in timeframe else "1h"
period_str = "7d" if "1分足" in timeframe else "180d"

target_pct = st.sidebar.slider("🎯 目標変動率 (±%)", 0.002, 0.05, 0.01, 0.001, format="%.3f")
fee_pct = st.sidebar.slider("💸 手数料・スプレッド考慮 (%)", 0.0, 0.005, 0.0005, 0.0001, format="%.4f")
stake_usd = st.sidebar.number_input("💵 仮想賭け金 ($)", 1.0, 10000.0, 5.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (足の本数)", 1, 20, 5)

be_rate_disp = (1 / (1 + PAYOUT_RATE)) * 100
st.sidebar.info(f"📊 損益分岐勝率: **{be_rate_disp:.2f}%** (ペ88%)")

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 データ・学習状態の管理")
if st.sidebar.button("🔄 AI履歴・学習状態をリセット", use_container_width=True):
    st.session_state.audit_monitoring = False
    files_to_delete = [VIRTUAL_TRADES_FILE, ANALYSIS_LOG_FILE]
    for file_path in files_to_delete:
        try:
            if file_path.exists(): file_path.unlink()
        except OSError:
            pass
    st.sidebar.success("リセット完了しました。再読み込みしてください。")
    st.rerun()


# -------------------- ⑦ 連続監視・判定待ちパイプライン --------------------

if "audit_monitoring" not in st.session_state:
    st.session_state.audit_monitoring = False

st.subheader(f"♾️ 監査監視・判定待ちパイプライン [{selected_symbol}]")
col1, col2 = st.columns(2)
with col1:
    if st.button("▶ 連続監視・学習を開始", use_container_width=True):
        st.session_state.audit_monitoring = True
with col2:
    if st.button("⏹ 連続監視を停止", use_container_width=True):
        st.session_state.audit_monitoring = False

if st.session_state.audit_monitoring:
    st.success(f"連続監視中 [{selected_symbol}]：判定待ち足の追跡とオンライン学習を稼働中")
else:
    st.info("停止中です。開始ボタンで連続監視・分析を開始できます。")

@st.fragment(run_every="30s")
def continuous_audit_loop():
    if not st.session_state.audit_monitoring:
        st.caption("監視は停止しています。")
        return

    with st.spinner("全主要市場のデータを同期中..."):
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

    model, scaler, is_trained = load_or_init_model(selected_symbol)

    try:
        feat_sc = scaler.transform(np.array([feat_row.values]))
    except Exception:
        X_live = features_table.dropna().values
        scaler.fit(X_live)
        feat_sc = scaler.transform(np.array([feat_row.values]))

    try:
        if is_trained and hasattr(model, "classes_") and len(model.classes_) == 2:
            probs = model.predict_proba(feat_sc)[0]
            classes_list = list(model.classes_)
            p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
            p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
            model_status = "学習済み・正常"
        else:
            p_low, p_high = 0.5, 0.5
            model_status = "⚠️ 未学習・初期状態（要ウォークフォワード検証）"
    except Exception:
        p_low, p_high = 0.5, 0.5
        model_status = "❌ 予測エラー・不確実"

    confidence = max(p_high, p_low)
    direction = "HIGH" if p_high >= p_low else "LOW"
    trade_decision = "取引実行" if abs(p_high - p_low) >= 0.04 else "見送り推奨 (HOLD)"

    signal_price = float(df_target["Close"].iloc[signal_i])
    signal_time = df_target.index[signal_i]

    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 基準価格: ${signal_price:,.2f} | 状態: {model_status}")

    card_class = "scenario-card-high" if direction == "HIGH" else "scenario-card-low"
    st.markdown(
        f"""
        <div class="{card_class}">
            <h2>🧭 方向予測: {direction} [{selected_symbol}]</h2>
            <p>モデル推定確率: HIGH <strong>{p_high:.1%}</strong> ｜ LOW <strong>{p_low:.1%}</strong> (確信度: {confidence:.2f})</p>
            <p>予測状態: <strong>{model_status}</strong> ｜ 取引判断: <strong>{trade_decision}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if VIRTUAL_TRADES_FILE.exists():
        try:
            trades_df = pd.read_csv(VIRTUAL_TRADES_FILE)
        except Exception:
            trades_df = pd.DataFrame(columns=["time", "symbol", "direction", "p_high", "p_low", "confidence", "model_status", "trade_decision", "result", "reason", "category", "pnl_usd"])
    else:
        trades_df = pd.DataFrame(columns=["time", "symbol", "direction", "p_high", "p_low", "confidence", "model_status", "trade_decision", "result", "reason", "category", "pnl_usd"])

    time_str = str(signal_time)
    if trades_df.empty or not ((trades_df["time"] == time_str) & (trades_df["symbol"] == selected_symbol)).any():
        future_window = df_target.loc[signal_time:].iloc[1 : hold_bars + 1]
        
        # 判定期間分の足が揃っているか確認
        if len(future_window) >= hold_bars:
            win_flag, result, reason, category = evaluate_target_outcome(signal_price, direction, future_window, target_pct, fee_pct)
        else:
            result, reason, category = "PENDING", "判定期間（未来足）のデータ待機中", "PENDING"

        actual_next_c = float(df_target["Close"].iloc[signal_i + 1])
        actual_curr_c = float(df_target["Close"].iloc[signal_i])
        direction_label = 1 if actual_next_c > actual_curr_c else 0
        
        try:
            model.partial_fit(feat_sc, np.array([direction_label]), classes=np.array([0, 1]))
            save_model_state(model, scaler, selected_symbol)
        except ValueError:
            model.partial_fit(feat_sc, np.array([direction_label]))
            save_model_state(model, scaler, selected_symbol)

        pnl = 0.0
        if trade_decision == "取引実行" and result in ("WIN", "LOSS"):
            pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd

        new_trade = pd.DataFrame([{
            "time": time_str,
            "symbol": selected_symbol,
            "direction": direction,
            "p_high": p_high,
            "p_low": p_low,
            "confidence": confidence,
            "model_status": model_status,
            "trade_decision": trade_decision,
            "result": result,
            "reason": reason,
            "category": category,
            "pnl_usd": pnl
        }])
        trades_df = pd.concat([trades_df, new_trade])
        trades_df.to_csv(VIRTUAL_TRADES_FILE, index=False, encoding="utf-8-sig")

    if not trades_df.empty:
        st.subheader("📋 監査監視・全件追跡ログ")
        st.dataframe(trades_df.sort_values("time", ascending=False).head(20), use_container_width=True, hide_index=True)
        
        executed_logs = trades_df[(trades_df["trade_decision"] == "取引実行") & (trades_df["result"].isin(["WIN", "LOSS"]))]
        v_wins = int((executed_logs["result"] == "WIN").sum())
        v_losses = int((executed_logs["result"] == "LOSS").sum())
        v_rate = v_wins / len(executed_logs) * 100 if len(executed_logs) > 0 else 0.0
        v_pnl = float(executed_logs["pnl_usd"].sum())
        st.caption(f"実取引実績: {v_wins}勝 {v_losses}敗 (勝率: {v_rate:.1f}%) | 累積損益: ${v_pnl:+.2f} | 記録総数: {len(trades_df)}件")

continuous_audit_loop()


# -------------------- ⑧ バックテスト実行ボタン --------------------

st.markdown("---")
st.subheader("🚀 監査対応ウォークフォワード検証の実行")

if st.button("▶️ 公平な検証トレーニングを開始", use_container_width=True):
    with st.spinner("全市場データを一括取得し、監査対応ウォークフォワード検証を実行中..."):
        all_dfs = fetch_all_markets_data(period=period_str, interval=interval_str)
        df_target = all_dfs.get(selected_symbol)

        if df_target is None or len(df_target) < 300:
            st.error("対象データの件数が不足しています（最低300本必要です）。期間を長くしてください。")
        else:
            res_df = run_audit_walkforward(
                selected_symbol,
                all_dfs,
                target_pct=target_pct,
                stake_usd=stake_usd,
                hold_bars=hold_bars,
                fee_pct=fee_pct,
            )
            show_dashboard(res_df)
elif ANALYSIS_LOG_FILE.exists():
    try:
        saved_df = pd.read_csv(ANALYSIS_LOG_FILE)
        if not saved_df.empty:
            show_dashboard(saved_df)
    except Exception:
        pass
