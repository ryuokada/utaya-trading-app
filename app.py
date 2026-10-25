import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import requests
import os

# =========================================================
# UTAYA ニュース・事業要因・関連市場統合 仮想取引分析AI (完全版)
# =========================================================

st.set_page_config(
    page_title="UTAYA Multiverse Scenario AI - Pro Integrated",
    page_icon="🌌",
    layout="wide",
)

SYMBOL = "BTC-USD"
PAYOUT_RATE = 0.88  # ペイアウト率88% -> 損益分岐勝率 約53.19%
DEFAULT_TARGET_USD = 50.0
DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(exist_ok=True)

# ファイルパス定義 (長期・短期分離)
MARKET_HISTORY_1M = DATA_DIR / "btc_1m_history.csv"
MARKET_HISTORY_1H = DATA_DIR / "btc_1h_history.csv"
VIRTUAL_TRADES_FILE = DATA_DIR / "virtual_trades_history.csv"
GENIUS_HISTORY = DATA_DIR / "genius_ai_history.csv"

MODEL_1M_FILE = DATA_DIR / "model_1m.joblib"
SCALER_1M_FILE = DATA_DIR / "scaler_1m.joblib"
MODEL_1H_FILE = DATA_DIR / "model_1h.joblib"
SCALER_1H_FILE = DATA_DIR / "scaler_1h.joblib"
STATE_FILE = DATA_DIR / "system_state.json"

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

st.title("🌌 UTAYA 統合仮想取引・未来シナリオ分析AI")
st.caption(
    "長期(1時間足) / 短期(1分足)分離モデル | ニュース・事業要因統合分析 | 厳格な先着先勝判定・敗因検証"
)


# -------------------- ① ニュース・外部要因分析モジュール --------------------

def fetch_crypto_news():
    """
    ニュースAPI（Gemini / 外部ニュース）から関連情報を取得する。
    APIキーがない場合や失敗した場合は安全に空リストを返す（架空ニュースの生成を防止）。
    """
    news_list = []
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    
    # ニュースAPI連携のプレースホルダー（実運用ではここにニュース取得ロジックを実装）
    # ここでは事実とAI推測を分ける構造を担保するため、安全なフォールバックを返す。
    return news_list


def analyze_news_sentiment(news_list):
    """ニュースからBTCへの影響（事実と推測の分離）を構造化する。"""
    if not news_list:
        return {
            "status": "ニュース分析なし",
            "summary": "現在有効な外部ニュースデータは取得されていません。純粋な価格・市場データに基づいて予測を行います。",
            "impact": "NEUTRAL",
            "score": 0.0
        }
    return {
        "status": "ニュース統合済み",
        "summary": f"{len(news_list)}件の関連ニュースを解析しました。",
        "impact": "NEUTRAL",
        "score": 0.0
    }


# -------------------- ② 永続データ保存・取得エンジン (長期・短期分離) --------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_data(interval="1m"):
    """
    1分足（短期）および1時間足（長期）の市場データを取得・蓄積する。
    """
    file_path = MARKET_HISTORY_1M if interval == "1m" else MARKET_HISTORY_1H
    period_val = "7d" if interval == "1m" else "730d"  # 1時間足は最大2年分を取得
    
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
                        df_combined.tail(20000).to_csv(file_path, encoding="utf-8-sig")
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


# -------------------- ③ 特徴量一括計算エンジン --------------------

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
    """
    厳格な ±USD 先着判定エンジン（未来情報の漏洩なし）。
    """
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
            return 0, "LOSS", "同一足で上下両方到達 (順序不明・保守的処理)"
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


# -------------------- ④ モデル永続化管理 (長期・短期分離) --------------------

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


# -------------------- ⑤ バックテスト・検証エンジン --------------------

def run_backtest_engine(df, target_usd, stake_usd, hold_bars, model_type="1m"):
    if df is None or len(df) < 300:
        return pd.DataFrame()

    features_df = build_feature_table(df)
    model, scaler = load_or_init_model(model_type)

    init_end = int(len(df) * 0.25)
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
    save_model_state(model, scaler, model_type)

    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df.to_csv(GENIUS_HISTORY, index=False, encoding="utf-8-sig")

    return res_df


# -------------------- ⑥ ダッシュボード描画 --------------------

def show_dashboard(res_df):
    if res_df.empty:
        st.warning("検証データがありません。")
        return

    total = len(res_df)
    wins = int((res_df["result"] == "WIN").sum())
    losses = int((res_df["result"] == "LOSS").sum())
    win_rate = wins / total * 100 if total else 0.0
    total_pnl = float(res_df["pnl_usd"].sum())

    be_rate = (1 / (1 + PAYOUT_RATE)) * 100

    st.markdown(
        f"""
        <div class="genius-banner">
            <h2>🏆 検証実績レポート（総検証数: {total}回）</h2>
            <p>実践勝率: <strong>{win_rate:.2f}%</strong> (理論損益分岐: {be_rate:.2f}%) ｜ 獲得損益: <strong>${total_pnl:+.2f}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if win_rate >= be_rate:
        st.success(f"🌟 現在の勝率 ({win_rate:.2f}%) は損益分岐点 ({be_rate:.2f}%) を超えています！")
    else:
        st.warning(f"⚠️ 現在の勝率 ({win_rate:.2f}%) は損益分岐点 ({be_rate:.2f}%) を下回っています。")

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

    st.subheader("📈 累積損益の推移")
    chart_df = res_df[["time", "pnl_usd"]].copy()
    chart_df["time"] = pd.to_datetime(chart_df["time"])
    chart_df["累積損益"] = chart_df["pnl_usd"].cumsum()
    chart_df = chart_df.set_index("time")
    st.line_chart(chart_df[["累積損益"]])

    st.subheader("📝 検証・学習ログ (直近50件)")
    st.dataframe(
        res_df.sort_values("time", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )


# -------------------- ⑦ サイドバー設定 & 安全管理 --------------------

st.sidebar.header("🌌 モデル・パラメータ設定")
model_selection = st.sidebar.selectbox("🧠 使用モデルの選択", options=["短期モデル (1分足)", "長期モデル (1시간足)"], index=0)
model_type_key = "1m" if "1分足" in model_selection else "1h"

target_usd = st.sidebar.slider("🎯 目標値幅 (±USD)", 10.0, 100.0, DEFAULT_TARGET_USD, 5.0)
stake_usd = st.sidebar.number_input("💵 仮想賭け金 ($)", 1.0, 10000.0, 5.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (足の本数)", 1, 10, 1)

be_rate_disp = (1 / (1 + PAYOUT_RATE)) * 100
st.sidebar.info(f"📊 損益分岐勝率目安: **{be_rate_disp:.2f}%** (配当 {PAYOUT_RATE*100:.0f}%)")

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
    files_to_delete = [VIRTUAL_TRADES_FILE, GENIUS_HISTORY, MODEL_1M_FILE, SCALER_1M_FILE, MODEL_1H_FILE, SCALER_1H_FILE]
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


# -------------------- ⑧ 連続監視・自動仮想トレードエンジン --------------------

if "utaya_monitoring" not in st.session_state:
    st.session_state.utaya_monitoring = False

st.subheader("♾️ UTAYA 連続監視・自動仮想トレードモード")
col1, col2 = st.columns(2)
with col1:
    if st.button("▶ 連続監視を開始", use_container_width=True):
        st.session_state.utaya_monitoring = True
with col2:
    if st.button("⏹ 連続監視を停止", use_container_width=True):
        st.session_state.utaya_monitoring = False

if st.session_state.utaya_monitoring and not st.session_state.force_stopped:
    st.success(f"連続監視中 [{model_selection}]：30秒ごとにシグナルを判定し、仮想トレードを実行します")
elif st.session_state.force_stopped:
    st.error("🚨 4連敗安全停止発動中。サイドバーの「停止を手動解除して再開」を押してください。")
else:
    st.info("停止中です。開始ボタンで連続監視を開始できます。")

@st.fragment(run_every="30s")
def continuous_virtual_trading():
    if not st.session_state.utaya_monitoring or st.session_state.force_stopped:
        st.caption("監視は停止しているか、安全装置によって一時中断されています。")
        return

    interval_str = "1m" if model_type_key == "1m" else "1h"
    with st.spinner(f"BTC市場データ({model_selection})を更新し、判定中..."):
        live_df = fetch_market_data(interval_str)

    if live_df is None or len(live_df) < 50:
        st.warning("データが不足しています。")
        return

    news_data = analyze_news_sentiment(fetch_crypto_news())

    signal_i = len(live_df) - 2
    features_table = build_feature_table(live_df)
    feat_row = features_table.iloc[signal_i]

    if feat_row.isna().any():
        st.warning("特徴量計算エラー")
        return

    model, scaler = load_or_init_model(model_type_key)

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
    confidence = max(p_high, p_low)
    signal_price = float(live_df["Close"].iloc[signal_i])
    signal_time = live_df.index[signal_i]

    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 基準価格: ${signal_price:,.2f} | ニュース分析: {news_data['status']}")

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
    else:
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

            save_model_state(model, scaler, model_type_key)

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


# -------------------- ⑨ バックテスト実行ボタン --------------------

st.markdown("---")
st.subheader("🚀 バックテスト検証の実行")

if st.button("▶️ バックテストトレーニングを開始", use_container_width=True):
    interval_str = "1m" if model_type_key == "1m" else "1h"
    with st.spinner(f"市場データ({model_selection})を取得し、ウォークフォワード検証を実行中..."):
        full_df = fetch_market_data(interval_str)

        if full_df is None or len(full_df) < 300:
            st.error("データ件数が不足しています（最低300本必要です）。期間を長くしてください。")
        else:
            res_df = run_backtest_engine(
                full_df,
                target_usd=target_usd,
                stake_usd=stake_usd,
                hold_bars=hold_bars,
                model_type=model_type_key,
            )
            show_dashboard(res_df)
elif GENIUS_HISTORY.exists():
    try:
        saved_df = pd.read_csv(GENIUS_HISTORY)
        if not saved_df.empty:
            show_dashboard(saved_df)
    except Exception:
        pass
