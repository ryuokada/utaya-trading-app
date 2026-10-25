import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="UTAYA BTC HILO - Web Dashboard",
    page_icon="⚡",
    layout="wide"
)

# --- 🎨 カスタムCSS ---
st.markdown("""
<style>
    .main-banner-high {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-banner-low {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-banner-even {
        background: linear-gradient(135deg, #355c7d, #6c5b7b);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .log-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
        font-family: monospace;
        color: #58a6ff;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ UTAYA BTC HILO - Web シミュレーター")

SYMBOL = "BTC-USD"
TARGET_USD = 50.0
PAYOUT_RATE = 0.88

# --- ① データ取得関数 ---
@st.cache_data(ttl=60)
def get_web_data(period="7d"):
    try:
        df = yf.download(SYMBOL, period=period, interval="1m", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].copy()
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        return df.sort_index()
    except:
        return None

# --- ② シグナル判定 ---
def make_signal(df, i):
    close = df["Close"].iloc[:i+1]
    if len(close) < 30:
        return "SKIP"
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    sma10 = float(close.tail(10).mean())
    sma30 = float(close.tail(30).mean())
    
    momentum = last - prev
    deviation = last - sma10

    if momentum > 0 and last > sma30 and deviation > 0:
        return "HIGH"
    if momentum < 0 and last < sma30 and deviation < 0:
        return "LOW"
    return "SKIP"

# --- ③ 先着判定 ---
def judge_first_touch(entry, direction, future, target=TARGET_USD):
    upper = entry + target
    lower = entry - target
    for _, candle in future.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])
        hit_up = high >= upper
        hit_down = low <= lower

        if hit_up and hit_down:
            return "AMBIGUOUS", "同値足で両方到達"
        if hit_up:
            return ("WIN" if direction == "HIGH" else "LOSS"), "上目標到達"
        if hit_down:
            return ("WIN" if direction == "LOW" else "LOSS"), "下目標到達"
    return "NO_TOUCH", "未到達"

# --- ④ サイドバー設定 ---
st.sidebar.header("⚙️ 検証パラメータ")
target_usd = st.sidebar.slider("🎯 目標値幅 (±USD)", 10.0, 100.0, 50.0, 5.0)
stake_usd = st.sidebar.number_input("💵 1回あたりの賭け金 ($)", min_value=1.0, value=5.0, step=1.0)
hold_bars = st.sidebar.slider("⏱️ 判定足数 (分)", 1, 5, 1)

# セッション状態
if 'running' not in st.session_state:
    st.session_state.running = False

# --- ⑤ メイン画面のリアルタイム表示 ---
df = get_web_data("1d")
if df is not None and len(df) > 30:
    last_price = float(df['Close'].iloc[-1])
    current_signal = make_signal(df, len(df)-2)
    
    if current_signal == "HIGH":
        st.markdown(f'<div class="main-banner-high"><h2>🚀 リアルタイムシグナル：HIGH（上狙い）</h2><p>BTC価格: ${last_price:,.2f}</p></div>', unsafe_allow_html=True)
    elif current_signal == "LOW":
        st.markdown(f'<div class="main-banner-low"><h2>📉 リアルタイムシグナル：LOW（下狙い）</h2><p>BTC価格: ${last_price:,.2f}</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="main-banner-even"><h2>⚖️ リアルタイムシグナル：見送り (SKIP)</h2><p>BTC価格: ${last_price:,.2f}</p></div>', unsafe_allow_html=True)
else:
    st.warning("市場データを読み込んでいます...")

st.markdown("---")

# --- ⑥ バックテスト実行セクション ---
st.subheader("📊 過去データによるバックテスト検証")

if st.button("▶️ バックテストを実行する", use_container_width=True):
    with st.spinner("BTCの1分足データを取得して検証中..."):
        full_df = get_web_data("7d")
        if full_df is not None:
            records = []
            for i in range(30, len(full_df) - hold_bars):
                direction = make_signal(full_df, i)
                if direction == "SKIP":
                    continue
                entry = float(full_df["Close"].iloc[i])
                future = full_df.iloc[i + 1 : i + 1 + hold_bars]
                result, reason = judge_first_touch(entry, direction, future, target_usd)
                
                if result == "WIN":
                    pnl = stake_usd * PAYOUT_RATE
                elif result == "LOSS":
                    pnl = -stake_usd
                else:
                    pnl = 0.0
                
                records.append({
                    "time": full_df.index[i],
                    "entry": entry,
                    "direction": direction,
                    "result": result,
                    "pnl": pnl
                })
            
            res_df = pd.DataFrame(records)
            if not res_df.empty:
                wins = int((res_df["result"] == "WIN").sum())
                losses = int((res_df["result"] == "LOSS").sum())
                decided = wins + losses
                win_rate = (wins / decided * 100) if decided > 0 else 0.0
                total_pnl = float(res_df["pnl"].sum())
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("総シグナル数", len(res_df))
                c2.metric("勝率", f"{win_rate:.1f}%", f"{wins}勝 {losses}敗")
                c3.metric("損益合計", f"${total_pnl:+.2f}")
                c4.metric("対象期間", "直近7日間")
                
                st.dataframe(res_df.head(20), use_container_width=True)
            else:
                st.info("条件に一致するシグナルが見つかりませんでした。")
        else:
            st.error("データの取得に失敗しました。")

# --- ⑦ 免責事項 ---
st.markdown("""
<div style="background-color: #1e1e1e; border-left: 4px solid #ffcc00; padding: 12px; border-radius: 5px; font-size: 13px; color: #cccccc; margin-top: 30px;">
    <b>⚠️ ご注意</b><br>
    本ツールはブラウザ上で動作する検証用シミュレーターです。実際の売買や投資を勧誘するものではありません。
</div>
""", unsafe_allow_html=True)
