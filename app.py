import streamlit as st
import yfinance as yf
import time

st.set_page_config(page_title="最強ポチポチUTAYA", layout="centered")

st.title("🌌 最強ポチポチUTAYA")
st.write("ボタンを10回ポチポチ連打して、ダマシを粉砕する『当たり宇宙』を顕現させろ！")

# セッション状態の初期化
if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.high_count = 0
    st.session_state.low_count = 0
    st.session_state.last_price = None
    st.session_state.history = []

def get_btc_price():
    try:
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="1d", interval="1m")
        if not df.empty:
            return float(df['Close'].iloc[-1])
    except:
        pass
    return None

# 現在価格の取得
current_price = get_btc_price()

if current_price is not None:
    st.metric(label="💰 現在のBTC価格 (USDT)", value=f"${current_price:,.2f}")
    
    # 初回セット
    if st.session_state.last_price is None:
        st.session_state.last_price = current_price

    # 巨大なポチポチボタン
    if st.button("🔴 【10回観測：真実の当たり宇宙を確定】", use_container_width=True):
        diff = current_price - st.session_state.last_price
        
        if diff >= 0:
            res = "HIGH"
            st.session_state.high_count += 1
        else:
            res = "LOW"
            st.session_state.low_count += 1
            
        st.session_state.count += 1
        st.session_state.history.append(f"{st.session_state.count}回目: ${current_price:,.2f} [{res}]")
        st.session_state.last_price = current_price

    st.markdown("---")
    st.subheader(f"🔬 観測進捗: 【 {st.session_state.count} / 10 回完了 】")
    
    # 履歴の表示
    for h in st.session_state.history:
        st.text(h)

    # 10回達成時の結果表示
    if st.session_state.count >= 10:
        st.markdown("### 🌟 10回観測完了：統計的真実の顕現")
        st.info(f"📊 統計内訳 ── HIGH: {st.session_state.high_count}回 / LOW: {st.session_state.low_count}回")
        
        if st.session_state.high_count > st.session_state.low_count:
            st.success("🚀🚀 【最終判定：『HIGHの当たり宇宙』が完全顕現！】\n\nクジラのダマシを統計で粉砕！『上』の現実が確定しました。")
        elif st.session_state.low_count > st.session_state.high_count:
            st.error("📉📉 【最終判定：『LOWの当たり宇宙』が完全顕現！】\n\nクジラのダマシを統計で粉砕！『下』の現実が確定しました。")
        else:
            st.warning("⚖️ 【同点】もう一度10回観測して宇宙を偏らせろ！")
            
        # リセット
        st.session_state.count = 0
        st.session_state.high_count = 0
        st.session_state.low_count = 0
        st.session_state.history = []
        st.balloons()
else:
    st.error("⚠️ 価格データの取得中... 少し待って再読み込みしてください。")
