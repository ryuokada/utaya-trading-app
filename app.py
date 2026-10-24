import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="UTAYA AI ハイロー・50$突破サバイバル口座",
    page_icon="⚡",
    layout="wide"
)

# --- 🎨 カスタムCSS：サイバーデザイン ---
st.markdown("""
<style>
    .main-banner-high {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 176, 155, 0.4);
        margin-bottom: 20px;
    }
    .main-banner-low {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
        margin-bottom: 20px;
    }
    .main-banner-even {
        background: linear-gradient(135deg, #355c7d, #6c5b7b);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(108, 91, 123, 0.4);
        margin-bottom: 20px;
    }
    .ai-log-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
        font-family: monospace;
        color: #58a6ff;
        margin-bottom: 20px;
    }
    .disclaimer-box {
        background-color: #1e1e1e;
        border-left: 4px solid #ffcc00;
        padding: 12px;
        border-radius: 5px;
        font-size: 13px;
        color: #cccccc;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ UTAYA AI 50$突破・サバイバルオートエンジン")

# --- ① リアルタイム判定エンジン ---
def get_live_market_status():
    try:
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="1d", interval="1m")
        if len(df) < 10:
            return "EVEN", 0.0, 0.0, 0.0
        
        p1 = float(df['Close'].iloc[-1])
        p2 = float(df['Close'].iloc[-2])
        mom = p1 - p2
        sma = float(df['Close'].iloc[-5:].mean())
        dev = p1 - sma
        
        if mom > 0 and dev >= 0:
            return "HIGH", p1, mom, dev
        elif mom < 0 and dev <= 0:
            return "LOW", p1, mom, dev
        else:
            return "EVEN", p1, mom, dev
    except:
        return "EVEN", 0.0, 0.0, 0.0

status, price, momentum, deviation = get_live_market_status()

# --- ② 画面最上部：現在のリアルタイム分析 ---
if status == "HIGH":
    st.markdown(f"""
        <div class="main-banner-high">
            <h1 style="margin:0; font-size: 26px;">🚀 リアルタイム分析：HIGH（上）チャンス到来</h1>
            <p style="margin:5px 0 0 0; font-size: 14px;">BTC価格: <b>${price:,.2f}</b> ｜ 初動モメンタム: <b>{momentum:+.2f}</b></p>
        </div>
    """, unsafe_allow_html=True)
elif status == "LOW":
    st.markdown(f"""
        <div class="main-banner-low">
            <h1 style="margin:0; font-size: 26px;">📉 リアルタイム分析：LOW（下）チャンス到来</h1>
            <p style="margin:5px 0 0 0; font-size: 14px;">BTC価格: <b>${price:,.2f}</b> ｜ 初動モメンタム: <b>{momentum:+.2f}</b></p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class="main-banner-even">
            <h1 style="margin:0; font-size: 26px;">⚖️ リアルタイム分析：【見送り中（ノイズ相場）】</h1>
            <p style="margin:5px 0 0 0; font-size: 14px;">BTC価格: <b>${price:,.2f}</b> ｜ <b>※3〜5$の初動が出るまで資金ガード中</b></p>
        </div>
    """, unsafe_allow_html=True)

# --- ③ サイドバー設定 ---
st.sidebar.header("⚙️ サバイバルチャレンジパラメータ")
initial_step_usd = st.sidebar.selectbox("🎯 第1段階の初動目標 (±ドル)", [3.0, 5.0, 10.0], index=1)
final_goal_usd = st.sidebar.number_input("🏁 最終勝利ゴール (±50$以上)", min_value=30.0, value=50.0, step=5.0)
payout_rate = st.sidebar.slider("💰 50$達成時の利益率 (%)", 70.0, 150.0, 100.0, 5.0)
stake_amount = st.sidebar.number_input("💵 挑戦ごとの投資額 ($)", min_value=1.0, value=5.0, step=1.0)

# セッション状態の初期化
if 'balance' not in st.session_state:
    st.session_state.balance = -80.0
    st.session_state.total_trades = 16
    st.session_state.wins = 6
    st.session_state.loses = 10
    st.session_state.ai_memory_weight = 0.45
    st.session_state.trade_history = []
    st.session_state.is_looping = False

def run_survival_step(step_target, final_target, stake):
    try:
        current_status, curr_price, curr_mom, curr_dev = get_live_market_status()
        
        if current_status == "EVEN":
            return False, "⚖️ 初動の勢い（3〜5$）が見られないため、AIは見送り（待機）しています。"
            
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="2d", interval="1m")
        if len(df) < 30:
            return False, "データ不足"
            
        # 実際の価格推移データから「初動3〜5$を突破し、さらに50$ゴールに到達したか」を判定
        entry_price = float(df['Close'].iloc[-10]) # 少し前の価格をエントリー仮定
        future_prices = df['Close'].tail(8).values
        
        # AIの適応力（連敗すると慎重になり、トレンドを厳しく選別する）
        streak_diff = st.session_state.wins - st.session_state.loses
        ai_accuracy = float(np.clip(0.42 + (streak_diff * 0.015), 0.30, 0.75))
        st.session_state.ai_memory_weight = ai_accuracy
        
        # サバイバル判定シミュレーション
        # ① まず第1段階（3〜5$）の初動をクリアできるか
        first_step_cleared = False
        final_goal_reached = False
        
        max_diff = 0.0
        for p in future_prices:
            diff = p - entry_price if current_status == "HIGH" else entry_price - p
            if diff > max_diff:
                max_diff = diff
                
        # 初動をクリアし、さらにAIの確率判定に合格した場合にゴール達成
        if max_diff >= step_target:
            if max_diff >= final_target and np.random.choice([True, False], p=[ai_accuracy, 1.0 - ai_accuracy]):
                final_goal_reached = True
                
        if final_goal_reached:
            profit = stake * (payout_rate / 100.0)
            st.session_state.balance += profit
            st.session_state.wins += 1
            result_str = "WIN 🏆"
            log_msg = f"🚀 【50$突破成功！】初動({step_target}$)をクリア後、見事に最終ゴール({final_target}$)へ到達！ (適応力: {ai_accuracy*100:.1f}%)"
        else:
            profit = -stake
            st.session_state.balance -= stake
            st.session_state.loses += 1
            result_str = "LOSE 💥"
            log_msg = f"📉 【途中失速・敗北】初動は動いたものの50$に届かずノイズに巻き戻されました…再学習中"
            
        st.session_state.total_trades += 1
        st.session_state.trade_history.insert(0, {
            "result": result_str,
            "amount": f"${profit:+.2f}",
            "balance": f"${st.session_state.balance:+.2f}",
            "msg": log_msg
        })
        
        return True, log_msg
    except Exception as e:
        return False, str(e)

# --- ④ 操作コントロール ---
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    if not st.session_state.is_looping:
        if st.button("▶️ 【50$突破サバイバル・オート開始】", use_container_width=True):
            st.session_state.is_looping = True
            st.rerun()
    else:
        if st.button("⏹️ 【オート一時停止】", use_container_width=True):
            st.session_state.is_looping = False
            st.rerun()
with col2:
    if st.button("⚡ 【手動で1回サバイバル挑戦】", use_container_width=True):
        executed, msg = run_survival_step(initial_step_usd, final_goal_usd, stake_amount)
        if not executed:
            st.info(msg)
        st.rerun()
with col3:
    if st.button("🔄 リセット"):
        st.session_state.is_looping = False
        st.session_state.balance = -80.0
        st.session_state.total_trades = 16
        st.session_state.wins = 6
        st.session_state.loses = 10
        st.session_state.ai_memory_weight = 0.45
        st.session_state.trade_history = []
        st.rerun()

st.markdown("---")

# --- ⑤ ダッシュボード ---
st.subheader("🏦 サバイバル口座・成長進化ダッシュボード")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("現在の口座損益", f"${st.session_state.balance:+.2f}", f"目標ゴール: ${final_goal_usd}")
with m2:
    win_p = (st.session_state.wins / st.session_state.total_trades * 100) if st.session_state.total_trades > 0 else 0.0
    st.metric("実効勝率", f"{win_p:.1f}%", f"{st.session_state.wins}勝 {st.session_state.loses}敗")
with m3:
    st.metric("AI現在の適応力", f"{st.session_state.ai_memory_weight*100:.1f}%")
with m4:
    st.metric("第1段階初動", f"±{initial_step_usd}$")

# 🧠 リアルタイムログ
if st.session_state.trade_history:
    st.markdown("### 🧠 50$突破チャレンジ・リアルタイムログ")
    latest = st.session_state.trade_history[0]
    st.markdown(f"""
        <div class="ai-log-box">
            <b>最新結果: {latest['result']}（損益: {latest['amount']}）</b><br>
            {latest['msg']}<br>
            <small>口座残高: {latest['balance']}</small>
        </div>
    """, unsafe_allow_html=True)
    
    for h in st.session_state.trade_history[1:5]:
        st.text(f"[{h['result']}] 損益: {h['amount']} ｜ 口座: {h['balance']} ｜ {h['msg']}")
else:
    st.info("👆 [50$突破サバイバル・オート開始] を押すと、初動3〜5$をクリアして50$ゴールを目指す本格的なサバイバル検証が始まります。")

# --- ⑥ 免責事項 ---
st.markdown("""
    <div class="disclaimer-box">
        <b>⚠️ 配信に関するご注意・免責事項</b><br>
        本ツールは相場分析およびシミュレーションの表示を目的としており、実際の利益を保証するものではありません。
    </div>
""", unsafe_allow_html=True)

# --- ⑦ ループ制御 ---
if st.session_state.is_looping:
    time.sleep(3.0)
    run_survival_step(initial_step_usd, final_goal_usd, stake_amount)
    st.rerun()
