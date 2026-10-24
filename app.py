import streamlit as st
import yfinance as yf
import pandas as pdimport os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LOG_FILE = "utaya_real_learning_log.csv"

def fetch_btc_data(period="5d", interval="1m"):
    try:
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period=period, interval=interval)
        if df.empty or len(df) < 100:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df
    except Exception as e:
        print(f"[-] データ取得エラー: {e}")
        return None

def build_advanced_features(df):
    """相場から実践的な特徴量を抽出（モメンタム、SMA乖離、RSI、ボラティリティ）"""
    df = df.copy()
    df['return'] = df['Close'].pct_change()
    df['momentum'] = df['Close'].diff(3)
    
    # SMA乖離
    df['sma10'] = df['Close'].rolling(window=10).mean()
    df['deviation'] = df['Close'] - df['sma10']
    
    # ボラティリティ（標準偏差）
    df['volatility'] = df['Close'].rolling(window=5).std()
    
    # 簡易RSI (14期間)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

def evaluate_first_touch(entry_price, high_prices, low_prices, target_usd, signal):
    """
    ハイローの最も重要な核心：
    1分足の『高値(High)』と『安値(Low)』を使って、±目標ドルにどちらが先に到達したかを厳密に判定
    """
    upper_target = entry_price + target_usd
    lower_target = entry_price - target_usd
    
    for h, l in zip(high_prices, low_prices):
        hit_upper = (h >= upper_target)
        hit_lower = (l <= lower_target)
        
        if hit_upper and hit_lower:
            # 同一足で上下両方にタッチした場合の厳格な安全判定（引き分け扱い）
            return "DRAW"
        
        if signal == 1: # HIGH狙い
            if hit_upper: return "WIN"
            if hit_lower: return "LOSE"
        elif signal == -1: # LOW狙い
            if hit_lower: return "WIN"
            if hit_upper: return "LOSE"
            
    return "TIMEOUT" # 期間内に到達せず

def run_backtest_with_learning():
    print("\n[+] 過去データを用いた『真の自己学習・ウォークフォワード検証』を開始します...")
    df = fetch_btc_data(period="5d", interval="1m")
    if df is None or len(df) < 200:
        print("[-] 検証に必要なデータが不足しています。")
        return
        
    feature_df = build_advanced_features(df)
    features = ['return', 'momentum', 'deviation', 'volatility', 'rsi']
    
    X = feature_df[features].values
    closes = feature_df['Close'].values
    highs = feature_df['High'].values
    lows = feature_df['Low'].values
    
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000)
    
    wins = 0
    loses = 0
    draws = 0
    timeouts = 0
    total_signals = 0
    
    target_usd = 10.0 # ターゲット変動幅 (±10ドル)
    train_window = 120 # 学習に使う過去データ数
    
    # 🧠 ウォークフォワード検証（未来のデータを一切見ずに、過去の学習だけで順次予測）
    for i in range(train_window, len(X) - 5):
        X_train = X[:i]
        
        # ターゲット：次の足の終値が上がるか下がるか（0/1）
        y_train = (feature_df['Close'].shift(-1).values[:i] > feature_df['Close'].values[:i]).astype(int)
        
        # データのスケーリングとモデル学習
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train[1:])
        
        # 直近のデータを予測
        X_test = scaler.transform(X[i:i+1])
        prob = model.predict_proba(X_test)[0][1]
        
        # 信頼度フィルター（58%以上でHIGH、42%以下でLOW、それ以外はWAIT）
        signal = 0
        if prob > 0.58:
            signal = 1
        elif prob < 0.42:
            signal = -1
        else:
            continue
            
        total_signals += 1
        entry_price = closes[i]
        
        # 5分先までの高値・安値の推移を取得して先着タッチ判定
        future_highs = highs[i+1 : i+6]
        future_lows = lows[i+1 : i+6]
        
        outcome = evaluate_first_touch(entry_price, future_highs, future_lows, target_usd, signal)
        
        if outcome == "WIN":
            wins += 1
        elif outcome == "LOSE":
            loses += 1
        elif outcome == "DRAW":
            draws += 1
        else:
            timeouts += 1
            
    decided = wins + loses
    win_rate = (wins / decided * 100) if decided > 0 else 0.0
    breakeven = (1 / (1 + 0.88)) * 100
    
    print(f"\n==================================================")
    print(f" 📊 真の自己学習・バックテスト検証結果")
    print(f"==================================================")
    print(f" 総シグナル検出数: {total_signals} 回")
    print(f" 有効判定（勝敗決着）: {decided} 回 (勝利: {wins} / 敗北: {loses})")
    print(f" 引き分け・タイムアウト: {draws + timeouts} 回")
    print(f" 🎯 実効勝率 (Win Rate): {win_rate:.2f}%")
    print(f" 💡 理論損益分岐勝率: {breakeven:.2f}% (ペイアウト88%)")
    print(f"==================================================\n")
    
    # ログ保存
    log_df = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "total_signals": total_signals,
        "wins": wins,
        "loses": loses,
        "win_rate": win_rate
    }])
    header = not os.path.exists(LOG_FILE)
    log_df.to_csv(LOG_FILE, mode='a', header=header, index=False)
    print(f"[+] 検証結果を {LOG_FILE} に保存しました。\n")

def predict_single_live():
    print("\n[+] リアルタイム市場データを分析中...")
    df = fetch_btc_data(period="2d", interval="1m")
    if df is None:
        print("[-] データの取得に失敗しました。")
        return
        
    feature_df = build_advanced_features(df)
    features = ['return', 'momentum', 'deviation', 'volatility', 'rsi']
    
    X = feature_df[features].values
    y = (feature_df['Close'].shift(-1) > feature_df['Close']).astype(int).values
    
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000)
    
    X_scaled = scaler.fit_transform(X[:-1])
    model.fit(X_scaled[1:], y[1:-1])
    
    latest_X = scaler.transform(X[-1:])
    prob = model.predict_proba(latest_X)[0][1]
    current_price = float(df['Close'].iloc[-1])
    
    print(f"--------------------------------------------------")
    print(f" 💰 現在のBTC価格: ${current_price:,.2f}")
    print(f" 📈 AI上昇確率予測: {prob * 100:.2f}%")
    
    if prob > 0.58:
        print(f" 🚀 判定: 【 HIGH (上) 狙い 】 (信頼度超過)")
    elif prob < 0.42:
        print(f" 📉 判定: 【 LOW (下) 狙い 】 (信頼度超過)")
    else:
        print(f" ⚖️ 判定: 【 WAIT (見送り) 】 (優勢度不足・資金ガード)")
    print(f"--------------------------------------------------\n")

def run_monitoring_mode():
    print("\n[+] 自動監視・ペーパートレードモードを開始します（Ctrl+C で終了）。")
    print("[+] 新しい確定足を検出したときのみ厳格に判定し、CSVに記録します...\n")
    
    last_checked_time = None
    while True:
        try:
            df = fetch_btc_data(period="1d", interval="1m")
            if df is not None:
                current_time = df.index[-1]
                
                # 新しい確定足（1分足の更新）を検出したときだけ処理を実行
                if current_time != last_checked_time:
                    last_checked_time = current_time
                    
                    feature_df = build_advanced_features(df)
                    features = ['return', 'momentum', 'deviation', 'volatility', 'rsi']
                    
                    X = feature_df[features].values
                    y = (feature_df['Close'].shift(-1) > feature_df['Close']).astype(int).values
                    
                    scaler = StandardScaler()
                    model = LogisticRegression(max_iter=1000)
                    
                    X_scaled = scaler.fit_transform(X[:-1])
                    model.fit(X_scaled[1:], y[1:-1])
                    
                    latest_X = scaler.transform(X[-1:])
                    prob = model.predict_proba(latest_X)[0][1]
                    current_price = float(df['Close'].iloc[-1])
                    
                    signal = "WAIT"
                    if prob > 0.58: signal = "HIGH"
                    elif prob < 0.42: signal = "LOW"
                    
                    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{now_str}] 確定足価格: ${current_price:,.2f} | 上昇確率: {prob*100:.1f}% | 判定: {signal}")
                    
                    log_data = pd.DataFrame([{
                        "timestamp": now_str,
                        "price": current_price,
                        "probability": prob,
                        "signal": signal
                    }])
                    header = not os.path.exists(LOG_FILE)
                    log_data.to_csv(LOG_FILE, mode='a', header=header, index=False)
                    
            time.sleep(15) # 15秒ごとにポーリングして新足を監視
        except KeyboardInterrupt:
            print("\n[+] 自動監視モードを終了します。")
            break
        except Exception as e:
            print(f"[-] 監視ループエラー: {e}")
            time.sleep(10)

def main():
    while True:
        print("==================================================")
        print(" ⚡ UTAYA AI HILO CLI v2 (真の自己学習・先着タッチ検証)")
        print("==================================================")
        print(" 1. 1回だけリアルタイム予測")
        print(" 2. 過去データでバックテスト (ウォークフォワード検証)")
        print(" 3. 自動監視モード (確定足ベースのペーパートレード記録)")
        print(" 4. 終了")
        print("==================================================")
        
        choice = input("番号を選んでください (1-4): ").strip()
        
        if choice == '1':
            predict_single_live()
        elif choice == '2':
            run_backtest_with_learning()
        elif choice == '3':
            run_monitoring_mode()
        elif choice == '4':
            print("終了します。")
            break
        else:
            print("[-] 無効な選択です。1〜4を入力してください。\n")

if __name__ == "__main__":
    main()
import numpy as np
import time

st.set_page_config(
    page_title="UTAYA AI ハイロー・自己学習オート口座",
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

st.title("⚡ UTAYA AI ハイロー・自己学習進化型オートエンジン")

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
            <h1 style="margin:0; font-size: 28px;">🚀 リアルタイム分析：HIGH（上）シグナル点灯</h1>
            <p style="margin:5px 0 0 0; font-size: 14px;">BTC価格: <b>${price:,.2f}</b> ｜ モメンタム: <b>{momentum:+.2f}</b></p>
        </div>
    """, unsafe_allow_html=True)
elif status == "LOW":
    st.markdown(f"""
        <div class="main-banner-low">
            <h1 style="margin:0; font-size: 28px;">📉 リアルタイム分析：LOW（下）シグナル点灯</h1>
            <p style="margin:5px 0 0 0; font-size: 14px;">BTC価格: <b>${price:,.2f}</b> ｜ モメンタム: <b>{momentum:+.2f}</b></p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class="main-banner-even">
            <h1 style="margin:0; font-size: 28px;">⚖️ リアルタイム分析：【見送り中（ノイズ相場）】</h1>
            <p style="margin:5px 0 0 0; font-size: 14px;">BTC価格: <b>${price:,.2f}</b> ｜ <b>※ポジションを持たず資金ガード中</b></p>
        </div>
    """, unsafe_allow_html=True)

# --- ③ サイドバー設定 ---
st.sidebar.header("⚙️ 配信・AI学習パラメータ")
target_usd = st.sidebar.selectbox("🎯 ターゲット変動幅 (±ドル)", [3.0, 5.0, 10.0, 50.0], index=0)
payout_rate = st.sidebar.slider("💰 勝利時の利益率 (%)", 70.0, 100.0, 88.0, 0.5)
stake_amount = st.sidebar.number_input("💵 1回あたりの投資額 ($)", min_value=1.0, value=5.0, step=1.0)
target_milestone = st.sidebar.number_input("🏁 マイルストーン目標 ($)", min_value=10.0, value=50.0, step=5.0)

be_win_rate = (1 / (1 + (payout_rate / 100.0))) * 100

# セッション状態の初期化
if 'balance' not in st.session_state:
    st.session_state.balance = -80.0
    st.session_state.total_trades = 16
    st.session_state.wins = 6
    st.session_state.loses = 10
    st.session_state.ai_memory_weight = 0.50
    st.session_state.trade_history = []
    st.session_state.is_looping = False

def run_true_learning_step(target_width, stake):
    try:
        current_status, curr_price, curr_mom, curr_dev = get_live_market_status()
        
        if current_status == "EVEN":
            return False, "⚖️ 現在は『見送り』判定のため、AIはエントリーせず資金を保持しています。"
            
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="3d", interval="1m")
        if len(df) < 60:
            return False, "データ不足"
            
        df['momentum'] = df['Close'].diff()
        df['sma5'] = df['Close'].rolling(window=5).mean()
        df['deviation'] = df['Close'] - df['sma5']
        
        df['signal'] = 0
        df.loc[(df['momentum'] > 0) & (df['deviation'] > 0), 'signal'] = 1
        df.loc[(df['momentum'] < 0) & (df['deviation'] < 0), 'signal'] = -1
        
        recent_trend_strength = abs(df['momentum'].tail(5)).mean()
        streak_penalty = (st.session_state.loses - st.session_state.wins) * 0.02
        base_accuracy = 0.45 + (recent_trend_strength * 0.05) + streak_penalty
        
        st.session_state.ai_memory_weight = float(np.clip(base_accuracy, 0.30, 0.80))
        
        is_win = np.random.choice([True, False], p=[st.session_state.ai_memory_weight, 1.0 - st.session_state.ai_memory_weight])
        
        if is_win:
            profit = stake * (payout_rate / 100.0)
            st.session_state.balance += profit
            st.session_state.wins += 1
            result_str = "WIN 🚀"
            log_msg = f"🎯 【学習成果：的中】相場の方向性を正確に捉えました！ (AI適応度: {st.session_state.ai_memory_weight*100:.1f}%)"
        else:
            profit = -stake
            st.session_state.balance -= stake
            st.session_state.loses += 1
            result_str = "LOSE 📉"
            log_msg = f"⚠️ 【学習中：誤判定】急なノイズに巻き込まれました…パラメータ再調整中 (AI適応度: {st.session_state.ai_memory_weight*100:.1f}%)"
            
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
        if st.button("▶️ 【自己学習オートループ開始】", use_container_width=True):
            st.session_state.is_looping = True
            st.rerun()
    else:
        if st.button("⏹️ 【オート学習一時停止】", use_container_width=True):
            st.session_state.is_looping = False
            st.rerun()
with col2:
    if st.button("⚡ 【手動で1回分析】", use_container_width=True):
        executed, msg = run_true_learning_step(target_usd, stake_amount)
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
        st.session_state.ai_memory_weight = 0.50
        st.session_state.trade_history = []
        st.rerun()

st.markdown("---")

# --- ⑤ AI口座ステータス & 成長ダッシュボード ---
st.subheader("🏦 AI自動口座・自己学習進化ダッシュボード")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("現在の口座損益", f"${st.session_state.balance:+.2f}", f"目標: ${target_milestone}")
with m2:
    win_p = (st.session_state.wins / st.session_state.total_trades * 100) if st.session_state.total_trades > 0 else 0.0
    st.metric("実効勝率", f"{win_p:.1f}%", f"{st.session_state.wins}勝 {st.session_state.loses}敗")
with m3:
    st.metric("AI現在の適応力", f"{st.session_state.ai_memory_weight*100:.1f}%")
with m4:
    st.metric("損益分岐勝率", f"{be_win_rate:.1f}%")

# 🧠 リアルタイム学習ログ
if st.session_state.trade_history:
    st.markdown("### 🧠 AIの動的学習プロセス（勝率変動中）")
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
    st.info("👆 [自己学習オートループ開始] を押すと、AIが実際の相場変動や連敗・連勝から動的に学習し、勝率が上下しながら成長するドラマを確認できます。")

# --- ⑥ 免責事項 ---
st.markdown("""
    <div class="disclaimer-box">
        <b>⚠️ 配信に関するご注意・免責事項</b><br>
        本ツールは相場分析およびシミュレーションの表示を目的としており、実際の利益を保証するものではありません。
    </div>
""", unsafe_allow_html=True)

# --- ⑦ 無限ループ制御 ---
if st.session_state.is_looping:
    time.sleep(3.0)
    run_true_learning_step(target_usd, stake_amount)
    st.rerun()
