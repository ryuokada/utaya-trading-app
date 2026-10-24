import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LOG_FILE = "utaya_trading_log.csv"

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

def build_features(df):
    # 特徴量の作成（モメンタム、SMA乖離、ボラティリティ、リターン）
    df = df.copy()
    df['return'] = df['Close'].pct_change()
    df['momentum'] = df['Close'].diff(3)
    df['sma10'] = df['Close'].rolling(window=10).mean()
    df['deviation'] = df['Close'] - df['sma10']
    df['volatility'] = df['Close'].rolling(window=5).std()
    
    # ターゲット（次の1分足の終値が上か下か）
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

def check_first_touch(entry_price, future_prices, target_usd):
    """ハイローの先着タッチ判定（±10ドル or ±50ドルなどの目標に先に到達したか）"""
    upper = entry_price + target_usd
    lower = entry_price - target_usd
    for p in future_prices:
        if p >= upper:
            return "WIN"   # 上目標に先に到達
        if p <= lower:
            return "LOSE"  # 下目標に先に到達
    return "WAIT"          # 60秒以内に到達せず（判定不能・引き分け扱い）

def predict_single():
    print("\n[+] リアルタイム市場データを分析中...")
    df = fetch_btc_data(period="2d", interval="1m")
    if df is None:
        print("[-] データの取得に失敗しました。")
        return
    
    feature_df = build_features(df)
    features = ['return', 'momentum', 'deviation', 'volatility']
    
    X = feature_df[features].values
    y = feature_df['target'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled[:-1], y[:-1]) # 未来リークを防ぎ直前までで学習
    
    latest_X = scaler.transform(X[-1:])
    prob = model.predict_proba(latest_X)[0][1]
    current_price = float(df['Close'].iloc[-1])
    
    print(f"--------------------------------------------------")
    print(f" 💰 現在のBTC価格: ${current_price:,.2f}")
    print(f" 📈 上昇確率予測: {prob * 100:.2f}%")
    
    if prob > 0.58:
        print(f" 🚀 判定: 【 HIGH (上) 狙い 】 (信頼度超過)")
    elif prob < 0.42:
        print(f" 📉 判定: 【 LOW (下) 狙い 】 (信頼度超過)")
    else:
        print(f" ⚖️ 判定: 【 WAIT (見送り) 】 (優勢度不足)")
    print(f"--------------------------------------------------\n")

def run_backtest():
    print("\n[+] 過去データを用いたウォークフォワード検証（バックテスト）を開始します...")
    df = fetch_btc_data(period="5d", interval="1m")
    if df is None or len(df) < 200:
        print("[-] 検証に必要なデータが不足しています。")
        return
        
    feature_df = build_features(df)
    features = ['return', 'momentum', 'deviation', 'volatility']
    
    X = feature_df[features].values
    y = feature_df['target'].values
    prices = feature_df['Close'].values
    
    scaler = StandardScaler()
    model = LogisticRegression()
    
    wins = 0
    loses = 0
    waits = 0
    total_signals = 0
    
    # ウォークフォワード検証（時系列に沿って学習と予測を繰り返す）
    train_window = 100
    for i in range(train_window, len(X) - 5):
        X_train = X[:i]
        y_train = y[:i]
        
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        X_test = scaler.transform(X[i:i+1])
        prob = model.predict_proba(X_test)[0][1]
        
        # 信頼度閾値（58%以上でHIGH、42%以下でLOW、それ以外はWAIT）
        signal = 0
        if prob > 0.58:
            signal = 1
        elif prob < 0.42:
            signal = -1
        else:
            waits += 1
            continue
            
        total_signals += 1
        entry_price = prices[i]
        future_window = prices[i+1 : i+6] # 5分先までの値動き
        
        # 先着タッチ判定のテスト（例として±10ドル幅を採用）
        target_usd = 10.0
        upper = entry_price + target_usd
        lower = entry_price - target_usd
        
        outcome = "WAIT"
        for p in future_window:
            if signal == 1:
                if p >= upper: outcome = "WIN"; break
                if p <= lower: outcome = "LOSE"; break
            elif signal == -1:
                if p <= lower: outcome = "WIN"; break
                if p >= upper: outcome = "LOSE"; break
                
        if outcome == "WIN":
            wins += 1
        elif outcome == "LOSE":
            loses += 1
        else:
            waits += 1
            
    decided = wins + loses
    win_rate = (wins / decided * 100) if decided > 0 else 0.0
    breakeven = (1 / (1 + 0.88)) * 100
    
    print(f"\n==================================================")
    print(f" 📊 バックテスト検証結果レポート (±10ドル先着判定)")
    print(f"==================================================")
    print(f" 総シグナル回数: {total_signals + waits} 回")
    print(f" 有効エントリー: {decided} 回 (見送り: {waits} 回)")
    print(f" 勝利数 / 敗北数: {wins}勝 {loses}敗")
    print(f" 🎯 実効勝率 (signal_win_rate): {win_rate:.2f}%")
    print(f" 💡 理論損益分岐勝率 (breakeven): {breakeven:.2f}% (ペイアウト88%)")
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
    print(f"[+] 結果を {LOG_FILE} に保存しました。\n")

def run_monitoring():
    print("\n[+] 自動監視モードを開始します（Ctrl+C で終了できます）。")
    print("[+] 60秒ごとに市場データを取得し、シグナルを判定・記録します...\n")
    while True:
        try:
            df = fetch_btc_data(period="1d", interval="1m")
            if df is not None:
                feature_df = build_features(df)
                features = ['return', 'momentum', 'deviation', 'volatility']
                
                X = feature_df[features].values
                y = feature_df['target'].values
                
                scaler = StandardScaler()
                model = LogisticRegression()
                
                X_scaled = scaler.fit_transform(X[:-1])
                model.fit(X_scaled, y[:-1])
                
                latest_X = scaler.transform(X[-1:])
                prob = model.predict_proba(latest_X)[0][1]
                current_price = float(df['Close'].iloc[-1])
                
                signal = "WAIT"
                if prob > 0.58: signal = "HIGH"
                elif prob < 0.42: signal = "LOW"
                
                now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now_str}] 価格: ${current_price:,.2f} | 上昇確率: {prob*100:.1f}% | 判定: {signal}")
                
                # ログ記録
                log_data = pd.DataFrame([{
                    "timestamp": now_str,
                    "price": current_price,
                    "probability": prob,
                    "signal": signal
                }])
                header = not os.path.exists(LOG_FILE)
                log_data.to_csv(LOG_FILE, mode='a', header=header, index=False)
                
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n[+] 自動監視モードを終了します。")
            break
        except Exception as e:
            print(f"[-] 監視ループエラー: {e}")
            time.sleep(10)

def main():
    while True:
        print("==================================================")
        print(" ⚡ UTAYA AI HILO CLI v2 (検証・実データモデル)")
        print("==================================================")
        print(" 1. 1回だけ予測 (リアルタイム)")
        print(" 2. 過去データでバックテスト (ウォークフォワード検証)")
        print(" 3. 自動監視モード (ペーパートレード・シグナル記録)")
        print(" 4. 終了")
        print("==================================================")
        
        choice = input("番号を選んでください (1-4): ").strip()
        
        if choice == '1':
            predict_single()
        elif choice == '2':
            run_backtest()
        elif choice == '3':
            run_monitoring()
        elif choice == '4':
            print("終了します。")
            break
        else:
            print("[-] 無効な選択です。1〜4を入力してください。\n")

if __name__ == "__main__":
    main()
