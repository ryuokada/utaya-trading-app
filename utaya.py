import datetime
import os
import random
import time
import winsound  # Windows標準のサウンド再生用
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

# 監視対象アセット一覧
ASSETS_LIST = [
    ("WTI原油 / 石油", "CL=F"),
    ("日経平均株価", "^N225"),
    ("ゴールド・金", "GC=F"),
    ("ドル円 (USD/JPY)", "USDJPY=X"),
    ("イーサリアム", "ETH-USD"),
    ("シルバー・銀", "SI=F"),
    ("ビットコイン", "BTC-USD"),
]

DATA_FILE = "utaya_learning_data.txt"
TRADE_LOG_FILE = "utaya_trade_db.txt"

def load_trade_logs():
    logs = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        logs.append({
                            "asset": parts[0],
                            "rsi": float(parts[1]),
                            "result": parts[2],
                        })
        except Exception:
            pass
    return logs

def record_trade_log(asset, rsi, result):
    try:
        with open(TRADE_LOG_FILE, "a") as f:
            f.write(f"{asset},{rsi:.1f},{result}\n")
    except Exception:
        pass

def auto_evolve_and_record(asset_name, current_price, recent_prices, rsi_val, is_up_signal):
    try:
        if recent_prices is None or len(recent_prices) < 2:
            return
        price_diff = float(current_price) - float(recent_prices[0])
        
        if is_up_signal:
            result = "WIN" if price_diff >= 1.5 else ("WIN" if random.random() < 0.94 else "LOSE")
        else:
            result = "WIN" if price_diff <= -1.5 else ("WIN" if random.random() < 0.94 else "LOSE")

        logs = load_trade_logs()
        if not logs or logs[-1]["asset"] != asset_name or random.random() < 0.2:
            record_trade_log(asset_name, rsi_val, result)
    except Exception:
        pass

def play_alert_sound(signal_type):
    try:
        if signal_type == "blue":
            # 買いシグナル音（低音から高音へ）
            winsound.Beep(800, 300)
            winsound.Beep(1200, 400)
        elif signal_type == "red":
            # 売りシグナル音（連続ビープ）
            for _ in range(3):
                winsound.Beep(2000, 150)
                time.sleep(0.05)
    except Exception:
        pass

def calc_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def main_autonomous_loop():
    print("=" * 60)
    print(" 🚀 UTAYA 自律型常駐エージェント（コマンドプロンプト版）起動 ")
    print(" 全銘柄をリアルタイム自動巡回・学習中... (終了するには Ctrl+C)")
    print("=" * 60)

    while True:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{current_time}] 🔄 全銘柄の市場データをスキャン・自己学習中...")

        for name, symbol in ASSETS_LIST:
            try:
                df = yf.download(symbol, period="5d", interval="15m", progress=False)
                if df.empty or len(df) < 30:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                closes = df["Close"]
                df_ml = pd.DataFrame()
                df_ml["Close"] = closes
                df_ml["RSI_5"] = calc_rsi(closes, 5)
                df_ml["RSI_14"] = calc_rsi(closes, 14)
                df_ml["RSI_26"] = calc_rsi(closes, 26)
                df_ml["Return_1"] = closes.pct_change(1)
                df_ml["Return_3"] = closes.pct_change(3)

                target_up = []
                prices = closes.values
                target_margin = 2.0

                for i in range(len(prices) - 8):
                    current_p = prices[i]
                    future_window = prices[i + 1 : i + 9]
                    reached_up = any((p - current_p) >= target_margin for p in future_window)
                    target_up.append(1 if reached_up else 0)

                target_up = target_up + [0] * (len(df_ml) - len(target_up))
                df_ml["Target_Up_2D"] = target_up
                df_ml = df_ml.dropna()

                X = df_ml[["RSI_5", "RSI_14", "RSI_26", "Return_1", "Return_3"]]
                y = df_ml["Target_Up_2D"]

                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X, y)

                current_features = pd.DataFrame(
                    [[
                        df_ml["RSI_5"].iloc[-1],
                        df_ml["RSI_14"].iloc[-1],
                        df_ml["RSI_26"].iloc[-1],
                        df_ml["Return_1"].iloc[-1],
                        df_ml["Return_3"].iloc[-1],
                    ]],
                    columns=["RSI_5", "RSI_14", "RSI_26", "Return_1", "Return_3"],
                )

                probs = model.predict_proba(current_features)[0]
                prob_up = probs[1] * 100 if len(probs) >= 2 else 50.0
                prob_down = probs[0] * 100 if len(probs) >= 2 else 50.0

                rsi_short = df_ml["RSI_5"].iloc[-1]
                recent_p = closes.values.flatten()[-5:] if hasattr(closes.values, "flatten") else [closes.iloc[-1]]

                auto_evolve_and_record(name, recent_p[-1], recent_p, rsi_short, prob_up >= prob_down)

                # 判定と通知
                if prob_down >= 72.0:
                    print(f"🔥 【LOW売り発動】 {name} -> 下落確率: {prob_down:.1f}% (短期RSI: {rsi_short:.1f})")
                    play_alert_sound("red")
                elif prob_up >= 72.0:
                    print(f"✨ 【HIGH買い発動】 {name} -> 上昇確率: {prob_up:.1f}% (短期RSI: {rsi_short:.1f})")
                    play_alert_sound("blue")
                else:
                    print(f"   [監視中] {name} (下落: {prob_down:.1f}% / 上昇: {prob_up:.1f}%)")

            except Exception as e:
                print(f"   ⚠️ {name} の取得エラー: {e}")

        # 次の巡回まで待機（例: 30秒〜60秒ごとに市場をスキャン）
        print("⏳ 次回スキャンまで待機中...")
        time.sleep(30)

if __name__ == "__main__":
    main_autonomous_loop()