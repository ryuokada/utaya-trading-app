import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

# 監視対象銘柄
TICKER = "GC=F"  # ゴールド・金
ASSET_NAME = "ゴールド・金 (GC=F)"
TARGET_MARGIN = 2.0  # 2ドル利確ターゲット

print(f"🚀 UTAYA ローカル常駐型リアルタイム予測エンジン起動: {ASSET_NAME}")
print("🛡️ 秒単位で市場データを自律スキャン中... 待機状態 (無音)")

last_signal = "none"

while True:
  try:
    # 1. 最新の市場データを取得
    df = yf.download(TICKER, period="5d", interval="15m", progress=False)
    if not df.empty and len(df) >= 30:
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      closes = df["Close"]

      # RSI計算関数
      def calc_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

      df_ml = pd.DataFrame()
      df_ml["Close"] = closes
      df_ml["RSI_5"] = calc_rsi(closes, 5)
      df_ml["RSI_14"] = calc_rsi(closes, 14)
      df_ml["RSI_26"] = calc_rsi(closes, 26)
      df_ml["Return_1"] = closes.pct_change(1)
      df_ml["Return_3"] = closes.pct_change(3)

      # ターゲット設定
      target_up = []
      prices = closes.values
      for i in range(len(prices) - 8):
        current_p = prices[i]
        future_window = prices[i + 1 : i + 9]
        reached_up = any((p - current_p) >= TARGET_MARGIN for p in future_window)
        target_up.append(1 if reached_up else 0)

      target_up = target_up + [0] * (len(df_ml) - len(target_up))
      df_ml["Target_Up_2D"] = target_up
      df_ml = df_ml.dropna()

      X = df_ml[["RSI_5", "RSI_14", "RSI_26", "Return_1", "Return_3"]]
      y = df_ml["Target_Up_2D"]

      # 機械学習モデルの訓練
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
      prob_up = probs[1] * 100
      prob_down = probs[0] * 100
      rsi_val = df_ml["RSI_5"].iloc[-1]

      # 2. 判定と出力（目まぐるしく変化するライブ数値）
      print(
          f"📊 [LIVEスキャン] RSI(5): {rsi_val:.1f} | 下落確率: {prob_down:.1f}%"
          f" | 上昇確率: {prob_up:.1f}%"
      )

      # 3. 適切なエントリーポイント（確率70%以上）に達した時のアクション
      if prob_down >= 70.0 and last_signal != "red":
        print(
            "\n🔥🔥🔥 【UTAYAベスト発動：LOW売りエントリー】 🔥🔥🔥\n"
            f"銘柄: {ASSET_NAME} | 下落確率: {prob_down:.1f}%\n"
        )
        # Windowsのシステムビープ音（ポキュン！に近いアラート音）を鳴らす
        try:
          import winsound

          winsound.Beep(1200, 300)  # 高音
          winsound.Beep(1800, 500)  # ポキュン！
        except Exception:
          print("\a")  
        last_signal = "red"

      elif prob_up >= 70.0 and last_signal != "blue":
        print(
            "\n✨✨✨ 【UTAYAベスト発動：HIGH買いエントリー】 ✨✨✨\n"
            f"銘柄: {ASSET_NAME} | 上昇確率: {prob_up:.1f}%\n"
        )
        try:
          import winsound

          winsound.Beep(800, 300)
          winsound.Beep(1200, 500)
        except Exception:
          print("\a")
        last_signal = "blue"

      elif prob_down < 60.0 and prob_up < 60.0:
        last_signal = "none"

    # 5秒ごとにライブ更新（数字が目まぐるしく動き続けるループ）
    time.sleep(5)

  except Exception as e:
    print(f"⚠️ 処理ループエラー: {e}")
    time.sleep(5)
