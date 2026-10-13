import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf

# Streamlitのページ設定
st.set_page_config(
    page_title="UTAYA リアルタイム予測", page_icon="🚀", layout="wide"
)

TICKER = "GC=F"  # ゴールド・金
ASSET_NAME = "ゴールド・金 (GC=F)"
TARGET_MARGIN = 2.0  # 2ドル利確ターゲット

st.title(f"🚀 UTAYA リアルタイム予測エンジン: {ASSET_NAME}")
st.write(
    "🛡️ 市場データをスキャン中...（自動更新またはブラウザのリロード機能で最新化されます）"
)

# 画面表示用のプレースホルダー
status_placeholder = st.empty()
metric_placeholder = st.empty()

# メインの処理ループ（※Streamlit上で回す場合は st.empty() や rerun を使うのが一般的です）
try:
  # 最新の市場データを取得
  df = yf.download(TICKER, period="5d", interval="15m", progress=False)
  if not df.empty and len(df) >= 30:
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    closes = df["Close"]


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

    # 画面に結果を表示
    metric_placeholder.metric(label="現在のRSI(5)", value=f"{rsi_val:.1f}")

    col1, col2 = st.columns(2)
    col1.metric(label="上昇確率", value=f"{prob_up:.1f}%")
    col2.metric(label="下落確率", value=f"{prob_down:.1f}%")

    if prob_down >= 70.0:
      st.error(
          "🔥🔥🔥 【UTAYAベスト発動：LOW売りエントリー】 下落確率: "
          f"{prob_down:.1f}% 🔥🔥🔥"
      )
    elif prob_up >= 70.0:
      st.success(
          "✨✨✨ 【UTAYAベスト発動：HIGH買いエントリー】 上昇確率: "
          f"{prob_up:.1f}% ✨✨✨"
      )
    else:
      st.info("📊 シグナル待ち中（様子見）")

except Exception as e:
  st.error(f"⚠️ 処理エラー: {e}")
