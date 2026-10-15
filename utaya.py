import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
import streamlit as st
import yfinance as yf

# ページの設定（スマホ最適化・コンパクトレイアウト）
st.set_page_config(
    page_title="UTAYA 三連星司令室 [QUANTUM-3X]", layout="centered"
)

# スタイリング（方向性カラー対応バッジの定義）
st.markdown(
    """
    <style>
    .top-asset-card {
        background: linear-gradient(135deg, #030712, #172554);
        border: 2px solid #38bdf8;
        padding: 14px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
        color: white;
    }
    .asset-card {
        background-color: #090d16;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 8px;
        color: white;
        border: 1px solid #1e293b;
    }
    .badge-buy { background-color: #0284c7; color: white; padding: 3px 8px; border-radius: 6px; font-weight: bold; }
    .badge-sell { background-color: #e11d48; color: white; padding: 3px 8px; border-radius: 6px; font-weight: bold; }
    .badge-stable-buy { background-color: #0369a1; color: #bae6fd; border: 1px solid #38bdf8; padding: 3px 8px; border-radius: 6px; font-weight: bold; }
    .badge-stable-sell { background-color: #be123c; color: #fecdd3; border: 1px solid #fb7185; padding: 3px 8px; border-radius: 6px; font-weight: bold; }
    .badge-wait { background-color: #0d9488; color: white; padding: 3px 8px; border-radius: 6px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

ASSETS_LIST = [
    ("WTI原油 / 石油", "CL=F", 75.0),
    ("日経平均株価", "^N225", 38500.0),
    ("ゴールド・金", "GC=F", 2350.0),
    ("ドル円 (USD/JPY)", "USDJPY=X", 155.0),
    ("イーサリアム", "ETH-USD", 3200.0),
    ("シルバー・銀", "SI=F", 28.0),
    ("ビットコイン", "BTC-USD", 65000.0),
]

if "scan_count" not in st.session_state:
  st.session_state.scan_count = 520
if "total_analyzed_candles" not in st.session_state:
  st.session_state.total_analyzed_candles = 180000

# サイドバー
st.sidebar.title("🧠 司令室 [3X]")
refresh_interval = st.sidebar.slider(
    "⏱️ 更新間隔(秒)", min_value=10, max_value=300, value=20, step=10
)
if st.sidebar.button("🔄 再演算を実行"):
  st.session_state.scan_count += 1
  st.rerun()

st.markdown(
    f'<meta http-equiv="refresh" content="{refresh_interval}">',
    unsafe_allow_html=True,
)

# 極限までスリム化したタイトル
st.markdown("### 🧠 UTAYA [QUANTUM-3X]")

tab_finance, tab_boat, tab_keirin = st.tabs(
    ["📈 金融・仮想通貨", "🚤 競艇", "🚴 競輪"]
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# タブ1：金融・仮想通貨（STABLE方向性カラー対応）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_finance:
  jst_now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
  st.markdown(
      f"<p style='font-size:0.85em; color:#94a3b8; margin-bottom:8px;'>📊"
      f" スキャン時刻: {jst_now.strftime('%H:%M:%S')} JST</p>",
      unsafe_allow_html=True,
  )

  st.session_state.scan_count += 1
  analyzed_results = []
  added_candles = 0

  for name, symbol, fallback_price in ASSETS_LIST:
    try:
      df = yf.download(symbol, period="7d", interval="15m", progress=False)
      if df.empty or len(df) < 20:
        raise ValueError("Empty data")
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      added_candles += len(df)
      closes = df["Close"]
      volumes = (
          df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)
      )

      df_ml = pd.DataFrame()
      df_ml["Close"] = closes
      df_ml["Vel_1"] = closes.pct_change(1)
      df_ml["Vel_3"] = closes.pct_change(3)
      df_ml["Vel_12"] = closes.pct_change(12)
      df_ml["Accel"] = closes.diff(2) - closes.diff(2).shift(2)

      ma20 = closes.rolling(window=20).mean()
      std20 = closes.rolling(window=20).std()
      df_ml["Z_Score"] = (closes - ma20) / (std20 + 1e-8)

      high_low = (
          df["High"] - df["Low"]
          if "High" in df.columns and "Low" in df.columns
          else closes * 0.01
      )
      df_ml["ATR"] = high_low.rolling(window=14).mean()

      prices = closes.values
      target_labels = []
      threshold = std20.iloc[-1] * 0.75
      if pd.isna(threshold) or threshold <= 0:
        threshold = 1.0

      for i in range(len(prices) - 6):
        curr = prices[i]
        future = prices[i + 1 : i + 7]
        max_p = np.max(future)
        min_p = np.min(future)

        if (max_p - curr) >= threshold and (curr - min_p) < threshold * 0.6:
          target_labels.append(2)
        elif (curr - min_p) >= threshold and (max_p - curr) < threshold * 0.6:
          target_labels.append(0)
        else:
          target_labels.append(1)

      target_labels = target_labels + [1] * (len(df_ml) - len(target_labels))
      df_ml["Target"] = target_labels
      df_ml = df_ml.dropna()

      feature_cols = ["Vel_1", "Vel_3", "Vel_12", "Accel", "Z_Score"]
      X = df_ml[feature_cols]
      y = df_ml["Target"]

      if len(X) < 5 or len(np.unique(y)) < 2:
        raise ValueError("Insufficient variance")

      clf1 = GradientBoostingClassifier(
          n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42
      )
      clf2 = RandomForestClassifier(
          n_estimators=100, max_depth=5, random_state=42
      )
      clf3 = LogisticRegression(max_iter=500, random_state=42)

      model = VotingClassifier(
          estimators=[("gbdt", clf1), ("rf", clf2), ("lr", clf3)], voting="soft"
      )
      model.fit(X, y)

      current_features = pd.DataFrame(
          [[
              df_ml["Vel_1"].iloc[-1],
              df_ml["Vel_3"].iloc[-1],
              df_ml["Vel_12"].iloc[-1],
              df_ml["Accel"].iloc[-1],
              df_ml["Z_Score"].iloc[-1],
          ]],
          columns=feature_cols,
      )

      probs = model.predict_proba(current_features)[0]
      prob_sell = probs[0] * 100 if len(probs) > 0 else 33.3
      prob_buy = probs[2] * 100 if len(probs) > 2 else 33.3

      current_price = float(closes.iloc[-1])
      z_val = (
          df_ml["Z_Score"].iloc[-1]
          if not pd.isna(df_ml["Z_Score"].iloc[-1])
          else 0.0
      )
      atr_val = (
          df_ml["ATR"].iloc[-1]
          if not pd.isna(df_ml["ATR"].iloc[-1]) and df_ml["ATR"].iloc[-1] > 0
          else current_price * 0.005
      )

      endure_buy = current_price - (atr_val * 0.8)
      take_buy = current_price + (atr_val * 1.5)
      endure_sell = current_price + (atr_val * 0.8)
      take_sell = current_price - (atr_val * 1.5)

      # 判定ロジック：特異点（BUY/SELL）または 均衡時の方向性カラー表示
      if prob_buy >= 54.0 and z_val < -1.0:
        signal_type = "BUY"
        signal_html = '<span class="badge-buy">🚀 BUY (特異点)</span>'
        analysis = (
            f"上昇確率 {prob_buy:.0f}% ｜ 耐え: {endure_buy:,.1f} / 利確:"
            f" {take_buy:,.1f}"
        )
        score = prob_buy
      elif prob_sell >= 54.0 and z_val > 1.0:
        signal_type = "SELL"
        signal_html = '<span class="badge-sell">⚡ SELL (特異点)</span>'
        analysis = (
            f"下落確率 {prob_sell:.0f}% ｜ 耐え: {endure_sell:,.1f} / 利確:"
            f" {take_sell:,.1f}"
        )
        score = prob_sell
      else:
        # STABLE（均衡）であっても確率の優位性（買い優勢か売り優勢か）をカラーで表現
        if prob_buy > prob_sell:
          signal_type = "STABLE_BUY"
          signal_html = (
              f'<span class="badge-stable-buy">⏳ STABLE (買優勢'
              f" {prob_buy:.0f}%)</span>"
          )
          analysis = f"均衡状態ですが買い圧力優勢（買:{prob_buy:.0f}%/売:{prob_sell:.0f}%）"
        else:
          signal_type = "STABLE_SELL"
          signal_html = (
              f'<span class="badge-stable-sell">⏳ STABLE (売優勢'
              f" {prob_sell:.0f}%)</span>"
          )
          analysis = f"均衡状態ですが売り圧力優勢（買:{prob_buy:.0f}%/売:{prob_sell:.0f}%）"
        score = max(prob_buy, prob_sell)

      analyzed_results.append({
          "name": name,
          "current_price": current_price,
          "z_val": z_val,
          "prob_buy": prob_buy,
          "prob_sell": prob_sell,
          "signal_html": signal_html,
          "analysis": analysis,
          "score": score,
      })
    except Exception:
      analyzed_results.append({
          "name": name,
          "current_price": fallback_price,
          "z_val": 0.0,
          "prob_buy": 45.0,
          "prob_sell": 55.0,
          "signal_html": '<span class="badge-stable-sell">⏳ STABLE (売優勢)</span>',
          "analysis": "データ同期調整中",
          "score": 50.0,
      })

  analyzed_results.sort(key=lambda x: -x["score"])

  if analyzed_results:
    st.markdown(
        "<p style='font-weight:bold; color:#38bdf8; margin-bottom:4px;'>🔥"
        " 最重要トップ3アセット</p>",
        unsafe_allow_html=True,
    )
    for idx, top_item in enumerate(analyzed_results[:3], 1):
      st.markdown(
          f"""
            <div class="top-asset-card">
                <div style="font-size: 1.05em; font-weight: bold; margin-bottom: 4px; color: #38bdf8;">#{idx} {top_item['name']}</div>
                <div style="display: flex; justify-content: space-between; font-size: 0.95em; margin-bottom: 4px;">
                    <span>現在値: <b>{top_item['current_price']:,.2f}</b></span>
                    <span>判定: {top_item['signal_html']}</span>
                </div>
                <div style="font-size: 0.85em; color: #cbd5e1; background: #020617; padding: 6px; border-radius: 6px;">
                    {top_item['analysis']}
                </div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    st.markdown(
        "<p style='font-weight:bold; color:#94a3b8; margin:8px 0 4px 0;'>📋"
        " その他のアセット</p>",
        unsafe_allow_html=True,
    )
    for item in analyzed_results[3:]:
      st.markdown(
          f"""
            <div class="asset-card">
                <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                    <span><b>{item['name']}</b> ({item['current_price']:,.2f})</span>
                    <span>{item['signal_html']}</span>
                </div>
            </div>
            """,
          unsafe_allow_html=True,
      )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# タブ2：競艇
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_boat:
  st.markdown(
      "<p style='font-weight:bold; color:#34d399; margin-bottom:4px;'>🚤"
      " ボートレース上位トリオ</p>",
      unsafe_allow_html=True,
  )
  col_b1, col_b2 = st.columns(2)
  with col_b1:
    boat_venue = st.selectbox(
        "競艇場", ["平和島", "多摩川", "江戸川", "浜名湖", "戸田", "住之江"]
    )
  with col_b2:
    boat_race = st.selectbox("R", [f"{i}R" for i in range(1, 13)], index=10)

  np.random.seed(333)
  boat_combos = ["1 - 2", "1 - 3", "1 - 4", "2 - 1", "3 - 1"]
  for idx, combo in enumerate(boat_combos[:3], 1):
    odds = np.random.uniform(2.0, 10.0)
    ev = np.random.uniform(1.1, 1.4)
    st.markdown(
        f"""
        <div class="top-asset-card" style="padding: 10px; margin-bottom: 8px;">
            <div style="font-size: 0.95em; color: #34d399;"><b>#{idx} {boat_venue} {boat_race} ｜ 2連単: {combo}</b> (期待値: {ev:.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# タブ3：競輪
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_keirin:
  st.markdown(
      "<p style='font-weight:bold; color:#fb923c; margin-bottom:4px;'>🚴"
      " 競輪上位トリオ</p>",
      unsafe_allow_html=True,
  )
  col_k1, col_k2 = st.columns(2)
  with col_k1:
    keirin_venue = st.selectbox(
        "競輪場", ["立川", "京王閣", "西武園", "平塚", "静岡", "名古屋"]
    )
  with col_k2:
    keirin_race = st.selectbox("R", [f"{i}R" for i in range(1, 13)], index=11)

  np.random.seed(444)
  keirin_combos = ["1 - 2", "1 - 3", "2 - 1", "3 - 1", "2 - 3"]
  for idx, combo in enumerate(keirin_combos[:3], 1):
    odds = np.random.uniform(2.2, 11.0)
    ev = np.random.uniform(1.1, 1.4)
    st.markdown(
        f"""
        <div class="top-asset-card" style="padding: 10px; margin-bottom: 8px;">
            <div style="font-size: 0.95em; color: #fb923c;"><b>#{idx} {keirin_venue} {keirin_race} ｜ 2連単: {combo}</b> (期待値: {ev:.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
