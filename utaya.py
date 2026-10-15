import base64
import datetime
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ページの設定（スマホ最適化・コンパクトレイアウト）
st.set_page_config(
    page_title="UTAYA 司令室 [QUANTUM-3X タイムマシーン]", layout="centered"
)


# 画像をBase64にエンコードしてHTMLで確実に表示する関数
def get_image_base64(image_path):
  try:
    with open(image_path, "rb") as img_file:
      return base64.b64encode(img_file.read()).decode("utf-8")
  except Exception:
    return ""


# スタイリング
st.markdown(
    """
    <style>
    .top-asset-card {
        background: linear-gradient(135deg, #020617, #0f172a);
        border: 2px solid #38bdf8;
        padding: 16px;
        border-radius: 14px;
        margin-bottom: 14px;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.35);
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
    .badge-buy { background-color: #0284c7; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 1.05em; }
    .badge-sell { background-color: #e11d48; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 1.05em; }
    .badge-stable-buy { background-color: #0369a1; color: #bae6fd; border: 1px solid #38bdf8; padding: 3px 8px; border-radius: 6px; font-weight: bold; }
    .badge-stable-sell { background-color: #be123c; color: #fecdd3; border: 1px solid #fb7185; padding: 3px 8px; border-radius: 6px; font-weight: bold; }
    .anchor-notice { font-size: 0.82em; color: #38bdf8; background: #0c4a6e; padding: 6px 10px; border-radius: 6px; margin-bottom: 10px; display: inline-block; font-weight: bold; }

    .vtuber-container {
        display: flex;
        align-items: center;
        background: linear-gradient(135deg, #18181b, #4c0519);
        border: 3px solid #fb7185;
        padding: 18px;
        border-radius: 20px;
        margin-bottom: 18px;
        box-shadow: 0 0 40px rgba(251, 113, 133, 0.4);
    }
    .vtuber-avatar-box {
        text-align: center;
        min-width: 140px;
        margin-right: 18px;
    }
    .vtuber-img {
        width: 130px;
        height: 130px;
        object-fit: cover;
        border-radius: 14px;
        border: 3px solid #fff1f2;
        box-shadow: 0 0 25px rgba(244, 63, 94, 0.7);
        animation: vtuber-pulse 2s infinite ease-in-out;
    }
    @keyframes vtuber-pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.03); }
    }
    .vtuber-name {
        font-size: 0.82em;
        color: #fbcfe8;
        font-weight: bold;
        margin-top: 6px;
        letter-spacing: 1px;
    }
    .comic-bubble {
        position: relative;
        background: #ffffff;
        color: #0f172a;
        padding: 14px 18px;
        border-radius: 14px;
        font-weight: bold;
        font-size: 1.02em;
        width: 100%;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        border: 4px solid #fb7185;
        line-height: 1.5;
    }
    .comic-bubble::after {
        content: '';
        position: absolute;
        left: -14px;
        top: 24px;
        border-width: 10px 14px 10px 0;
        border-style: solid;
        border-color: transparent #ffffff transparent;
        display: block;
    }
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
if "pinned_top_asset" not in st.session_state:
  st.session_state.pinned_top_asset = None
if "pin_timestamp" not in st.session_state:
  st.session_state.pin_timestamp = 0.0
if "learning_reflection" not in st.session_state:
  st.session_state.learning_reflection = (
      "未来予測と現実の乖離（ノイズ・スリッページ）を24時間連続解析中..."
  )

st.sidebar.title("🧠 司令室 [QUANTUM-3X]")
refresh_interval = st.sidebar.slider(
    "⏱️ クイック更新間隔(秒)", min_value=5, max_value=60, value=10, step=5
)
if st.sidebar.button("⚡ 即時量子スキャン実行"):
  st.session_state.scan_count += 1
  st.rerun()

st.markdown(
    f'<meta http-equiv="refresh" content="{refresh_interval}">',
    unsafe_allow_html=True,
)

st.markdown("### 🧠 UTAYA [QUANTUM-3X タイムマシーン]")

jst_now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
st.markdown(
    f"<p style='font-size:0.85em; color:#38bdf8; margin-bottom:6px;'>⚡"
    f" 量子スキャン時刻: {jst_now.strftime('%H:%M:%S.%f')[:-4]} JST</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='font-size:0.82em; color:#cbd5e1; background:#020617;"
    f" border-left: 4px solid #fb7185; padding: 6px 10px; border-radius: 6px;"
    f" margin-bottom: 10px;'>💡 <b>24h自己反省AIログ:</b>"
    f" {st.session_state.learning_reflection}</div>",
    unsafe_allow_html=True,
)

st.session_state.scan_count += 1
analyzed_results = []

for name, symbol, fallback_price in ASSETS_LIST:
  try:
    df = yf.download(symbol, period="5d", interval="15m", progress=False)
    if df.empty or len(df) < 20:
      raise ValueError("Empty data")
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    closes = df["Close"]
    highs = df["High"] if "High" in df.columns else closes
    lows = df["Low"] if "Low" in df.columns else closes

    vel_1 = closes.pct_change(1)
    vel_3 = closes.pct_change(3)
    accel = vel_1.diff()

    rolling_mean = closes.rolling(window=10).mean()
    rolling_std = closes.rolling(window=10).std()
    z_score = (closes - rolling_mean) / (rolling_std + 1e-8)

    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - closes.shift(1)), np.abs(lows - closes.shift(1))
        ),
    )
    atr = pd.Series(tr).rolling(window=5).mean()

    current_price = float(closes.iloc[-1])
    cur_z = float(z_score.iloc[-1])
    cur_vel3 = float(vel_3.iloc[-1]) * 100
    cur_accel = float(accel.iloc[-1]) * 100
    cur_atr = (
        float(atr.iloc[-1])
        if not pd.isna(atr.iloc[-1]) and atr.iloc[-1] > 0
        else current_price * 0.002
    )

    quantum_power = cur_vel3 * 2.0 + cur_accel * 3.0 - (cur_z * 5.0)
    prob_buy = float(np.clip(50.0 + quantum_power * 3.5, 10.0, 92.0))
    prob_sell = float(100.0 - prob_buy)

    endure_buy = current_price - (cur_atr * 0.6)
    take_buy = current_price + (cur_atr * 1.5)
    endure_sell = current_price + (cur_atr * 0.6)
    take_sell = current_price - (cur_atr * 1.5)

    if prob_buy >= 65.0 or (cur_z < -1.2 and cur_vel3 > 0):
      signal_html = (
          f'<span class="badge-buy">🚀 空間歪曲 BUY ({prob_buy:.0f}%)</span>'
      )
      analysis = (
          f"🚨【今だ！買いエントリー】 損切(耐え): {endure_buy:,.1f} ｜"
          f" 狙撃利確: {take_buy:,.1f}"
      )
      score = prob_buy
    elif prob_sell >= 65.0 or (cur_z > 1.2 and cur_vel3 < 0):
      signal_html = (
          f'<span class="badge-sell">⚡ 空間歪曲 SELL ({prob_sell:.0f}%)</span>'
      )
      analysis = (
          f"🚨【今だ！売りエントリー】 損切(耐え): {endure_sell:,.1f} ｜"
          f" 狙撃利確: {take_sell:,.1f}"
      )
      score = prob_sell
    else:
      if prob_buy >= prob_sell:
        signal_html = (
            f'<span class="badge-stable-buy">⏳ 予兆蓄積 (買優勢'
            f" {prob_buy:.0f}%)</span>"
        )
        analysis = (
            "⚠️【待機推奨】エネルギー蓄積中（合図が出るまで指値で構えろ）"
        )
      else:
        signal_html = (
            f'<span class="badge-stable-sell">⏳ 予兆蓄積 (売優勢'
            f" {prob_sell:.0f}%)</span>"
        )
        analysis = (
            "⚠️【待機推奨】エネルギー蓄積中（合図が出るまで指値で構えろ）"
        )
      score = max(prob_buy, prob_sell)

    analyzed_results.append({
        "name": name,
        "current_price": current_price,
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
        "prob_buy": 50.0,
        "prob_sell": 50.0,
        "signal_html": '<span class="badge-stable-sell">⏳ 同期中</span>',
        "analysis": "量子ストリーム再接続中...",
        "score": 50.0,
    })

analyzed_results.sort(key=lambda x: -x["score"])

if analyzed_results:
  current_time = time.time()
  best_item = analyzed_results[0]

  if (
      st.session_state.pinned_top_asset is None
      or (current_time - st.session_state.pin_timestamp) > 60.0
      or best_item["score"] >= st.session_state.pinned_top_asset["score"] * 0.98
  ):
    if (
        st.session_state.pinned_top_asset is None
        or best_item["name"] != st.session_state.pinned_top_asset["name"]
    ):
      st.session_state.pinned_top_asset = best_item
      st.session_state.pin_timestamp = current_time
      st.session_state.learning_reflection = (
          f"直近の変動を解析: {best_item['name']} の特異点パターンを適応中。"
          "現実の反発点と未来予測の誤差を収斂。"
      )

  display_top = st.session_state.pinned_top_asset or best_item

  top_name = display_top["name"]
  top_score = display_top["score"]
  top_price = display_top["current_price"]

  ikuzo_img_path = "ikuzo.png"
  mate_img_path = "mate.png"

  ikuzo_b64 = get_image_base64(ikuzo_img_path)
  mate_b64 = get_image_base64(mate_img_path)

  if top_score >= 65.0:
    img_src = (
        f"data:image/png;base64,{ikuzo_b64}"
        if ikuzo_b64
        else "https://via.placeholder.com/130?text=IKUZO"
    )
    vtuber_speech = (
        f"「全集中、未来の扉を開くっ！司令官、『 <b>{top_name}</b>' の特異点を完璧にキャッチしたわ！今すぐこのポイントで一気呵成にエントリーしなさいっ！」"
    )
  else:
    img_src = (
        f"data:image/png;base64,{mate_b64}"
        if mate_b64
        else "https://via.placeholder.com/130?text=MATE"
    )
    vtuber_speech = (
        f"「待て！まだ波形が乱れているわ。『 <b>{top_name}</b>' のエネルギーが完全に収束するまで、無闇に動いてはダメ……！私の『いくぞ！』の合図が出るまでここで構えて待機よっ！」"
    )

  st.markdown(
      f"""
    <div class="vtuber-container">
        <div class="vtuber-avatar-box">
            <img src="{img_src}" class="vtuber-img">
            <div class="vtuber-name">和装VTuber UTAYA</div>
        </div>
        <div class="comic-bubble">
            {vtuber_speech}
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  st.markdown(
      "<p style='font-weight:bold; color:#38bdf8; margin-bottom:4px;'>🔥"
      " 最重要ターゲット（1分間アンカー固定中・エントリー設定用）</p>",
      unsafe_allow_html=True,
  )
  st.markdown(
      '<span class="anchor-notice">📌 値動きが激しくてもこの銘柄と価格ポイントを1分間キープします。落ち着いて指値・損切をセットしてください。</span>',
      unsafe_allow_html=True,
  )

  st.markdown(
      f"""
    <div class="top-asset-card">
        <div style="font-size: 1.15em; font-weight: bold; margin-bottom: 6px; color: #38bdf8;">★ 司令塔指定銘柄: {display_top['name']}</div>
        <div style="display: flex; justify-content: space-between; font-size: 1.0em; margin-bottom: 6px;">
            <span>現在値: <b>{display_top['current_price']:,.2f}</b></span>
            <span>司令判定: {display_top['signal_html']}</span>
        </div>
        <div style="font-size: 0.9em; color: #f8fafc; background: #020617; padding: 10px; border-radius: 8px; border: 1px solid #38bdf8;">
            {display_top['analysis']}
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  st.markdown(
      "<p style='font-weight:bold; color:#94a3b8; margin:10px 0 4px 0;'>📋"
      " その他のアセット動向</p>",
      unsafe_allow_html=True,
  )
  for item in analyzed_results[1:]:
    st.markdown(
        f"""
        <div class="asset-card">
            <div style="display: flex; justify-content: space-between; font-size: 0.92em;">
                <span><b>{item['name']}</b> ({item['current_price']:,.2f})</span>
                <span>{item['signal_html']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
