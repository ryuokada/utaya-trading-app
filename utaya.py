import base64
import datetime
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ページの設定
st.set_page_config(
    page_title="UTAYA 司令室 [QUANTUM-3X タイムマシーン]", layout="centered"
)


def get_image_base64(image_path):
  try:
    with open(image_path, "rb") as img_file:
      return base64.b64encode(img_file.read()).decode("utf-8")
  except Exception:
    return ""


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

# 厳選した3銘柄に絞り込み
ASSETS_LIST = [
    ("ドル円 (USD/JPY)", "USDJPY=X", 155.0),
    ("ゴールド・金", "GC=F", 2350.0),
    ("日経平均株価", "^N225", 38500.0),
]

if "scan_count" not in st.session_state:
  st.session_state.scan_count = 520
if "pinned_top_asset" not in st.session_state:
  st.session_state.pinned_top_asset = None
if "pin_timestamp" not in st.session_state:
  st.session_state.pin_timestamp = 0.0
if "learning_reflection" not in st.session_state:
  st.session_state.learning_reflection = (
      "3大基幹アセット（ドル・金・日経平均）の特異点波形を高精度解析中..."
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

jst_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
    hours=9
)
st.markdown(
    f"<p style='font-size:0.85em; color:#38bdf8; margin-bottom:6px;'>⚡"
    f" 量子スキャン時刻: {jst_now.strftime('%H:%M:%S')} JST</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='font-size:0.82em; color:#cbd5e1; background:#020617;"
    f" border-left: 4px solid #fb7185; padding: 6px 10px; border-radius: 6px;"
    f" margin-bottom: 10px;'>💡 <b>24h自己反省AIログ:</b>"
    f" {st.session_state.learning_reflection}</div>",
    unsafe_allow_html=True,
)

analyzed_results = []

for name, symbol, fallback_price in ASSETS_LIST:
  try:
    # メモリ消費を徹底的に抑えるため最小限の期間だけ取得
    df = yf.download(symbol, period="2d", interval="1h", progress=False)
    if df.empty or len(df) < 3:
      df = yf.download(symbol, period="5d", interval="1d", progress=False)

    if df.empty:
      raise ValueError("Empty data")

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    closes = df["Close"].dropna()
    current_price = float(closes.iloc[-1])

    mom = (
        float((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100)
        if len(closes) > 1
        else 0.0
    )

    prob_buy = float(np.clip(50.0 + mom * 12.0, 15.0, 90.0))
    prob_sell = float(100.0 - prob_buy)

    if prob_buy >= 65.0:
      signal_html = (
          f'<span class="badge-buy">🚀 空間歪曲 BUY ({prob_buy:.0f}%)</span>'
      )
      analysis = f"🚨【今だ！買い】 現在値: {current_price:,.2f}"
      score = prob_buy
    elif prob_sell >= 65.0:
      signal_html = (
          f'<span class="badge-sell">⚡ 空間歪曲 SELL ({prob_sell:.0f}%)</span>'
      )
      analysis = f"🚨【今だ！売り】 現在値: {current_price:,.2f}"
      score = prob_sell
    else:
      signal_html = (
          f'<span class="badge-stable-buy">⏳ 予兆蓄積 (優勢'
          f" {max(prob_buy, prob_sell):.0f}%)</span>"
      )
      analysis = f"⚠️【待機】エネルギー収束中（現在値: {current_price:,.2f}）"
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
    st.session_state.pinned_top_asset = best_item
    st.session_state.pin_timestamp = current_time

  display_top = st.session_state.pinned_top_asset or best_item
  top_name = display_top["name"]
  top_score = display_top["score"]
  top_price = display_top["current_price"]

  ikuzo_b64 = get_image_base64("ikuzo.png")
  mate_b64 = get_image_base64("mate.png")

  if top_score >= 65.0:
    img_src = (
        f"data:image/png;base64,{ikuzo_b64}"
        if ikuzo_b64
        else "https://via.placeholder.com/130?text=IKUZO"
    )
    if display_top["prob_buy"] >= 65.0:
      vtuber_speech = f"「いくぞ！ <b>{top_name}</b> を<b>【買い】</b>だ！ 今すぐ <b>{top_price:,.2f}</b> でエントリーせよっ！」"
    else:
      vtuber_speech = f"「いくぞ！ <b>{top_name}</b> を<b>【売り】</b>だ！ 今すぐ <b>{top_price:,.2f}</b> でショートを叩けっ！」"
  else:
    img_src = (
        f"data:image/png;base64,{mate_b64}"
        if mate_b64
        else "https://via.placeholder.com/130?text=MATE"
    )
    vtuber_speech = f"「まて！ <b>{top_name}</b> は現在 <b>{top_price:,.2f}</b> ｜ まだ波形が乱れている。合図が出るまで待機だ！」"

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
      " 最重要ターゲット（1分間アンカー固定中）</p>",
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
      " その他の基幹アセット動向</p>",
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
