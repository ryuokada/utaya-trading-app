import base64
import datetime
import time
import streamlit as st

# ページの設定
st.set_page_config(page_title="UTAYA 司令室 [QUANTUM-3X]", layout="centered")


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
    </style>
    """,
    unsafe_allow_html=True,
)

# 厳選した3大アセット（メモリを一切使わない安定ロジック用）
ASSETS_LIST = [
    ("ドル円 (USD/JPY)", 154.80, 0.45),
    ("ゴールド・金", 2365.50, 0.82),
    ("日経平均株価", 38750.00, -0.30),
]

if "scan_count" not in st.session_state:
  st.session_state.scan_count = 520

st.sidebar.title("🧠 司令室 [QUANTUM-3X]")
refresh_interval = st.sidebar.slider(
    "⏱️ クイック更新間隔(秒)", min_value=5, max_value=60, value=10, step=5
)
if st.sidebar.button("⚡ 即時量子スキャン実行"):
  st.session_state.scan_count += 1
  st.rerun()

st.markdown("### 🧠 UTAYA [QUANTUM-3X タイムマシーン]")

jst_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
    hours=9
)
st.markdown(
    f"<p style='font-size:0.85em; color:#38bdf8; margin-bottom:6px;'>⚡"
    f" 量子スキャン時刻: {jst_now.strftime('%H:%M:%S')} JST</p>",
    unsafe_allow_html=True,
)

analyzed_results = []
for name, base_price, change in ASSETS_LIST:
  # 変動を微小にシミュレートしつつメモリ消費をゼロにする高速演算
  np_pseudo_val = change + ((st.session_state.scan_count % 5) * 0.05)
  current_price = base_price * (1 + np_pseudo_val / 100)

  if np_pseudo_val > 0.5:
    signal_html = '<span class="badge-buy">🚀 空間歪曲 BUY (78%)</span>'
    analysis = f"🚨【今だ！買い】 勢い加速中（現在値: {current_price:,.2f}）"
    score = 78.0
  elif np_pseudo_val < -0.2:
    signal_html = '<span class="badge-sell">⚡ 空間歪曲 SELL (72%)</span>'
    analysis = f"🚨【今だ！売り】 下降トレンド発生（現在値: {current_price:,.2f}）"
    score = 72.0
  else:
    signal_html = '<span class="badge-stable-buy">⏳ 予兆蓄積 (優勢 60%)</span>'
    analysis = f"⚠️【待機】エネルギー収束中（現在値: {current_price:,.2f}）"
    score = 60.0

  analyzed_results.append({
      "name": name,
      "current_price": current_price,
      "signal_html": signal_html,
      "analysis": analysis,
      "score": score,
  })

analyzed_results.sort(key=lambda x: -x["score"])
best_item = analyzed_results[0]

ikuzo_b64 = get_image_base64("ikuzo.png")
img_src = (
    f"data:image/png;base64,{ikuzo_b64}"
    if ikuzo_b64
    else "https://via.placeholder.com/130?text=IKUZO"
)

vtuber_speech = f"「いくぞ！ <b>{best_item['name']}</b> を狙え！ 今すぐ <b>{best_item['current_price']:,.2f}</b> でエントリーの準備だっ！」"

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
    " 最重要ターゲット</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="top-asset-card">
        <div style="font-size: 1.15em; font-weight: bold; margin-bottom: 6px; color: #38bdf8;">★ 司令塔指定銘柄: {best_item['name']}</div>
        <div style="display: flex; justify-content: space-between; font-size: 1.0em; margin-bottom: 6px;">
            <span>現在値: <b>{best_item['current_price']:,.2f}</b></span>
            <span>司令判定: {best_item['signal_html']}</span>
        </div>
        <div style="font-size: 0.9em; color: #f8fafc; background: #020617; padding: 10px; border-radius: 8px; border: 1px solid #38bdf8;">
            {best_item['analysis']}
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
