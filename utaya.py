import base64
import datetime
import streamlit as st

# ページの設定
st.set_page_config(page_title="UTAYA 司令室 [QUANTUM-3X]", layout="centered")


def get_image_base64(image_path):
  try:
    with open(image_path, "rb") as img_file:
      return base64.b64encode(img_file.read()).decode("utf-8")
  except Exception:
    return ""


# スタイリングの定義
st.markdown(
    """
    <style>
    .top-asset-card {
        background: linear-gradient(135deg, #020617, #0f172a);
        border: 2px solid #38bdf8;
        padding: 20px;
        border-radius: 14px;
        margin-bottom: 16px;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.35);
        color: white;
    }
    .asset-card {
        background-color: #090d16;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
        border: 1px solid #1e293b;
    }
    .badge-buy { background-color: #0284c7; color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 1.1em; }
    .badge-sell { background-color: #e11d48; color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 1.1em; }
    .badge-stable { background-color: #0369a1; color: #bae6fd; border: 1px solid #38bdf8; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    
    .vtuber-container {
        display: flex;
        align-items: center;
        background: linear-gradient(135deg, #18181b, #4c0519);
        border: 3px solid #fb7185;
        padding: 20px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 40px rgba(251, 113, 133, 0.4);
    }
    .vtuber-avatar-box {
        text-align: center;
        min-width: 140px;
        margin-right: 20px;
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
        font-size: 0.85em;
        color: #fbcfe8;
        font-weight: bold;
        margin-top: 8px;
        letter-spacing: 1px;
    }
    .comic-bubble {
        position: relative;
        background: #ffffff;
        color: #0f172a;
        padding: 16px 20px;
        border-radius: 14px;
        font-weight: bold;
        font-size: 1.1em;
        width: 100%;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        border: 4px solid #fb7185;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# アセットリスト
ASSETS_LIST = [
    ("ドル円 (USD/JPY)", 154.85, 0.45),
    ("ゴールド・金", 2368.20, 0.92),
    ("日経平均株価", 38820.00, -0.25),
]

if "scan_count" not in st.session_state:
  st.session_state.scan_count = 520

# サイドバー
st.sidebar.title("🧠 司令室 [QUANTUM-3X]")
refresh_interval = st.sidebar.slider(
    "⏱️ クイック更新間隔(秒)", min_value=5, max_value=60, value=10, step=5
)
if st.sidebar.button("⚡ 即時量子スキャン実行"):
  st.session_state.scan_count += 1
  st.rerun()

# タイトルと時刻
st.markdown("### 🧠 UTAYA 司令室 [QUANTUM-3X タイムマシーン]")
jst_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
    hours=9
)
st.markdown(
    f"<p style='font-size:0.9em; color:#38bdf8; margin-bottom:8px;'>⚡"
    f" 量子スキャン時刻: {jst_now.strftime('%H:%M:%S')} JST</p>",
    unsafe_allow_html=True,
)

analyzed_results = []
for name, base_price, change in ASSETS_LIST:
  pseudo_change = change + ((st.session_state.scan_count % 3) * 0.02)
  current_price = base_price * (1 + pseudo_change / 100)

  if pseudo_change > 0.5:
    signal_html = '<span class="badge-buy">🚀 空間歪曲 BUY (79%)</span>'
    analysis = f"🚨【今だ！買い】 上昇トレンド加速中（現在値: {current_price:,.2f}）"
    score = 79.0
  elif pseudo_change < -0.1:
    signal_html = '<span class="badge-sell">⚡ 空間歪曲 SELL (74%)</span>'
    analysis = f"🚨【今だ！売り】 下降トレンド発生（現在値: {current_price:,.2f}）"
    score = 74.0
  else:
    signal_html = '<span class="badge-stable">⏳ 予兆蓄積 (優勢 62%)</span>'
    analysis = f"⚠️【待機】エネルギー収束中（現在値: {current_price:,.2f}）"
    score = 62.0

  analyzed_results.append({
      "name": name,
      "current_price": current_price,
      "signal_html": signal_html,
      "analysis": analysis,
      "score": score,
  })

analyzed_results.sort(key=lambda x: -x["score"])
best_item = analyzed_results[0]

# 画像読み込み（フォールバック対応）
ikuzo_b64 = get_image_base64("ikuzo.png")
img_src = (
    f"data:image/png;base64,{ikuzo_b64}"
    if ikuzo_b64
    else "https://via.placeholder.com/130/ff4b4b/ffffff?text=UTAYA"
)

vtuber_speech = f"「司令官！ <b>{best_item['name']}</b> の特異点をキャッチしたわ！ 今すぐ <b>{best_item['current_price']:,.2f}</b> でエントリーの準備をしなさいっ！」"

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
    "<p style='font-weight:bold; color:#38bdf8; margin-bottom:6px;'>🔥"
    " 最重要ターゲット</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="top-asset-card">
        <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 8px; color: #38bdf8;">★ 司令塔指定銘柄: {best_item['name']}</div>
        <div style="display: flex; justify-content: space-between; font-size: 1.05em; margin-bottom: 8px;">
            <span>現在値: <b>{best_item['current_price']:,.2f}</b></span>
            <span>司令判定: {best_item['signal_html']}</span>
        </div>
        <div style="font-size: 0.95em; color: #f8fafc; background: #020617; padding: 12px; border-radius: 8px; border: 1px solid #38bdf8;">
            {best_item['analysis']}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='font-weight:bold; color:#94a3b8; margin:12px 0 6px 0;'>📋"
    " その他の基幹アセット動向</p>",
    unsafe_allow_html=True,
)
for item in analyzed_results[1:]:
  st.markdown(
      f"""
        <div class="asset-card">
            <div style="display: flex; justify-content: space-between; font-size: 0.95em;">
                <span><b>{item['name']}</b> ({item['current_price']:,.2f})</span>
                <span>{item['signal_html']}</span>
            </div>
        </div>
        """,
      unsafe_allow_html=True,
  )
