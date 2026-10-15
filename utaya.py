import datetime
import streamlit as st
import yfinance as yf

# ページの設定
st.set_page_config(page_title="UTAYA 司令室", layout="centered")

st.markdown(
    """
    <style>
    .main-card {
        background: linear-gradient(135deg, #020617, #0f172a);
        border: 2px solid #38bdf8;
        padding: 20px;
        border-radius: 14px;
        color: white;
        text-align: center;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.35);
        margin-bottom: 20px;
    }
    .badge-buy { background-color: #0284c7; color: white; padding: 6px 14px; border-radius: 8px; font-weight: bold; font-size: 1.2em; }
    .badge-sell { background-color: #e11d48; color: white; padding: 6px 14px; border-radius: 8px; font-weight: bold; font-size: 1.2em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 UTAYA 司令室 [QUANTUM-3X]")

jst_now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
st.write(f"⚡ スキャン時刻: {jst_now.strftime('%H:%M:%S')} JST")

# 主要アセットのシンプル取得
ASSETS = [
    ("ドル円 (USD/JPY)", "USDJPY=X"),
    ("ゴールド・金", "GC=F"),
    ("ビットコイン", "BTC-USD"),
    ("日経平均株価", "^N225"),
]

for name, symbol in ASSETS:
  try:
    df = yf.download(symbol, period="5d", interval="1h", progress=False)
    if not df.empty:
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      current_price = float(df["Close"].iloc[-1])
      prev_price = float(df["Close"].iloc[-5])
      change = ((current_price - prev_price) / prev_price) * 100

      if change > 0:
        badge = '<span class="badge-buy">🚀 買い優勢</span>'
      else:
        badge = '<span class="badge-sell">⚡ 売り優勢</span>'

      st.markdown(
          f"""
            <div class="main-card">
                <h3>{name}</h3>
                <p style="font-size: 1.5em; font-weight: bold;">現在値: {current_price:,.2f}</p>
                <p>モメンタム変動: {change:+.2f}%</p>
                <div>{badge}</div>
            </div>
            """,
          unsafe_allow_html=True,
      )
  except Exception:
    st.write(f"⚠️ {name}: データ同期中...")
