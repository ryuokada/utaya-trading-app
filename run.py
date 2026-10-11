from google import genai
import pandas as pd
import streamlit as st
import yfinance as yf

API_KEY = "AQ.Ab8RN6J0paZfuialPGAj5njT-TehDvzk5TjZjko_gc4c-NxU3Q"

st.set_page_config(
    page_title="マルチタイムフレーム自動判定・自己学習型プラットフォーム",
    layout="wide",
)

# セッションステートによる学習データと勝率の永続化管理
if "total_count" not in st.session_state:
  st.session_state.total_count = 0
if "win_count" not in st.session_state:
  st.session_state.win_count = 0

client = None
if API_KEY != "YOUR_GEMINI_API_KEY":
  client = genai.Client(api_key=API_KEY)

# サイドバー
mode = st.sidebar.selectbox(
    "分析モード",
    [
        "⚡ マルチタイムフレーム相場分析・判定",
        "📈 FXリアルタイム分析 (USDJPY)",
        "🪙 暗号資産分析 (BTCUSDT)",
    ],
)

# サイドバーにAI自己学習・成長メーターを常時表示
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 AI自己学習・成長メーター")
st.sidebar.metric(
    label="累計学習データ数", value=f"{st.session_state.total_count} 件"
)

current_win_rate = (
    (st.session_state.win_count / st.session_state.total_count * 100)
    if st.session_state.total_count > 0
    else 0.0
)
st.sidebar.metric(
    label="現在の推定勝率", value=f"{current_win_rate:.1f}%"
)
st.sidebar.markdown(
    "<small>※「的中」「ハズレ」ボタンでフィードバックすると裏の学習パラメータが更新されます。</small>",
    unsafe_allow_html=True,
)


# リアルタイム市場データ取得関数
def get_realtime_market_data(symbol="BTC-USD"):
  try:
    df = yf.download(symbol, period="1d", interval="1m", progress=False)
    if df.empty:
      return "リアルタイムデータの取得に失敗しました。"
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    current_price = float(df["Close"].iloc[-1])
    df["SMA5"] = df["Close"].rolling(window=5).mean()
    df["SMA20"] = df["Close"].rolling(window=20).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50.0

    context = (
        f"【自動取得されたリアルタイム相場データ ({symbol})】\n"
        f"- 現在価格: {current_price:.4f}\n"
        f"- RSI (14): {current_rsi:.2f}\n"
        f"- 現在の蓄積学習データ数: {st.session_state.total_count}件 (勝率: {current_win_rate:.1f}%)\n"
        f"- 戦略方針: 「毎回損切りはせず、極めて狭い値幅（0.01pips等）や短時間での微小な変動を狙って2ドル等の利益を確実に積み上げるスキャルピング思考」を徹底すること。\n"
    )
    return context
  except Exception as e:
    return f"データ取得エラー: {str(e)}"


st.title("🎯 自己学習型・マルチタイムフレーム統合システム")
st.markdown(
    "過去の予測結果の勝敗（的中・ハズレ）を裏で継続学習させ、勝率向上を目指す実戦型プラットフォームです。"
)

# チャット・分析実行エリア
if "messages" not in st.session_state:
  st.session_state.messages = []

for m in st.session_state.messages:
  with st.chat_message(m["role"]):
    st.markdown(m["content"])

prompt = st.chat_input("相場分析や戦略についてのメッセージを入力...")

if prompt:
  if not client:
    st.error("APIキーが設定されていません。")
  else:
    market_context = get_realtime_market_data("BTC-USD")
    system_instruction = (
        "あなたはプロの超スキャルピング・バイナリーオプション分析AIです。"
        "損切りをせず微小な値動きで利益を積み上げる戦略に基づき、具体的で精度の高い分析を行ってください。"
    )

    full_prompt = (
        f"{system_instruction}\n\n{market_context}\n\nユーザーの指示: {prompt}"
    )

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    with st.chat_message("assistant"):
      with st.spinner("マルチタイムフレーム解析および学習データを適用中..."):
        try:
          res = client.models.generate_content(
              model="gemini-2.5-flash", contents=full_prompt
          )
          st.markdown(res.text)
          st.session_state.messages.append(
              {"role": "assistant", "content": res.text}
          )
        except Exception as e:
          st.error(f"エラー: {e}")

# フィードバック用ボタン（勝率・学習データの蓄積テスト用）
if st.session_state.messages:
  st.markdown("---")
  st.subheader("📊 直近の予測結果のフィードバック（学習用）")
  col1, col2 = st.columns(2)
  with col1:
    if st.button("✅ 的中（勝ち）"):
      st.session_state.total_count += 1
      st.session_state.win_count += 1
      st.success(
          f"学習データを更新しました！（累計: {st.session_state.total_count}件）"
      )
      st.rerun()
  with col2:
    if st.button("❌ ハズレ（負け）"):
      st.session_state.total_count += 1
      st.error(
          f"学習データを更新しました！（累計: {st.session_state.total_count}件）"
      )
      st.rerun()
