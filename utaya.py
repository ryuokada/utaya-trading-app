import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf

# ページの設定
st.set_page_config(
    page_title="UTAYA 自律型クオンツAI・未来予測",
    page_icon="📈",
    layout="wide",
)

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


def calc_rsi(series, period):
  delta = series.diff()
  gain = delta.where(delta > 0, 0).rolling(window=period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
  rs = gain / loss
  return 100 - (100 / (1 + rs))


st.title("🚀 次世代自律型クオンツAI・未来予測プラットフォーム")
st.markdown(
    "スマホ・PCからいつでもリアルタイムで全銘柄の機械学習予測・確率判定を確認できます。"
)

# サイドバー設定
if st.sidebar.button("🔄 データを再スキャン・予測実行"):
  st.rerun()

st.subheader(
    f"📊 リアルタイム市場スキャン結果 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
)

results = []

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
    current_price = float(closes.iloc[-1])

    if prob_down >= 72.0:
      signal = "🔥 売り (LOW)"
    elif prob_up >= 72.0:
      signal = "✨ 買い (HIGH)"
    else:
      signal = "⏳ 見送り (待機)"

    results.append({
        "銘柄": name,
        "現在値": f"{current_price:,.2f}",
        "短期RSI(5)": f"{rsi_short:.1f}",
        "上昇確率": f"{prob_up:.1f}%",
        "下落確率": f"{prob_down:.1f}%",
        "AI判定": signal,
    })
  except Exception as e:
    continue

if results:
  df_result = pd.DataFrame(results)
  st.dataframe(df_result, use_container_width=True, hide_index=True)
else:
  st.warning("現在データを取得中です。少々お待ちください。")
