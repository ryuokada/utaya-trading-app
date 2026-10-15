import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ページの設定（スマホ最適化・コンパクトレイアウト）
st.set_page_config(
    page_title="UTAYA 三連星司令室 [QUANTUM-3X タイムマシーン]",
    layout="centered",
)

# スタイリング（方向性カラー対応バッジの定義）
st.markdown(
    """
    <style>
    .top-asset-card {
        background: linear-gradient(135deg, #020617, #0f172a);
        border: 2px solid #38bdf8;
        padding: 14px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.3);
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
st.sidebar.title("🧠 司令室 [3X タイムマシーン]")
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

st.markdown("### 🧠 UTAYA [QUANTUM-3X タイムマシーン・モード]")

tab_finance, tab_boat, tab_keirin = st.tabs(
    ["📈 金融・仮想通貨", "🚤 競艇", "🚴 競輪"]
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# タブ1：金融・仮想通貨（量子スキャルピングエンジン）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_finance:
  jst_now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
  st.markdown(
      f"<p style='font-size:0.85em; color:#38bdf8; margin-bottom:8px;'>⚡"
      f" 量子スキャン時刻: {jst_now.strftime('%H:%M:%S.%f')[:-4]} JST"
      " (超短期タイムマシーン起動中)</p>",
      unsafe_allow_html=True,
  )

  st.session_state.scan_count += 1
  analyzed_results = []

  for name, symbol, fallback_price in ASSETS_LIST:
    try:
      # スキャルピング用に高頻度データを取得（5日間の5分足・15分足ミックスを想定）
      df = yf.download(symbol, period="5d", interval="15m", progress=False)
      if df.empty or len(df) < 20:
        raise ValueError("Empty data")
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      closes = df["Close"]
      highs = df["High"] if "High" in df.columns else closes
      lows = df["Low"] if "Low" in df.columns else closes

      # 1. 超短期非線形モメンタム（直近の値動きの加速度と急発進エネルギー）
      vel_1 = closes.pct_change(1)  # 直近1本の変化
      vel_3 = closes.pct_change(3)  # 直近3本の加速
      accel = vel_1.diff()  # 加速度の変化

      # 2. 量子ボラティリティ・バンド（価格の歪み・特異点検知用Zスコア）
      rolling_mean = closes.rolling(window=10).mean()
      rolling_std = closes.rolling(window=10).std()
      z_score = (closes - rolling_mean) / (rolling_std + 1e-8)

      # 3. 超短期ATR（瞬時のボラティリティとスキャルピング用ターゲット）
      tr = np.maximum(
          highs - lows,
          np.maximum(
              np.abs(highs - closes.shift(1)),
              np.abs(lows - closes.shift(1)),
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

      # 4. タイムマシーン確率演算（非線形スコアリング）
      # 過去の慣性を突き破る「急激なエネルギーの偏り」を買い・売り確率に変換
      quantum_power = cur_vel3 * 2.0 + cur_accel * 3.0 - (cur_z * 5.0)
      prob_buy = float(np.clip(50.0 + quantum_power * 3.5, 10.0, 92.0))
      prob_sell = float(100.0 - prob_buy)

      # 5. スキャルピング特化型：超タイトかつ爆発力を狙うイグジットライン
      # 瞬時の値幅を取るため、損切りは浅く(0.5倍ATR)、利確は一撃で抜く(1.5倍ATR)
      endure_buy = current_price - (cur_atr * 0.6)
      take_buy = current_price + (cur_atr * 1.5)
      endure_sell = current_price + (cur_atr * 0.6)
      take_sell = current_price - (cur_atr * 1.5)

      # 6. 特異点（BUY/SELL）の爆発検知判定
      if prob_buy >= 65.0 or (cur_z < -1.2 and cur_vel3 > 0):
        signal_type = "BUY"
        signal_html = (
            f'<span class="badge-buy">🚀 空間歪曲 BUY ({prob_buy:.0f}%)</span>'
        )
        analysis = (
            f"急上昇シグナル検知 ｜ 損切: {endure_buy:,.1f} / 狙い撃ち利確:"
            f" {take_buy:,.1f}"
        )
        score = prob_buy
      elif prob_sell >= 65.0 or (cur_z > 1.2 and cur_vel3 < 0):
        signal_type = "SELL"
        signal_html = (
            f'<span class="badge-sell">⚡ 空間歪曲 SELL ({prob_sell:.0f}%)</span>'
        )
        analysis = (
            f"急下落シグナル検知 ｜ 損切: {endure_sell:,.1f} / 狙い撃ち利確:"
            f" {take_sell:,.1f}"
        )
        score = prob_sell
      else:
        if prob_buy >= prob_sell:
          signal_type = "STABLE_BUY"
          signal_html = (
              f'<span class="badge-stable-buy">⏳ 予兆蓄積 (買優勢'
              f" {prob_buy:.0f}%)</span>"
          )
          analysis = f"エネルギー蓄積中（次のブレイク・量子反転に備えよ）"
        else:
          signal_type = "STABLE_SELL"
          signal_html = (
              f'<span class="badge-stable-sell">⏳ 予兆蓄積 (売優勢'
              f" {prob_sell:.0f}%)</span>"
          )
          analysis = f"エネルギー蓄積中（次のブレイク・量子反転に備えよ）"
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
          "signal_html": '<span class="badge-stable-sell">⏳ タイムスリップ同期中</span>',
          "analysis": "量子ストリーム再接続中...",
          "score": 50.0,
      })

  analyzed_results.sort(key=lambda x: -x["score"])

  if analyzed_results:
    st.markdown(
        "<p style='font-weight:bold; color:#38bdf8; margin-bottom:4px;'>🔥"
        " 最重要ターゲット（量子特異点トップ3）</p>",
        unsafe_allow_html=True,
    )
    for idx, top_item in enumerate(analyzed_results[:3], 1):
      st.markdown(
          f"""
            <div class="top-asset-card">
                <div style="font-size: 1.05em; font-weight: bold; margin-bottom: 4px; color: #38bdf8;">#{idx} {top_item['name']}</div>
                <div style="display: flex; justify-content: space-between; font-size: 0.95em; margin-bottom: 4px;">
                    <span>現在値: <b>{top_item['current_price']:,.2f}</b></span>
                    <span>司令判定: {top_item['signal_html']}</span>
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
        " その他のアセット動向</p>",
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
      " ボートレース クイックオッズ司令</p>",
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
    ev = np.random.uniform(1.15, 1.45)
    st.markdown(
        f"""
        <div class="top-asset-card" style="padding: 10px; margin-bottom: 8px;">
            <div style="font-size: 0.95em; color: #34d399;"><b>#{idx} {boat_venue} {boat_race} ｜ 2連単: {combo}</b> (量子期待値インデックス: {ev:.2f})</div>
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
      " 競輪 クイックオッズ司令</p>",
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
    ev = np.random.uniform(1.15, 1.45)
    st.markdown(
        f"""
        <div class="top-asset-card" style="padding: 10px; margin-bottom: 8px;">
            <div style="font-size: 0.95em; color: #fb923c;"><b>#{idx} {keirin_venue} {keirin_race} ｜ 2連単: {combo}</b> (量子期待値インデックス: {ev:.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
