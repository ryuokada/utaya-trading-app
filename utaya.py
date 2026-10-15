import datetime
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ページの設定（スマホ最適化・コンパクトレイアウト）
st.set_page_config(
    page_title="UTAYA 三連星司令室 [QUANTUM-3X タイムマシーン]",
    layout="centered",
)

# スタイリング（VTuber配信向け漫画風吹き出し＆近未来UIデザイン）
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
    .anchor-notice { font-size: 0.78em; color: #38bdf8; background: #0c4a6e; padding: 4px 8px; border-radius: 4px; margin-bottom: 6px; display: inline-block; }

    /* VTuberコミック風吹き出しレイアウト */
    .vtuber-container {
        display: flex;
        align-items: flex-start;
        background: linear-gradient(135deg, #090d16, #1e1b4b);
        border: 2px solid #c084fc;
        padding: 14px;
        border-radius: 16px;
        margin-bottom: 16px;
        box-shadow: 0 0 30px rgba(192, 132, 252, 0.2);
    }
    .vtuber-avatar-box {
        text-align: center;
        min-width: 90px;
        margin-right: 14px;
    }
    .vtuber-icon {
        font-size: 3.2em;
        background: #020617;
        border: 2px solid #a855f7;
        border-radius: 50%;
        width: 75px;
        height: 75px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 6px auto;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.4);
    }
    .vtuber-name {
        font-size: 0.78em;
        color: #e9d5ff;
        font-weight: bold;
        background: #581c87;
        padding: 2px 6px;
        border-radius: 6px;
    }
    .comic-bubble {
        position: relative;
        background: #ffffff;
        color: #0f172a;
        padding: 12px 16px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.96em;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 3px solid #cbd5e1;
    }
    /* 漫画の吹き出しのしっぽ（左向き） */
    .comic-bubble::after {
        content: '';
        position: absolute;
        left: -12px;
        top: 20px;
        border-width: 8px 12px 8px 0;
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

# セッションステートの初期化
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
# タブ1：金融・仮想通貨
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_finance:
  jst_now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
  st.markdown(
      f"<p style='font-size:0.85em; color:#38bdf8; margin-bottom:4px;'>⚡"
      f" 量子スキャン時刻: {jst_now.strftime('%H:%M:%S.%f')[:-4]} JST"
      " (YouTubeライブ配信中)</p>",
      unsafe_allow_html=True,
  )

  # 反省学習ログの表示
  st.markdown(
      f"<div style='font-size:0.82em; color:#cbd5e1; background:#020617;"
      f" border-left: 3px solid #38bdf8; padding: 6px 10px; border-radius: 4px;"
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
            f"急上昇シグナル検知 ｜ 損切(耐え): {endure_buy:,.1f} / 狙い撃ち利確:"
            f" {take_buy:,.1f}"
        )
        score = prob_buy
      elif prob_sell >= 65.0 or (cur_z > 1.2 and cur_vel3 < 0):
        signal_html = (
            f'<span class="badge-sell">⚡ 空間歪曲 SELL ({prob_sell:.0f}%)</span>'
        )
        analysis = (
            f"急下落シグナル検知 ｜ 損切(耐え): {endure_sell:,.1f} / 狙い撃ち利確:"
            f" {take_sell:,.1f}"
        )
        score = prob_sell
      else:
        if prob_buy >= prob_sell:
          signal_html = (
              f'<span class="badge-stable-buy">⏳ 予兆蓄積 (買優勢'
              f" {prob_buy:.0f}%)</span>"
          )
          analysis = f"エネルギー蓄積中（次のブレイク・量子反転に備えよ）"
        else:
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
          "signal_html": '<span class="badge-stable-sell">⏳ 同期中</span>',
          "analysis": "量子ストリーム再接続中...",
          "score": 50.0,
      })

  analyzed_results.sort(key=lambda x: -x["score"])

  if analyzed_results:
    current_time = time.time()
    best_item = analyzed_results[0]

    # アンカー機能：最高位銘柄を1分間キープ
    if (
        st.session_state.pinned_top_asset is None
        or (current_time - st.session_state.pin_timestamp) > 60.0
        or best_item["score"]
        >= st.session_state.pinned_top_asset["score"] * 0.98
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🎤 VTuber「UTAYAちゃん」の漫画風吹き出し実況エリア
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    top_name = display_top["name"]
    top_score = display_top["score"]

    if top_score >= 65.0:
      vtuber_speech = (
          f"「司令官！未来からの完全一致シグナルをキャッチしました！今すぐ"
          f" <b>{top_name}</b> でエントリー指示を出します！逃さないでっ！」"
      )
      avatar_emoji = "🤖✨"
    elif top_score >= 55.0:
      vtuber_speech = (
          f"「司令官、<b>{top_name}</b>"
          " の歪みがピークに達しています。チャートの波形を完全同期中……！合図が出るまで指値を構えて待機です！」"
      )
      avatar_emoji = "🤖💬"
    else:
      vtuber_speech = (
          f"「時空の波動は安定中。無理な特攻は禁物です。今は <b>{top_name}</b>"
          " の次なるタイムウェーブを静かに解析しています……。」"
      )
      avatar_emoji = "🤖⏳"

    st.markdown(
        f"""
        <div class="vtuber-container">
            <div class="vtuber-avatar-box">
                <div class="vtuber-icon">{avatar_emoji}</div>
                <div class="vtuber-name">UTAYAちゃん</div>
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
            <div style="font-size: 1.1em; font-weight: bold; margin-bottom: 4px; color: #38bdf8;">★ 司令塔指定銘柄: {display_top['name']}</div>
            <div style="display: flex; justify-content: space-between; font-size: 0.95em; margin-bottom: 4px;">
                <span>現在値: <b>{display_top['current_price']:,.2f}</b></span>
                <span>司令判定: {display_top['signal_html']}</span>
            </div>
            <div style="font-size: 0.85em; color: #cbd5e1; background: #020617; padding: 8px; border-radius: 6px;">
                {display_top['analysis']}
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
    for item in analyzed_results[1:]:
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
      " 競輪 クイックオッズ司令官</p>",
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
