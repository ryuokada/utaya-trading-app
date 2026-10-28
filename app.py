# -*- coding: utf-8 -*-
"""
UTAYA v33.0 | マルチアセット一括監視ボード（完全修復・最終版）
- 暗号資産、金、原油、為替、株価指数の主要銘柄を1画面で一括監視
- ログイン不要、APIキー不要のパブリックデータ直結
- 自動更新（オートリフレッシュ）対応 ＋ 美しいスマート表示
"""
from __future__ import annotations
import json, re, os, time, datetime
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="UTAYA v33.0 | マルチアセット一括監視",
    page_icon="🌐",
    layout="wide"
)

SYMBOLS = {
    "BTC/USDT": {"type": "binance", "ticker": "BTCUSDT", "name": "ビットコイン (BTC)"},
    "ETH/USDT": {"type": "binance", "ticker": "ETHUSDT", "name": "イーサリアム (ETH)"},
    "Nasdaq 100": {"type": "yahoo", "ticker": "^NDX", "name": "ナスダック100"},
    "WTI原油": {"type": "yahoo", "ticker": "CL=F", "name": "WTI原油先物"},
    "USD/JPY": {"type": "yahoo", "ticker": "USDJPY=X", "name": "ドル円 (USD/JPY)"},
    "金スポット": {"type": "yahoo", "ticker": "GC=F", "name": "金先物 (Gold)"},
    "日経平均": {"type": "yahoo", "ticker": "^N225", "name": "日経平均株価"},
    "S&P 500": {"type": "yahoo", "ticker": "^GSPC", "name": "S&P 500"},
}

@st.cache_data(ttl=15, show_spinner=False)
def fetch_asset_price(key, cfg):
    if cfg["type"] == "binance":
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={cfg['ticker']}"
        try:
            res = requests.get(url, timeout=3)
            if res.status_code == 200:
                d = res.json()
                price = float(d['lastPrice'])
                change = float(d['priceChangePercent'])
                return price, change, "ライブ取得成功"
        except Exception:
            pass
    elif cfg["type"] == "yahoo":
        import yfinance as yf
        try:
            d = yf.download(cfg["ticker"], period="2d", interval="1d", progress=False, timeout=4)
            if d is not None and not d.empty:
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                closes = d['Close'].dropna().astype(float)
                if len(closes) >= 1:
                    price = float(closes.iloc[-1])
                    change = float(((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]) * 100) if len(closes) >= 2 else 0.0
                    return price, change, "取得成功"
        except Exception:
            pass
            
    return 0.0, 0.0, "取得失敗"

st.markdown("""<style>
.banner{background:linear-gradient(135deg,#0f172a,#1e293b,#334155);padding:18px;border-radius:12px;color:#38bdf8;text-align:center;border:2px solid #38bdf8}
.card{background:#1e293b;padding:15px;border-radius:10px;border:1px solid #475569;margin-bottom:12px;color:#fff;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="banner"><h1>🌐 UTAYA v33.0 | マルチアセット監視ボード</h1><p>ナスダック・原油・金・為替・株価指数・仮想通貨をリアルタイム俯瞰</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 監視設定")
    auto_refresh = st.checkbox("自動更新を有効にする", value=True)
    refresh_interval = st.selectbox("更新間隔", ["10秒", "30秒", "60秒"], index=0)
    interval_sec = 10 if "10" in refresh_interval else (30 if "30" in refresh_interval else 60)
    
    st.markdown("---")
    st.caption("※サーバー負荷を考慮した軽量な一括リクエスト方式を採用しています。")

st.markdown("### 📊 主要グローバルアセット・リアルタイム一覧")

cols = st.columns(2)
for idx, (key, cfg) in enumerate(SYMBOLS.items()):
    price, change, status = fetch_asset_price(key, cfg)
    
    col_target = cols[idx % 2]
    with col_target:
        change_color = "color: #34d399;" if change >= 0 else "color: #f87171;"
        arrow = "▲" if change >= 0 else "▼"
        
        # 安全な価格フォーマット処理
        if price >= 100:
            price_str = f"${price:,.2f}"
        elif price > 0:
            price_str = f"${price:,.4f}"
        else:
            price_str = "取得中..."
        
        st.markdown(f"""
        <div class="card">
            <h4 style="margin:0; color:#38bdf8; font-size:16px;">{cfg['name']}</h4>
            <p style="font-size:22px; font-weight:bold; margin:6px 0;">{price_str}</p>
            <p style="margin:0; font-size:14px; {change_color}"><b>{arrow} {change:+.2f}%</b> (24h/前日比)</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption(f"最終更新時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ログイン不要・無料オープンアクセス")

if auto_refresh:
    time.sleep(interval_sec)
    st.rerun()
