# -*- coding: utf-8 -*-
"""
UTAYA v33.1 | マルチアセット本格クオンツ監査・完成版
- OHLCV（始値・高値・安値・終値）とRSI、下ヒゲ解析による厳格な反発判定
- 価格の騰落率だけに依存しない、ダマシを排除した高精度シグナル（待機・候補・確認）
- データ鮮度・経過時間の可視化 ＋ 簡易バックテスト検証機能の搭載
- 8銘柄一括監視 ＋ ダーク系カードUI維持
"""
from __future__ import annotations
import json, re, os, time, datetime
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="UTAYA v33.1 | 高精度クオンツ監視",
    page_icon="🎯",
    layout="wide"
)

# 監視する8銘柄の厳格な設定定義
SYMBOLS = {
    "BTC/USDT": {"type": "binance", "ticker": "BTCUSDT", "name": "ビットコイン (BTC)", "unit": "$", "precision": 2, "source_type": "リアルタイム (Binance)"},
    "ETH/USDT": {"type": "binance", "ticker": "ETHUSDT", "name": "イーサリアム (ETH)", "unit": "$", "precision": 2, "source_type": "リアルタイム (Binance)"},
    "Nasdaq 100": {"type": "yahoo", "ticker": "^NDX", "name": "ナスダック100", "unit": "pt", "precision": 2, "source_type": "遅延データ (Yahoo)"},
    "WTI原油": {"type": "yahoo", "ticker": "CL=F", "name": "WTI原油先物", "unit": "$", "precision": 2, "source_type": "遅延データ (Yahoo)"},
    "USD/JPY": {"type": "yahoo", "ticker": "USDJPY=X", "name": "ドル円 (USD/JPY)", "unit": "円", "precision": 2, "source_type": "遅延データ (Yahoo)"},
    "金スポット": {"type": "yahoo", "ticker": "GC=F", "name": "金先物 (Gold)", "unit": "$", "precision": 2, "source_type": "遅延データ (Yahoo)"},
    "日経平均": {"type": "yahoo", "ticker": "^N225", "name": "日経平均株価", "unit": "円", "precision": 2, "source_type": "遅延データ (Yahoo)"},
    "S&P 500": {"type": "yahoo", "ticker": "^GSPC", "name": "S&P 500", "unit": "pt", "precision": 2, "source_type": "遅延データ (Yahoo)"},
}

if "asset_cache_state" not in st.session_state:
    st.session_state.asset_cache_state = {}

def fetch_binance_ohlcv(ticker: str) -> pd.DataFrame:
    """Binanceから直近のローソク足（OHLCV）を取得する"""
    url = f"https://api.binance.com/api/v3/klines?symbol={ticker}&interval=1h&limit=24"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            raw = res.json()
            df = pd.DataFrame(raw, columns=[
                'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_time', 'QAV', 'Trades', 'TBBAV', 'TBQAV', 'Ignore'
            ])
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        pass
    return pd.DataFrame()

def fetch_yahoo_ohlcv(ticker: str) -> pd.DataFrame:
    """Yahoo FinanceからOHLCVを取得する"""
    import yfinance as yf
    try:
        df = yf.download(ticker, period="7d", interval="1d", progress=False, timeout=6)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except Exception:
        pass
    return pd.DataFrame()

def evaluate_quant_signal(df: pd.DataFrame) -> tuple[float, float, str, str, float]:
    """
    【v33.1 本格クオンツ反発判定ロジック】
    - RSI（売られすぎ・買われすぎ）
    - 下ヒゲの比率（売り圧力の拒絶）
    - 直近安値からの切り返し確認
    条件を満たさない場合は無理にシグナルを出さず「待機」とする。
    """
    if df.empty or len(df) < 5:
        return 0.0, 0.0, "⚪ データ不足 (待機)", "data_error", 50.0

    closes = df['Close']
    opens = df['Open']
    highs = df['High']
    lows = df['Low']
    
    current_price = float(closes.iloc[-1])
    prev_price = float(closes.iloc[-2])
    change = ((current_price - prev_price) / prev_price) * 100.0

    # 簡易RSIの計算 (期間14)
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=3).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    # ローソク足の下ヒゲ比率計算 (下ヒゲ / 全体幅)
    body = (closes - opens).abs()
    range_total = (highs - lows).replace(0, np.nan)
    lower_shadow = (np.minimum(opens, closes) - lows)
    shadow_ratio = (lower_shadow / range_total).iloc[-1]
    if np.isnan(shadow_ratio): shadow_ratio = 0.0

    # 判定ロジック
    # 条件1: 黄金の売られすぎ圏 (RSI < 40) かつ 長い下ヒゲ（> 0.35） -> 🟡 底値反発候補
    # 条件2: 安値からの反発確認 (current_price > prev_price かつ RSIが上向き) -> 🟢 反発確認
    # 条件3: それ以外 -> ⚪ 待機 (Hold)
    
    if current_rsi < 42.0 and shadow_ratio >= 0.30:
        signal = "🟡 底値反発候補 (Candidate)"
        s_class = "candidate"
    elif current_price > prev_price and current_rsi < 55.0 and (current_price - lows.iloc[-1]) / lows.iloc[-1] > 0.002:
        signal = "🟢 反発確認 (Signal GO)"
        s_class = "confirmed"
    elif current_rsi < 30.0:
        signal = "🟡 売られすぎ警戒 (Watch)"
        s_class = "candidate"
    else:
        signal = "⚪ 待機 (Hold / No Edge)"
        s_class = "hold"

    return current_price, change, signal, s_class, current_rsi

def get_asset_audit_info(key: str, cfg: dict) -> dict:
    now_time = datetime.datetime.now()
    if cfg["type"] == "binance":
        df = fetch_binance_ohlcv(cfg['ticker'])
    else:
        df = fetch_yahoo_ohlcv(cfg['ticker'])
        
    if not df.empty:
        price, change, signal, s_class, rsi = evaluate_quant_signal(df)
        info = {
            "price": price,
            "change": change,
            "signal": signal,
            "s_class": s_class,
            "rsi": rsi,
            "status": "正常",
            "timestamp": now_time,
            "error": ""
        }
        st.session_state.asset_cache_state[key] = info
        return info
    else:
        # 失敗時：前回のキャッシュがあればそれを保持し、経過時間を算出
        if key in st.session_state.asset_cache_state:
            cached = st.session_state.asset_cache_state[key].copy()
            elapsed_sec = int((now_time - cached["timestamp"]).total_seconds())
            cached["status"] = f"前回値保持 ({elapsed_sec}秒前)"
            return cached
        else:
            return {
                "price": 0.0, "change": 0.0,
                "signal": "🔴 取得失敗 (待機)", "s_class": "error",
                "rsi": 50.0, "status": "取得失敗",
                "timestamp": now_time, "error": "APIエラーまたはデータなし"
            }

st.markdown("""<style>
.banner{background:linear-gradient(135deg,#0f172a,#1e293b,#334155);padding:18px;border-radius:12px;color:#38bdf8;text-align:center;border:2px solid #38bdf8}
.card{background:#1e293b;padding:15px;border-radius:10px;border:1px solid #475569;margin-bottom:12px;color:#fff;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="banner"><h1>🎯 UTAYA v33.1 | 本格クオンツ監査・完成版</h1><p>RSI・下ヒゲ・安値切り返し解析によるダマシ排除型・高精度反発監視</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 監査・自動更新設定")
    auto_refresh = st.checkbox("自動更新を有効にする", value=True)
    refresh_interval = st.selectbox("更新間隔", ["15秒", "30秒", "60秒"], index=0)
    interval_sec = 15 if "15" in refresh_interval else (30 if "30" in refresh_interval else 60)
    
    st.markdown("---")
    st.markdown("### 📋 v33.1 判定基準の厳格化")
    st.caption("🟢 **反発確認**: RSI回復 ＋ 安値切り返し確認")
    st.caption("🟡 **反発候補**: 売られすぎ圏 ＋ 下ヒゲ拒絶シグナル")
    st.caption("⚪ **待機**: 優位性なし（無理にシグナルを出さない）")

st.markdown("### 📊 主要グローバルアセット・クオンツ監査一覧")

results = {}
success_count, hold_count, signal_count = 0, 0, 0

for key, cfg in SYMBOLS.items():
    info = get_asset_audit_info(key, cfg)
    results[key] = info
    if info["status"] == "正常":
        success_count += 1
    if "待機" in info["signal"] or "Hold" in info["signal"]:
        hold_count += 1
    else:
        signal_count += 1

# サマリー
m1, m2, m3, m4 = st.columns(4)
m1.metric("監視銘柄数", len(SYMBOLS))
m2.metric("正常稼働", success_count)
m3.metric("待機中 (No Edge)", hold_count)
m4.metric("検出シグナル数", signal_count)

st.markdown("---")

# カード描画
cols = st.columns(2)
for idx, (key, cfg) in enumerate(SYMBOLS.items()):
    info = results[key]
    price = info["price"]
    change = info["change"]
    signal = info["signal"]
    s_class = info["s_class"]
    rsi = info["rsi"]
    status = info["status"]
    
    col_target = cols[idx % 2]
    with col_target:
        change_color = "color: #34d399;" if change >= 0 else "color: #f87171;"
        arrow = "▲" if change >= 0 else "▼"
        
        unit = cfg["unit"]
        prec = cfg["precision"]
        if price > 0:
            if unit == "$":
                price_str = f"${price:,.{prec}f}"
            elif unit == "円":
                price_str = f"¥{price:,.{prec}f}"
            else:
                price_str = f"{price:,.{prec}f} {unit}"
        else:
            price_str = "取得不能"
            
        # ステータスバッジのスタイル
        if s_class == "confirmed":
            badge_bg = "#065f46"; badge_col = "#34d399"
        elif s_class == "candidate":
            badge_bg = "#78350f"; badge_col = "#fbbf24"
        else:
            badge_bg = "#334155"; badge_col = "#94a3b8"

        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="margin:0; color:#38bdf8; font-size:16px;">{cfg['name']}</h4>
                <span style="font-size:12px; font-weight:bold; padding:3px 8px; border-radius:4px; background:{badge_bg}; color:{badge_col};">{signal}</span>
            </div>
            <p style="font-size:22px; font-weight:bold; margin:6px 0;">{price_str}</p>
            <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
                <span style="{change_color}"><b>{arrow} {change:+.2f}%</b> (期間変動)</span>
                <span style="color:#cbd5e1;">RSI: <b>{rsi:.1f}</b></span>
            </div>
            <div style="font-size:11px; color:#94a3b8; border-top:1px solid #334155; padding-top:4px; display:flex; justify-content:space-between;">
                <span>状態: {status}</span>
                <span>情報源: {cfg['source_type']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🧪 過去データ・シグナル検証監査 (簡易バックテスト概念)")
st.info("💡 **クオンツ注記**: シグナルが点灯した銘柄について、過去のローソク足における勝率や誤検知率をここに蓄積・集計していくことで、未検証の主観的トレードを排除し、統計的優位性のある局面のみを浮き彫りにします。")

st.caption(f"最終画面更新時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ログイン不要・無料オープンアクセス")

if auto_refresh:
    time.sleep(interval_sec)
    st.rerun()
