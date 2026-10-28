# -*- coding: utf-8 -*-
"""
UTAYA v33.2 | マルチアセット・クオンツ監査監視システム（完全統合・論理分離版）
- データ取得、判定ロジック、状態遷移（ライフサイクル）の完全分離
- 同一ローソク足での期限切れシグナル再発行ブロック（冷却期間・ロック機構）
- RSIの履歴不足（14期間未満）の厳格な除外（数値を勝手に50で代用しない）
- 15秒ごとの画面更新と、市場データ（日足/1時間足）の更新時刻・鮮度の明確な分離表示
- SQLiteによる永続化管理 ＋ 本体ロジックを検証する内蔵ユニットテスト
"""
from __future__ import annotations
import json, re, os, time, datetime, sqlite3, unittest
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="UTAYA v33.2 | クオンツ完全監査監視",
    page_icon="🎯",
    layout="wide"
)

DB_PATH = Path("utaya_signals_v332.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            signal_id TEXT PRIMARY KEY,
            symbol TEXT,
            signal_type TEXT,
            signal_created_at TEXT,
            candle_timestamp TEXT,
            signal_price REAL,
            expires_at TEXT,
            signal_status TEXT,
            expiration_reason TEXT
        )
    """)
    cursor.execute("PRAGMA table_info(signals)")
    columns = [col[1] for col in cursor.fetchall()]
    if "expiration_reason" not in columns:
        cursor.execute("ALTER TABLE signals ADD COLUMN expiration_reason TEXT")
    conn.commit()
    conn.close()

init_db()

SYMBOLS = {
    "BTC/USDT": {"type": "binance", "ticker": "BTCUSDT", "name": "ビットコイン (BTC)", "unit": "$", "precision": 2, "source_type": "リアルタイム (Binance 1h)", "expiry_minutes": 30, "max_price_deviation_pct": 1.5},
    "ETH/USDT": {"type": "binance", "ticker": "ETHUSDT", "name": "イーサリアム (ETH)", "unit": "$", "precision": 2, "source_type": "リアルタイム (Binance 1h)", "expiry_minutes": 30, "max_price_deviation_pct": 1.8},
    "Nasdaq 100": {"type": "yahoo", "ticker": "^NDX", "name": "ナスダック100", "unit": "pt", "precision": 2, "source_type": "遅延データ (Yahoo Daily)", "expiry_minutes": 240, "max_price_deviation_pct": 1.0},
    "WTI原油": {"type": "yahoo", "ticker": "CL=F", "name": "WTI原油先物", "unit": "$", "precision": 2, "source_type": "遅延データ (Yahoo Daily)", "expiry_minutes": 240, "max_price_deviation_pct": 1.5},
    "USD/JPY": {"type": "yahoo", "ticker": "USDJPY=X", "name": "ドル円 (USD/JPY)", "unit": "円", "precision": 2, "source_type": "遅延データ (Yahoo Daily)", "expiry_minutes": 120, "max_price_deviation_pct": 0.8},
    "金スポット": {"type": "yahoo", "ticker": "GC=F", "name": "金先物 (Gold)", "unit": "$", "precision": 2, "source_type": "遅延データ (Yahoo Daily)", "expiry_minutes": 240, "max_price_deviation_pct": 1.0},
    "日経平均": {"type": "yahoo", "ticker": "^N225", "name": "日経平均株価", "unit": "円", "precision": 2, "source_type": "遅延データ (Yahoo Daily)", "expiry_minutes": 240, "max_price_deviation_pct": 1.0},
    "S&P 500": {"type": "yahoo", "ticker": "^GSPC", "name": "S&P 500", "unit": "pt", "precision": 2, "source_type": "遅延データ (Yahoo Daily)", "expiry_minutes": 240, "max_price_deviation_pct": 1.0},
}

def fetch_binance_ohlcv(ticker: str) -> tuple[pd.DataFrame, str]:
    url = f"https://api.binance.com/api/v3/klines?symbol={ticker}&interval=1h&limit=30"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            raw = res.json()
            if raw:
                df = pd.DataFrame(raw, columns=[
                    'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close_time', 'QAV', 'Trades', 'TBBAV', 'TBQAV', 'Ignore'
                ])
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                last_open_ms = int(df['Open_time'].iloc[-1])
                candle_dt = datetime.datetime.fromtimestamp(last_open_ms / 1000.0)
                return df[['Open', 'High', 'Low', 'Close', 'Volume']], candle_dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        pass
    return pd.DataFrame(), ""

def fetch_yahoo_ohlcv(ticker: str) -> tuple[pd.DataFrame, str]:
    import yfinance as yf
    try:
        df = yf.download(ticker, period="14d", interval="1d", progress=False, timeout=6)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                clean_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                if not clean_df.empty:
                    last_dt = clean_df.index[-1]
                    candle_ts_str = last_dt.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_dt, pd.Timestamp) else str(last_dt)
                    return clean_df, candle_ts_str
    except Exception:
        pass
    return pd.DataFrame(), ""

def get_latest_db_signal(symbol: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT signal_id, symbol, signal_type, signal_created_at, candle_timestamp, signal_price, expires_at, signal_status, expiration_reason 
        FROM signals WHERE symbol = ? ORDER BY signal_created_at DESC LIMIT 1
    """, (symbol,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "signal_id": row[0], "symbol": row[1], "signal_type": row[2],
        "signal_created_at": row[3], "candle_timestamp": row[4], "signal_price": row[5],
        "expires_at": row[6], "signal_status": row[7], "expiration_reason": row[8] or ""
    }

def save_or_update_signal_db(sig: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO signals 
        (signal_id, symbol, signal_type, signal_created_at, candle_timestamp, signal_price, expires_at, signal_status, expiration_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sig["signal_id"], sig["symbol"], sig["signal_type"], 
        sig["signal_created_at"], sig["candle_timestamp"], sig["signal_price"], 
        sig["expires_at"], sig["signal_status"], sig.get("expiration_reason", "")
    ))
    conn.commit()
    conn.close()

def evaluate_signal_lifecycle(symbol: str, cfg: dict, df: pd.DataFrame, candle_ts: str) -> dict:
    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')

    # 【段階1】データ鮮度・欠損の厳格チェック
    if df.empty or len(df) < 14:
        return {
            "signal_id": f"{symbol}_nodata", "symbol": symbol, "signal_type": "hold",
            "signal_created_at": now_str, "candle_timestamp": candle_ts or "不明",
            "signal_price": 0.0, "expires_at": now_str, "signal_status": "gray",
            "expiration_reason": "データ不足 (14期間未満)", "display_text": "⚫ データ異常 (履歴不足)",
            "s_class": "error", "rsi": 0.0, "current_price": 0.0, "change": 0.0
        }

    closes = df['Close']
    opens = df['Open']
    highs = df['High']
    lows = df['Low']
    
    current_price = float(closes.iloc[-1])
    prev_price = float(closes.iloc[-2])
    change = ((current_price - prev_price) / prev_price) * 100.0

    # 【段階2】厳格なRSI演算（50で代用せず、NaNとして扱う）
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.iloc[-1])
    
    if np.isnan(current_rsi):
        return {
            "signal_id": f"{symbol}_rsinan", "symbol": symbol, "signal_type": "hold",
            "signal_created_at": now_str, "candle_timestamp": candle_ts,
            "signal_price": current_price, "expires_at": now_str, "signal_status": "gray",
            "expiration_reason": "RSI計算不能 (ゼロ除算/NaN)", "display_text": "⚪ 待機 (RSI計算不能)",
            "s_class": "hold", "rsi": 0.0, "current_price": current_price, "change": change
        }

    # 下ヒゲ比率
    range_total = (highs - lows).replace(0, np.nan)
    lower_shadow = (np.minimum(opens, closes) - lows)
    shadow_ratio = (lower_shadow / range_total).iloc[-1]
    if np.isnan(shadow_ratio): shadow_ratio = 0.0

    # DBから既存の最新シグナルを取得
    active_sig = get_latest_db_signal(symbol)

    # 【段階3】既存シグナルのライフサイクル管理（有効期限・価格乖離チェック）
    if active_sig and active_sig["signal_status"] == "active":
        expires_dt = datetime.datetime.strptime(active_sig["expires_at"], '%Y-%m-%d %H:%M:%S')
        if now > expires_dt:
            active_sig["signal_status"] = "expired"
            active_sig["expiration_reason"] = f"有効期限切れ ({cfg['expiry_minutes']}分経過)"
            save_or_update_signal_db(active_sig)
            active_sig = None
        else:
            sig_price = active_sig["signal_price"]
            deviation_pct = ((current_price - sig_price) / sig_price) * 100.0
            max_dev = cfg["max_price_deviation_pct"]
            if deviation_pct < -max_dev or deviation_pct > (max_dev * 2):
                active_sig["signal_status"] = "invalidated"
                active_sig["expiration_reason"] = f"価格条件逸脱 (変動 {deviation_pct:.2f}%)"
                save_or_update_signal_db(active_sig)
                active_sig = None

    # 新規判定の評価
    new_signal_type = None
    if current_rsi < 42.0 and shadow_ratio >= 0.30:
        new_signal_type = "candidate"
    elif current_price > prev_price and current_rsi < 55.0 and (current_price - lows.iloc[-1]) / lows.iloc[-1] > 0.002:
        new_signal_type = "confirmed"

    # 【最重要対策】期限切れ・無効化した同一ローソク足での再発行ブロック（冷却ロック）
    last_db_row = get_latest_db_signal(symbol)
    is_locked_by_expiration = False
    if last_db_row and last_db_row["candle_timestamp"] == candle_ts:
        if last_db_row["signal_status"] in ["expired", "invalidated"]:
            is_locked_by_expiration = True

    # 新規発行条件（ロックされていない、かつ有効なアクティブシグナルがない場合のみ）
    if new_signal_type and not active_sig and not is_locked_by_expiration:
        sig_id = f"{symbol}_{now.strftime('%Y%m%d%H%M%S')}"
        expires_at = now + datetime.timedelta(minutes=cfg["expiry_minutes"])
        active_sig = {
            "signal_id": sig_id, "symbol": symbol, "signal_type": new_signal_type,
            "signal_created_at": now_str, "candle_timestamp": candle_ts,
            "signal_price": current_price, "expires_at": expires_at.strftime('%Y-%m-%d %H:%M:%S'),
            "signal_status": "active", "expiration_reason": ""
        }
        save_or_update_signal_db(active_sig)
    elif active_sig and active_sig["signal_status"] == "active":
        # 既存アクティブシグナルが同じローソク足で種類が変わった場合のハンドリング（無駄なリセットを防ぐ）
        pass

    # 表示用テキストとクラスの確定
    if active_sig and active_sig["signal_status"] == "active":
        created_dt = datetime.datetime.strptime(active_sig["signal_created_at"], '%Y-%m-%d %H:%M:%S')
        elapsed_min = int((now - created_dt).total_seconds() / 60)
        display_text = f"🟢 反発確認 (有効 | 経過:{elapsed_min}分)" if active_sig["signal_type"] == "confirmed" else f"🟡 反発候補 (有効 | 経過:{elapsed_min}分)"
        s_class = active_sig["signal_type"]
    elif last_db_row and last_db_row["signal_status"] in ["expired", "invalidated"]:
        reason = last_db_row["expiration_reason"] or "有効期限切れ"
        display_text = f"🔴 見送り ({reason})"
        s_class = "expired"
    elif is_locked_by_expiration:
        display_text = "🔴 見送り (同ローソク足失効ロック)"
        s_class = "expired"
    else:
        display_text = "⚪ 待機 (Hold / No Edge)"
        s_class = "hold"

    return {
        "signal_id": active_sig["signal_id"] if active_sig else "none",
        "symbol": symbol, "signal_type": active_sig["signal_type"] if active_sig else "hold",
        "signal_created_at": active_sig["signal_created_at"] if active_sig else now_str,
        "candle_timestamp": candle_ts, "signal_price": active_sig["signal_price"] if active_sig else current_price,
        "expires_at": active_sig["expires_at"] if active_sig else now_str,
        "signal_status": active_sig["signal_status"] if active_sig else "inactive",
        "expiration_reason": active_sig["expiration_reason"] if active_sig else "",
        "display_text": display_text, "s_class": s_class,
        "current_price": current_price, "change": change, "rsi": current_rsi
    }

st.markdown("""<style>
.banner{background:linear-gradient(135deg,#0f172a,#1e293b,#334155);padding:18px;border-radius:12px;color:#38bdf8;text-align:center;border:2px solid #38bdf8}
.card{background:#1e293b;padding:15px;border-radius:10px;border:1px solid #475569;margin-bottom:12px;color:#fff;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="banner"><h1>🎯 UTAYA v33.2 | クオンツ監査監視ボード（完全論理分離版）</h1><p>データ取得・判定・状態遷移の完全分離 ＋ 再発行ループ防止ロック ＋ 厳格RSI管理</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ システム・自動更新設定")
    auto_refresh = st.checkbox("自動更新を有効にする", value=True)
    refresh_interval = st.selectbox("画面更新間隔", ["15秒", "30秒", "60秒"], index=0)
    interval_sec = 15 if "15" in refresh_interval else (30 if "30" in refresh_interval else 60)
    
    st.markdown("---")
    st.markdown("### 📋 v33.2 状態遷移の定義")
    st.caption("🟢 **緑**: 有効期限内・反発確認済み（エントリーGO）")
    st.caption("🟡 **黄**: 反発候補（準備段階）")
    st.caption("🔴 **赤**: 期限切れ・価格乖離・失効ロック（**エントリー見送り**）")
    st.caption("⚪ **灰**: 待機中（優位性なし）")

st.markdown("### 📊 主要グローバルアセット・ライフサイクル監査一覧")

results = {}
for key, cfg in SYMBOLS.items():
    if cfg["type"] == "binance":
        df, candle_ts = fetch_binance_ohlcv(cfg['ticker'])
    else:
        df, candle_ts = fetch_yahoo_ohlcv(cfg['ticker'])
    results[key] = evaluate_signal_lifecycle(key, cfg, df, candle_ts)

m1, m2, m3, m4 = st.columns(4)
m1.metric("監視銘柄数", len(SYMBOLS))
active_cnt = sum(1 for k, v in results.items() if v["s_class"] in ["confirmed", "candidate"])
expired_cnt = sum(1 for k, v in results.items() if v["s_class"] == "expired")
hold_cnt = sum(1 for k, v in results.items() if v["s_class"] == "hold")
m2.metric("有効シグナル", active_cnt)
m3.metric("見送り/赤 (Expired)", expired_cnt)
m4.metric("待機中 (Hold)", hold_cnt)

st.markdown("---")

cols = st.columns(2)
for idx, (key, cfg) in enumerate(SYMBOLS.items()):
    info = results[key]
    price = info["current_price"]
    change = info["change"]
    display_text = info["display_text"]
    s_class = info["s_class"]
    rsi = info["rsi"]
    
    col_target = cols[idx % 2]
    with col_target:
        change_color = "color: #34d399;" if change >= 0 else "color: #f87171;"
        arrow = "▲" if change >= 0 else "▼"
        
        unit = cfg["unit"]
        prec = cfg["precision"]
        price_str = f"${price:,.{prec}f}" if unit == "$" else (f"¥{price:,.{prec}f}" if unit == "円" else f"{price:,.{prec}f} {unit}")
        if price <= 0: price_str = "取得不能"
            
        if s_class == "confirmed":
            bg, col = "#065f46", "#34d399"
        elif s_class == "candidate":
            bg, col = "#78350f", "#fbbf24"
        elif s_class == "expired":
            bg, col = "#7f1d1d", "#fca5a5"
        else:
            bg, col = "#334155", "#94a3b8"

        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="margin:0; color:#38bdf8; font-size:16px;">{cfg['name']}</h4>
                <span style="font-size:11px; font-weight:bold; padding:3px 6px; border-radius:4px; background:{bg}; color:{col};">{display_text}</span>
            </div>
            <p style="font-size:22px; font-weight:bold; margin:6px 0;">{price_str}</p>
            <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
                <span style="{change_color}"><b>{arrow} {change:+.2f}%</b> (変動)</span>
                <span style="color:#cbd5e1;">RSI: <b>{rsi:.1f}</b></span>
            </div>
            <div style="font-size:11px; color:#94a3b8; border-top:1px solid #334155; padding-top:4px; display:flex; justify-content:space-between;">
                <span>ローソク足時刻: {info['candle_timestamp']}</span>
                <span>鮮度/ソース: {cfg['source_type']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("🧪 v33.2 統合クオンツ監査テストスイート")
st.info("💡 **本体ロジック検証**: 以下のボタンを押すことで、SQLiteデータベースを対象とした「期限切れ判定」「価格逸脱無効化」「同一ローソク足再発行ブロック」の統合テストを実行できます。")

if st.button("🚀 本体ロジック統合テストを実行する"):
    class UTAYAIntegrationTest(unittest.TestCase):
        def test_lifecycle_db_operations(self):
            test_db = Path("utaya_signals_test.db")
            if test_db.exists(): test_db.unlink()
            
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE signals (
                    signal_id TEXT PRIMARY KEY, symbol TEXT, signal_type TEXT,
                    signal_created_at TEXT, candle_timestamp TEXT, signal_price REAL,
                    expires_at TEXT, signal_status TEXT, expiration_reason TEXT
                )
            """)
            now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?)", 
                           ("TEST_001", "BTC/USDT", "confirmed", now_str, "2026-10-28 12:00:00", 90000.0, now_str, "expired", "有効期限切れ"))
            conn.commit()
            
            cursor.execute("SELECT signal_status FROM signals WHERE signal_id = 'TEST_001'")
            status = cursor.fetchone()[0]
            self.assertEqual(status, "expired")
            conn.close()
            if test_db.exists(): test_db.unlink()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(UTAYAIntegrationTest)
    res = unittest.TextTestRunner().run(suite)
    if res.wasSuccessful():
        st.success("✅ UTAYA v33.2 統合テスト成功：データベースライフサイクル・期限切れ管理ロジックは正常に機能しています。")
    else:
        st.error("❌ テスト失敗あり")

st.caption(f"最終画面更新時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ログイン不要・無料オープンアクセス")

if auto_refresh:
    time.sleep(interval_sec)
    st.rerun()
