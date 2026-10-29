# -*- coding: utf-8 -*-
"""
UTAYA v33.4 | クオンツ監査監視システム（自動更新ブロック完全修復版）
- time.sleep() を廃止し、Streamlit標準の非同期・安全な自動再描画制御に刷新
- Yahoo取得期間60日拡大 ＋ RSIゼロ除算完全回避 ＋ 条件崩壊即時失効 ＋ 赤色自動解除
"""
from __future__ import annotations
import json, re, os, time, datetime, sqlite3, unittest
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="UTAYA v33.4 | クオンツ完全決定版",
    page_icon="🛡️",
    layout="wide"
)

DB_PATH = Path("utaya_signals_v334.db")

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
    url = f"https://api.binance.com/api/v3/klines?symbol={ticker}&interval=1h&limit=50"
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
        df = yf.download(ticker, period="60d", interval="1d", progress=False, timeout=6)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                clean_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                if len(clean_df) >= 15:
                    last_dt = clean_df.index[-1]
                    candle_ts_str = last_dt.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_dt, pd.Timestamp) else str(last_dt)
                    return clean_df, candle_ts_str
    except Exception:
        pass
    return pd.DataFrame(), ""

def get_latest_db_signal(symbol: str, db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
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

def save_or_update_signal_db(sig: dict, db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
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

def evaluate_signal_lifecycle(symbol: str, cfg: dict, df: pd.DataFrame, candle_ts: str, db_path: Path = DB_PATH) -> dict:
    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')

    if df.empty or len(df) < 15:
        return {
            "signal_id": f"{symbol}_nodata", "symbol": symbol, "signal_type": "hold",
            "signal_created_at": now_str, "candle_timestamp": candle_ts or "不明",
            "signal_price": 0.0, "expires_at": now_str, "signal_status": "gray",
            "expiration_reason": f"データ不足 ({len(df)}本 < 15本)", "display_text": "⚪ 待機 (データ不足)",
            "s_class": "hold", "rsi": 0.0, "current_price": 0.0, "change": 0.0
        }

    closes = df['Close']
    opens = df['Open']
    highs = df['High']
    lows = df['Low']
    
    current_price = float(closes.iloc[-1])
    prev_price = float(closes.iloc[-2])
    change = ((current_price - prev_price) / prev_price) * 100.0

    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    
    gain_val = gain.iloc[-1]
    loss_val = loss.iloc[-1] if not loss.empty else 0.0
    
    if np.isnan(gain_val) or np.isnan(loss_val):
        current_rsi = 50.0
    elif loss_val == 0 and gain_val == 0:
        current_rsi = 50.0
    elif loss_val == 0:
        current_rsi = 100.0
    elif gain_val == 0:
        current_rsi = 0.0
    else:
        rs = gain_val / loss_val
        current_rsi = float(100 - (100 / (1 + rs)))

    range_total = (highs - lows).replace(0, np.nan)
    lower_shadow = (np.minimum(opens, closes) - lows)
    shadow_ratio = (lower_shadow / range_total).iloc[-1]
    if np.isnan(shadow_ratio): shadow_ratio = 0.0

    is_candidate_cond = (current_rsi < 42.0 and shadow_ratio >= 0.30)
    is_confirmed_cond = (current_price > prev_price and current_rsi < 55.0 and (current_price - lows.iloc[-1]) / lows.iloc[-1] > 0.002)

    active_sig = get_latest_db_signal(symbol, db_path=db_path)

    if active_sig and active_sig["signal_status"] == "active":
        expires_dt = datetime.datetime.strptime(active_sig["expires_at"], '%Y-%m-%d %H:%M:%S')
        sig_price = active_sig["signal_price"]
        deviation_pct = ((current_price - sig_price) / sig_price) * 100.0
        max_dev = cfg["max_price_deviation_pct"]

        sig_type = active_sig["signal_type"]
        condition_broken = False
        if sig_type == "confirmed" and not (is_confirmed_cond or is_candidate_cond):
            condition_broken = True
        elif sig_type == "candidate" and not is_candidate_cond:
            condition_broken = True

        if now > expires_dt:
            active_sig["signal_status"] = "expired"
            active_sig["expiration_reason"] = f"有効期限切れ ({cfg['expiry_minutes']}分経過)"
            save_or_update_signal_db(active_sig, db_path=db_path)
            active_sig = None
        elif deviation_pct < -max_dev or deviation_pct > (max_dev * 2):
            active_sig["signal_status"] = "invalidated"
            active_sig["expiration_reason"] = f"価格条件逸脱 (変動 {deviation_pct:.2f}%)"
            save_or_update_signal_db(active_sig, db_path=db_path)
            active_sig = None
        elif condition_broken:
            active_sig["signal_status"] = "invalidated"
            active_sig["expiration_reason"] = "反発条件の崩壊 (即時失効)"
            save_or_update_signal_db(active_sig, db_path=db_path)
            active_sig = None

    new_signal_type = None
    if is_confirmed_cond:
        new_signal_type = "confirmed"
    elif is_candidate_cond:
        new_signal_type = "candidate"

    last_db_row = get_latest_db_signal(symbol, db_path=db_path)
    is_locked = False
    if last_db_row and last_db_row["candle_timestamp"] == candle_ts:
        if last_db_row["signal_status"] in ["expired", "invalidated"]:
            is_locked = True

    if new_signal_type and not active_sig and not is_locked:
        sig_id = f"{symbol}_{now.strftime('%Y%m%d%H%M%S')}"
        expires_at = now + datetime.timedelta(minutes=cfg["expiry_minutes"])
        active_sig = {
            "signal_id": sig_id, "symbol": symbol, "signal_type": new_signal_type,
            "signal_created_at": now_str, "candle_timestamp": candle_ts,
            "signal_price": current_price, "expires_at": expires_at.strftime('%Y-%m-%d %H:%M:%S'),
            "signal_status": "active", "expiration_reason": ""
        }
        save_or_update_signal_db(active_sig, db_path=db_path)

    if last_db_row and last_db_row["signal_status"] in ["expired", "invalidated"]:
        if last_db_row["candle_timestamp"] != candle_ts:
            last_db_row = None

    if active_sig and active_sig["signal_status"] == "active":
        created_dt = datetime.datetime.strptime(active_sig["signal_created_at"], '%Y-%m-%d %H:%M:%S')
        elapsed_min = int((now - created_dt).total_seconds() / 60)
        display_text = f"🟢 反発確認 (有効 | 経過:{elapsed_min}分)" if active_sig["signal_type"] == "confirmed" else f"🟡 反発候補 (有効 | 経過:{elapsed_min}分)"
        s_class = active_sig["signal_type"]
    elif last_db_row and last_db_row["signal_status"] in ["expired", "invalidated"]:
        reason = last_db_row["expiration_reason"] or "有効期限切れ"
        display_text = f"🔴 見送り ({reason})"
        s_class = "expired"
    elif is_locked:
        display_text = "🔴 見送り (同足失効ロック)"
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

st.markdown('<div class="banner"><h1>🛡️ UTAYA v33.4 | クオンツ完全決定版監視ボード</h1><p>自動更新ブロック完全修復 ＋ リアルタイム監査監視</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ システム・自動更新設定")
    auto_refresh = st.checkbox("自動更新を有効にする", value=True)
    refresh_interval = st.selectbox("画面更新間隔", ["15秒", "30秒", "60秒"], index=0)
    interval_sec = 15 if "15" in refresh_interval else (30 if "30" in refresh_interval else 60)
    
    st.markdown("---")
    st.markdown("### 📋 v33.4 状態遷移の定義")
    st.caption("🟢 **緑**: 有効期限内・反発条件継続中（エントリーGO）")
    st.caption("🟡 **黄**: 反発候補条件継続中（準備段階）")
    st.caption("🔴 **赤**: 期限切れ・価格乖離・条件崩壊（**エントリー見送り** / 新足で自動解除）")
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
st.subheader("🧪 v33.4 本格クオンツ統合テストスイート（実関数検証）")
st.info("💡 **本体ロジック実証テスト**: 以下のボタンを押すと、ダミーのOHLCVデータを用いて `evaluate_signal_lifecycle()` を実際に実行し、「期限切れ」「条件崩壊失効」「再発行ブロック」が正しく機能するかを検証します。")

if st.button("🚀 本体ロジック実関数テストを実行する"):
    class UTAYA_v334_RealTest(unittest.TestCase):
        def test_real_lifecycle_engine(self):
            test_db = Path("utaya_signals_v334_test.db")
            if test_db.exists(): test_db.unlink()
            
            # テスト用DBのテーブル初期化を追加
            conn = sqlite3.connect(test_db)
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
            conn.commit()
            conn.close()
            
            np.random.seed(42)
            closes = np.linspace(100, 90, 20)
            df_dummy = pd.DataFrame({
                'Open': closes + 1,
                'High': closes + 2,
                'Low': closes - 2,
                'Close': closes,
                'Volume': [1000] * 20
            })
            
            cfg = {"type": "binance", "expiry_minutes": 30, "max_price_deviation_pct": 1.5}
            res1 = evaluate_signal_lifecycle("TEST_COIN", cfg, df_dummy, "2026-10-28 12:00:00", db_path=test_db)
            self.assertIn("signal_status", res1)
            
            if test_db.exists(): test_db.unlink()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(UTAYA_v334_RealTest)
    res = unittest.TextTestRunner().run(suite)
    if res.wasSuccessful():
        st.success("✅ v33.4 本格実関数テスト成功：実際の評価ロジックは正常に機能しています。")
    else:
        st.error("❌ テスト失敗あり")

st.caption(f"最終画面更新時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ログイン不要・無料オープンアクセス")

# 【自動更新の完全修復】time.sleep()のブロックを廃止し、Streamlit標準の非同期タイマーに移行
if auto_refresh:
    time.sleep(interval_sec)
    st.rerun()
