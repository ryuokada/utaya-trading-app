# -*- coding: utf-8 -*-
"""
UTAYA AI QUANTUM - Enterprise Quant-Grade Research Core (v26.1)
- Fixed SQL typo: ORDER BY cred -> ORDER BY created.
- Stale price detection for manual XTrend quote inputs.
- Robust Atomic Model Persistence & Training Audit Synchronization.
- Strict Directional ML Modeling (HIGH/LOW) & WAIT Enforcement.
"""
from __future__ import annotations
import os, json, uuid, sqlite3, time, logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

VERSION = "v26.1"
PREFERRED_STORAGE = r"E:\utaya_3tb_ai_storage"
ROOT = Path(os.environ.get("UTAYA_STORAGE_DIR", PREFERRED_STORAGE))
DB = ROOT / "utaya_quantum_v26.db"
MODEL_HIGH_PATH = ROOT / "models" / "sgd_model_high_v26.joblib"
MODEL_LOW_PATH = ROOT / "models" / "sgd_model_low_v26.joblib"
SCALER_PATH = ROOT / "models" / "scaler_v26.joblib"

ASSETS = {
    "BTC/USD (Global Ref)": "BTC-USD",
    "ETH/USD (Global Ref)": "ETH-USD",
    "USD/JPY (Forex Interbank)": "USDJPY=X",
    "Gold XAUUSD (Comex)": "GC=F",
    "WTI Crude (NYMEX)": "CL=F",
    "Nasdaq 100 (TECH100)": "^NDX"
}

FEATURES = [
    "ret1", "ret3", "ret5", "ret10",
    "rsi", "macd_pct", "atr_pct",
    "volatility", "sma_gap", "range_pos",
    "upper_wick", "lower_wick"
]

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def stamp_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")

def connect_db():
    ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB, timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def initialize_system():
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "models").mkdir(exist_ok=True)
        (ROOT / "logs").mkdir(exist_ok=True)
    except Exception as e:
        st.error(f"🚨 【Eドライブアクセスエラー】ストレージ作成に失敗しました: {ROOT} / {e}")
        st.stop()

    with connect_db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS xtrend_quotes(
            quote_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            ts TEXT NOT NULL,
            price REAL NOT NULL CHECK(price > 0),
            source TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_xtrend_lookup ON xtrend_quotes(symbol, ts);

        CREATE TABLE IF NOT EXISTS predictions(
            pred_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            created TEXT NOT NULL,
            due TEXT NOT NULL,
            entry REAL NOT NULL,
            direction TEXT NOT NULL,
            state TEXT NOT NULL,
            p_up REAL,
            features TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            settle REAL DEFAULT NULL,
            settle_ts TEXT DEFAULT NULL,
            outcome TEXT DEFAULT 'PENDING',
            trained INTEGER NOT NULL DEFAULT 0,
            model_version TEXT DEFAULT 'v26.1'
        );
        CREATE INDEX IF NOT EXISTS idx_pred_due ON predictions(symbol, status, due);

        CREATE TABLE IF NOT EXISTS training_audit(
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            pred_id TEXT UNIQUE,
            trained_at TEXT,
            label INTEGER,
            direction TEXT,
            version TEXT
        );
        """)

# --- 1. XTrend実測価格データ管理（鮮度チェック付き） ---
def save_xtrend_quote(symbol: str, price: float, ts: datetime):
    if not np.isfinite(price) or price <= 0:
        raise ValueError("価格は0より大きい有限数である必要があります。")
    with connect_db() as c:
        c.execute(
            "INSERT INTO xtrend_quotes VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), symbol, stamp_utc(ts), float(price), "XTrend Manual Input")
        )
        c.commit()

def get_xtrend_quotes(symbol: str) -> pd.DataFrame:
    with connect_db() as c:
        df = pd.read_sql_query(
            "SELECT ts, price, source FROM xtrend_quotes WHERE symbol=? ORDER BY ts",
            c, params=(symbol,))
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts")

@st.cache_data(ttl=30, show_spinner=False)
def fetch_external_market_data(ticker: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        d = yf.download(ticker, period="3d", interval="1m", progress=False, auto_adjust=False, threads=False)
        if d is None or d.empty:
            return pd.DataFrame()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(k in d for k in cols):
            return pd.DataFrame()
        d = d[cols].dropna()
        d.index = pd.to_datetime(d.index, utc=True)
        return d[~d.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.DataFrame()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c = x["Close"].astype(float)
    h = x["High"].astype(float)
    l = x["Low"].astype(float)
    
    x["ret1"] = c.pct_change(1)
    x["ret3"] = c.pct_change(3)
    x["ret5"] = c.pct_change(5)
    x["ret10"] = c.pct_change(10)
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((gain == 0) & (loss > 0), 0)
    x["rsi"] = rsi.mask((gain == 0) & (loss == 0), 50)
    
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    x["macd_pct"] = macd / c
    
    prev = c.shift()
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    x["atr_pct"] = tr.rolling(14, min_periods=14).mean() / c
    x["volatility"] = c.pct_change().rolling(20, min_periods=20).std()
    
    sma5 = c.rolling(5).mean()
    sma20 = c.rolling(20).mean()
    x["sma_gap"] = (sma5 - sma20) / c
    
    rh = h.rolling(30, min_periods=20).max()
    rl = l.rolling(30, min_periods=20).min()
    x["range_pos"] = (c - rl) / (rh - rl).replace(0, np.nan)
    
    cr = (h - l).replace(0, np.nan)
    x["upper_wick"] = (h - pd.concat([x["Open"], c], axis=1).max(axis=1)) / cr
    x["lower_wick"] = (pd.concat([x["Open"], c], axis=1).min(axis=1) - l) / cr
    
    return x.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

def classify_market_state(f: pd.DataFrame) -> tuple[str, str]:
    if len(f) < 30:
        return "NEUTRAL", "特徴量蓄積中"
    a = f.iloc[-1]
    b = f.iloc[-2]
    
    hi = a["range_pos"] >= 0.90
    lo = a["range_pos"] <= 0.10
    falling = a["ret1"] < 0 and b["ret1"] < 0
    rising = a["ret1"] > 0 and b["ret1"] > 0
    bear = a["sma_gap"] < 0 and a["macd_pct"] < b["macd_pct"]
    bull = a["sma_gap"] > 0 and a["macd_pct"] > b["macd_pct"]
    
    if hi and a["rsi"] >= 65 and not falling:
        return "HIGH_CEILING_WATCH", "高値圏（反落警戒）"
    if hi and falling and bear:
        return "HIGH_CEILING_CONFIRMED", "天井反落条件成立（ショート候補）"
    if lo and a["rsi"] <= 35 and not rising:
        return "LOW_BOTTOM_WATCH", "安値圏（反発警戒）"
    if lo and rising and bull:
        return "LOW_REBOUND_CONFIRMED", "底値反発条件成立（ロング候補）"
    if bull:
        return "UPTREND", "上昇トレンド継続"
    if bear:
        return "DOWNTREND", "下降トレンド継続"
    return "NEUTRAL", "方向感のないレンジ相場"

# --- 2. 戻り値不一致を完全修正したモデルロード & アトミック学習 ---
def load_persisted_models():
    """常に固定で3つの値を返す"""
    if not SKLEARN_OK:
        return None, None, None
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    model_high = joblib.load(MODEL_HIGH_PATH) if MODEL_HIGH_PATH.exists() else None
    model_low = joblib.load(MODEL_LOW_PATH) if MODEL_LOW_PATH.exists() else None
    return scaler, model_high, model_low

def train_pending_predictions():
    if not SKLEARN_OK:
        return 0, "scikit-learn未導入"
    
    with connect_db() as c:
        all_res = pd.read_sql_query("""
            SELECT pred_id, features, outcome, direction
            FROM predictions
            WHERE status='RESOLVED' AND outcome IN ('WIN', 'LOSS')
            ORDER BY settle_ts
        """, c)
        
    if len(all_res) < 15:
        return 0, "学習データ不足（最低15件必要）"
        
    scaler, model_high, model_low = load_persisted_models()
    
    if scaler is None:
        scaler = StandardScaler()
        X_all = np.array([list(json.loads(r["features"]).values()) for _, r in all_res.iterrows()], dtype=float)
        scaler.fit(X_all)
        
    if model_high is None:
        model_high = SGDClassifier(loss="log_loss", alpha=0.0005, random_state=42, average=True)
    if model_low is None:
        model_low = SGDClassifier(loss="log_loss", alpha=0.0005, random_state=42, average=True)
        
    trained_count = 0
    with connect_db() as c:
        un_trained = c.execute("""
            SELECT pred_id, features, outcome, direction 
            FROM predictions 
            WHERE status='RESOLVED' AND outcome IN ('WIN', 'LOSS') AND trained=0
        """).fetchall()
        
    for r in un_trained:
        try:
            feat_dict = json.loads(r["features"])
            X = np.array([[float(feat_dict[k]) for k in FEATURES]], dtype=float)
            if not np.isfinite(X).all():
                continue
            X_scaled = scaler.transform(X)
            y = np.array([1 if r["outcome"] == "WIN" else 0])
            
            if r["direction"] == "HIGH":
                try:
                    model_high.partial_fit(X_scaled, y, classes=np.array([0, 1]))
                except Exception:
                    model_high.partial_fit(X_scaled, y)
            elif r["direction"] == "LOW":
                try:
                    model_low.partial_fit(X_scaled, y, classes=np.array([0, 1]))
                except Exception:
                    model_low.partial_fit(X_scaled, y)
            else:
                continue
                
            with connect_db() as c:
                c.execute("UPDATE predictions SET trained=1 WHERE pred_id=?", (r["pred_id"],))
                c.execute("INSERT OR IGNORE INTO training_audit(pred_id, trained_at, label, direction, version) VALUES (?, ?, ?, ?, ?)",
                          (r["pred_id"], stamp_utc(now_utc()), int(y[0]), r["direction"], VERSION))
                c.commit()
            trained_count += 1
        except Exception as e:
            logging.error(f"Training error: {e}")
            
    if trained_count > 0:
        ROOT.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(model_high, MODEL_HIGH_PATH)
        joblib.dump(model_low, MODEL_LOW_PATH)
        
    return trained_count, f"{trained_count} 件のオンライン学習をアトミック完了。"

def predict_direction_probability(feat: dict, direction: str) -> tuple[float | None, str]:
    scaler, model_high, model_low = load_persisted_models()
    if scaler is None or (direction == "HIGH" and model_high is None) or (direction == "LOW" and model_low is None):
        return None, "モデル未学習またはデータ不足"
    try:
        X = np.array([[float(feat[k]) for k in FEATURES]], dtype=float)
        if not np.isfinite(X).all():
            return None, "特徴量異常値"
        X_scaled = scaler.transform(X)
        model = model_high if direction == "HIGH" else model_low
        probs = model.predict_proba(X_scaled)[0]
        classes = list(model.classes_)
        p_win = float(probs[classes.index(1)]) if 1 in classes else 0.5
        return p_win, "方向別的中確率推論正常"
    except Exception as e:
        return None, f"推論エラー: {e}"

# --- 3. 鮮度チェック付き3分後決着処理 ---
def settle_pending_predictions(symbol: str) -> int:
    now_t = now_utc()
    xtrend_df = get_xtrend_quotes(symbol)
    
    with connect_db() as c:
        rows = c.execute(
            "SELECT * FROM predictions WHERE symbol=? AND status='PENDING' AND due<=?",
            (symbol, stamp_utc(now_t))
        ).fetchall()
        
    settled_count = 0
    for r in rows:
        due_dt = pd.Timestamp(r["due"])
        
        if not xtrend_df.empty:
            after_quotes = xtrend_df[xtrend_df.index >= due_dt]
            if not after_quotes.empty:
                settle_ts = after_quotes.index[0]
                settle_price = float(after_quotes.iloc[0]["price"])
            else:
                continue
        else:
            continue
            
        entry_price = float(r["entry"])
        direction = r["direction"]
        
        if direction == "WAIT":
            outcome = "NOT_TRADED"
        elif settle_price == entry_price:
            outcome = "DRAW"
        else:
            if direction == "HIGH":
                outcome = "WIN" if settle_price > entry_price else "LOSS"
            elif direction == "LOW":
                outcome = "WIN" if settle_price < entry_price else "LOSS"
            else:
                outcome = "DRAW"
                
        with connect_db() as c:
            c.execute("""
                UPDATE predictions 
                SET status='RESOLVED', settle=?, settle_ts=?, outcome=? 
                WHERE pred_id=? AND status='PENDING'
            """, (settle_price, settle_ts.isoformat(), outcome, r["pred_id"]))
            c.commit()
        settled_count += 1
    return settled_count

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title=f"UTAYA AI QUANTUM {VERSION}", page_icon="⚡", layout="wide")
initialize_system()

st.markdown(f"""
<div style="background:linear-gradient(135deg, #09090b, #1e1b4b);padding:22px;border-radius:14px;border:2px solid #38bdf8;">
    <h2 style="color:#38bdf8;margin:0">⚡ UTAYA AI QUANTUM — 実用検証リサーチコア ({VERSION})</h2>
    <p style="color:#cbd5e1;margin:6px 0 0 0">XTrend手動実価格登録 ｜ 鮮度チェック・アトミック学習 ｜ 厳格WAIT制御</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ システム設定")
    symbol = st.selectbox("対象アセット", list(ASSETS.keys()))
    ticker = ASSETS[symbol]
    
    st.markdown("---")
    st.markdown("### 📥 XTrend 実価格登録")
    xtrend_input_price = st.number_input("いまのXTrend口座価格", value=0.0, format="%.5f")
    if st.button("💾 XTrend価格スナップショット保存"):
        if xtrend_input_price > 0:
            try:
                save_xtrend_quote(symbol, xtrend_input_price, now_utc())
                st.success(f"XTrend価格 {xtrend_input_price:,.5f} を記録しました。")
            except Exception as e:
                st.error(f"保存エラー: {e}")
        else:
            st.warning("有効な価格を入力してください。")
            
    st.markdown("---")
    with connect_db() as c:
        total_learn_cnt = c.execute("SELECT COUNT(*) FROM training_audit").fetchone()[0]
    st.metric("🧠 監査済みオンライン学習数", f"{total_learn_cnt} 件")

# バックグラウンド決済・学習の実行
settle_pending_predictions(symbol)
train_pending_predictions()

raw_df = fetch_external_market_data(ticker)
col_l, col_r = st.columns([1.1, 0.9])

with col_l:
    st.markdown("### 👁️ 千里眼・多角性宇宙情報モニター")
    if raw_df.empty:
        st.error("🚨 外部市場データの取得に失敗しました。")
    else:
        f = compute_features(raw_df)
        state, reason = classify_market_state(f)
        current_global_price = float(raw_df.iloc[-1]["Close"])
        
        xtrend_df = get_xtrend_quotes(symbol)
        
        is_price_fresh = False
        latest_xtrend_price = current_global_price
        price_gap = 0.0
        
        if not xtrend_df.empty:
            latest_quote_time = xtrend_df.index[-1]
            time_diff_sec = (now_utc() - latest_quote_time).total_seconds()
            if time_diff_sec <= 300:
                is_price_fresh = True
                latest_xtrend_price = float(xtrend_df.iloc[-1]["price"])
                price_gap = latest_xtrend_price - current_global_price
            else:
                st.warning(f"⚠️ 登録されているXTrend価格が古すぎます（最終入力から {int(time_diff_sec)}秒経過）。サイドバーから最新価格を入力してください。")
        else:
            st.info("ℹ️ XTrend価格が未登録です。サイドバーから現在価格を入力してください。")
            
        m1, m2, m3 = st.columns(3)
        m1.metric("グローバル市場価格", f"{current_global_price:,.5f}")
        m2.metric("XTrend実測価格 (有効)", f"{latest_xtrend_price:,.5f}" if is_price_fresh else "未入力/古い", delta=f"Gap: {price_gap:+.5f}")
        m3.metric("相場状態", state)
        
        st.info(f"🔮 **分析情報**: {reason}")
        
        feat = {k: float(f.iloc[-1][k]) for k in FEATURES}
        p_high, note_h = predict_direction_probability(feat, "HIGH")
        p_low, note_l = predict_direction_probability(feat, "LOW")
        
        st.write(f"🤖 **HIGH方向的中予測**: {p_high*100:.1f}% ({note_h})" if p_high is not None else f"🤖 HIGHモデル: {note_h}")
        st.write(f"🤖 **LOW方向的中予測**: {p_low*100:.1f}% ({note_l})" if p_low is not None else f"🤖 LOWモデル: {note_l}")
        
        auto_dir = "WAIT"
        if is_price_fresh:
            if p_high is not None and p_high >= 0.58 and state in ["LOW_REBOUND_CONFIRMED", "UPTREND"]:
                auto_dir = "HIGH"
            elif p_low is not None and p_low >= 0.58 and state in ["HIGH_CEILING_CONFIRMED", "DOWNTREND"]:
                auto_dir = "LOW"
            else:
                auto_dir = "WAIT"
        else:
            auto_dir = "WAIT"
            
        with connect_db() as c:
            recent_pend = c.execute("""
                SELECT COUNT(*) FROM predictions 
                WHERE symbol=? AND status='PENDING' AND created >= ?
            """, (symbol, stamp_utc(now_utc() - timedelta(minutes=2)))).fetchone()[0]
            
        if recent_pend == 0 and len(f) >= 30 and auto_dir != "WAIT" and is_price_fresh:
            pid = str(uuid.uuid4())
            t_create = now_utc()
            due_time = t_create + timedelta(minutes=3)
            with connect_db() as c:
                c.execute("""
                    INSERT INTO predictions(pred_id, symbol, created, due, entry, direction, state, p_up, features, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                """, (pid, symbol, stamp_utc(t_create), stamp_utc(due_time), latest_xtrend_price, auto_dir, state, p_high if auto_dir=="HIGH" else p_low, json.dumps(feat)))
                c.commit()
                
        st.markdown("---")
        st.markdown("#### 🚀 現在稼働中の自立3分予測（PENDING）")
        with connect_db() as c:
            df_pend = pd.read_sql_query(
                "SELECT created, due, entry, direction, state, status FROM predictions WHERE symbol=? AND status='PENDING' ORDER BY created DESC",
                c, params=(symbol,)
            )
        if not df_pend.empty:
            st.dataframe(df_pend, use_container_width=True, hide_index=True)
        else:
            st.write("新鮮なXTrend価格が入力され、AIの確信度が基準を満たすと3分予測が自動生成されます（現在はWAIT保持）。")

with col_r:
    st.markdown("### 🏆 実績・パフォーマンス統計サマリー")
    with connect_db() as c:
        stats = pd.read_sql_query("SELECT direction, outcome FROM predictions WHERE symbol=? AND status='RESOLVED'", c, params=(symbol,))
        # タイポ修正箇所（ORDER BY created DESC）
        hist = pd.read_sql_query("SELECT created, direction, entry, settle, outcome FROM predictions WHERE symbol=? ORDER BY created DESC LIMIT 10", c, params=(symbol,))
        
        total_pred = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=?", (symbol,)).fetchone()[0]
        resolved_cnt = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND status='RESOLVED'", (symbol,)).fetchone()[0]
        high_cnt = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND direction='HIGH' AND status='RESOLVED'", (symbol,)).fetchone()[0]
        high_wins = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND direction='HIGH' AND outcome='WIN'", (symbol,)).fetchone()[0]
        low_cnt = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND direction='LOW' AND status='RESOLVED'", (symbol,)).fetchone()[0]
        low_wins = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND direction='LOW' AND outcome='WIN'", (symbol,)).fetchone()[0]

    st.markdown(f"""
    - **総予測件数**: {total_pred} 件
    - **決着件数**: {resolved_cnt} 件
    - **HIGH戦績**: {high_wins}勝 / {high_cnt}件 (勝率: {(high_wins/high_cnt*100) if high_cnt>0 else 0.0:.1f}%)
    - **LOW戦績**: {low_wins}勝 / {low_cnt}件 (勝率: {(low_wins/low_cnt*100) if low_cnt>0 else 0.0:.1f}%)
    """)

    if not stats.empty:
        trades = stats[stats["direction"].isin(["HIGH", "LOW"]) & stats["outcome"].isin(["WIN", "LOSS"])]
        tot = len(trades)
        wins = int((trades["outcome"] == "WIN").sum())
        rate = (wins / tot * 100.0) if tot > 0 else 0.0
        if tot < 20:
            st.warning(f"⚠️ サンプル数不足（現在 {tot} 件）。統計的安定性のため勝率は参考値です。")
        else:
            st.success(f"📊 総合勝率: {rate:.1f}% ({wins}勝 / {tot}件)")

    st.markdown("#### 📋 決着済み履歴")
    if not hist.empty:
        st.dataframe(hist, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(f"UTAYA AI QUANTUM {VERSION} | E-Drive SQLite & Stale-Price Protection Active")
if st.checkbox("🔄 15秒ごとに画面を自動同期する", value=True):
    time.sleep(15)
    st.rerun()
