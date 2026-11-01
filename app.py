# -*- coding: utf-8 -*-
"""
UTAYA AI QUANTUM - Enterprise Quant-Grade Autonomous Core (v24.0)
- True Data Source Isolation (XTrend Reference vs External Feed).
- Rigorous 3-Minute Forward Settlement & UTC Timezone Synchronization.
- Fixed StandardScaler Persistence & Directional ML Training (SGDClassifier).
- Strict WAIT Enforcement & Comprehensive Performance Analytics.
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

VERSION = "v24.0"
PREFERRED_STORAGE = r"E:\utaya_3tb_ai_storage"
ROOT = Path(os.environ.get("UTAYA_STORAGE_DIR", PREFERRED_STORAGE))
DB = ROOT / "utaya_quantum_v24.db"
MODEL_PATH = ROOT / "models" / "sgd_model_v24.joblib"
SCALER_PATH = ROOT / "models" / "scaler_v24.joblib"
LOG_PATH = ROOT / "logs" / "utaya_v24.log"

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
        test_file = ROOT / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
    except Exception as e:
        st.error(f"🚨 【Eドライブアクセスエラー】ストレージパスに書き込めません: {ROOT} / {e}")
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

        CREATE TABLE IF NOT EXISTS market_ticks(
            tick_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            ts TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            source TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ticks_lookup ON market_ticks(symbol, ts);

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
            model_version TEXT DEFAULT 'v24.0'
        );
        CREATE INDEX IF NOT EXISTS idx_pred_due ON predictions(symbol, status, due);

        CREATE TABLE IF NOT EXISTS training_audit(
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            pred_id TEXT UNIQUE,
            trained_at TEXT,
            label INTEGER,
            score_loss REAL,
            version TEXT
        );
        """)

# --- 1. データ取得・検証ロジック ---
def save_xtrend_quote(symbol: str, price: float, ts: datetime):
    if not np.isfinite(price) or price <= 0:
        raise ValueError("価格は0より大きい有限数である必要があります。")
    with connect_db() as c:
        c.execute(
            "INSERT INTO xtrend_quotes VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), symbol, stamp_utc(ts), float(price), "XTrend Manual/CSV")
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
        d = d[~d.index.duplicated(keep="last")].sort_index()
        
        # 異常値・欠損チェック
        if (d['Close'] <= 0).any() or d['Close'].isna().any():
            return pd.DataFrame()
        return d
    except Exception as e:
        logging.error(f"External market data fetch error: {e}")
        return pd.DataFrame()

# --- 2. 特徴量エンジニアリング ＆ 相場状態分類 ---
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
        return "NEUTRAL", "特徴量データ蓄積中（30分足未満）"
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

# --- 3. オンライン学習エンジンの修正（固定StandardScaler & 永続化） ---
def load_persisted_model():
    if not SKLEARN_OK or not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception:
        return None, None

def train_pending_predictions():
    if not SKLEARN_OK:
        return 0, "scikit-learn未導入"
    
    model, scaler = load_persisted_model()
    is_new = (model is None or scaler is None)
    
    with connect_db() as c:
        rows = c.execute("""
            SELECT pred_id, direction, features, outcome 
            FROM predictions 
            WHERE status='RESOLVED' AND outcome IN ('WIN', 'LOSS') AND trained=0
            ORDER BY settle_ts
        """).fetchall()
        
    if not rows:
        return 0, "未学習の決着データなし"
        
    if is_new:
        scaler = StandardScaler()
        # 初期フィット用のデータを全抽出
        all_res = pd.read_sql_query("SELECT features FROM predictions WHERE status='RESOLVED' AND outcome IN ('WIN','LOSS')", connect_db())
        if len(all_res) < 10:
            return 0, "学習サンプル不足（最低10件必要）"
        X_init = np.array([list(json.loads(r["features"]).values()) for _, r in all_res.iterrows()], dtype=float)
        scaler.fit(X_init)
        
        model = SGDClassifier(loss="log_loss", alpha=0.0005, random_state=42, average=True)
        model.partial_fit(scaler.transform(X_init), np.array([1 if r["outcome"]=="WIN" else 0 for _, r in all_res.iterrows()]), classes=np.array([0, 1]))
        trained_count = len(all_res)
        
        # 既存全件を学習済みマーク
        with connect_db() as c:
            c.execute("UPDATE predictions SET trained=1 WHERE status='RESOLVED' AND outcome IN ('WIN','LOSS')")
            c.commit()
    else:
        trained_count = 0
        for r in rows:
            try:
                feat_dict = json.loads(r["features"])
                X = np.array([[float(feat_dict[k]) for k in FEATURES]], dtype=float)
                if not np.isfinite(X).all():
                    continue
                # 勝敗ラベル: WIN=1, LOSS=0
                y = np.array([1 if r["outcome"] == "WIN" else 0])
                
                X_scaled = scaler.transform(X)
                model.partial_fit(X_scaled, y)
                
                with connect_db() as c:
                    c.execute("UPDATE predictions SET trained=1 WHERE pred_id=?", (r["pred_id"],))
                    c.execute("INSERT OR IGNORE INTO training_audit(pred_id, trained_at, label, version) VALUES (?, ?, ?, ?)",
                              (r["pred_id"], stamp_utc(now_utc()), int(y[0]), VERSION))
                    c.commit()
                trained_count += 1
            except Exception as e:
                logging.error(f"Training error for pred {r['pred_id']}: {e}")
                
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return trained_count, f"正常に {trained_count} 件をオンライン学習しました。"

def predict_probability(feat: dict) -> tuple[float | None, str]:
    model, scaler = load_persisted_model()
    if model is None or scaler is None:
        return None, "モデル未初期化または学習データ不足"
    try:
        X = np.array([[float(feat[k]) for k in FEATURES]], dtype=float)
        if not np.isfinite(X).all():
            return None, "特徴量異常値"
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[0]
        classes = list(model.classes_)
        p_up = float(probs[classes.index(1)]) if 1 in classes else 0.5
        return p_up, "モデル推論正常"
    except Exception as e:
        return None, f"推論エラー: {e}"

# --- 4. 3分後の決着処理（未来リーク防止・厳格UTC判定） ---
def settle_pending_predictions(symbol: str, raw_df: pd.DataFrame) -> int:
    if raw_df.empty:
        return 0
    now_t = now_utc()
    with connect_db() as c:
        rows = c.execute(
            "SELECT * FROM predictions WHERE symbol=? AND status='PENDING' AND due<=?",
            (symbol, stamp_utc(now_t))
        ).fetchall()
        
    settled_count = 0
    for r in rows:
        due_dt = pd.Timestamp(r["due"])
        # 対象時刻以降の最初の有効価格を正確に取得（過去データ使用禁止）
        after_ticks = raw_df[raw_df.index >= due_dt]
        if after_ticks.empty:
            continue  # まだ有効な価格データがないためPENDING維持
            
        settle_ts = after_ticks.index[0]
        settle_price = float(after_ticks.iloc[0]["Close"])
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

# ヘッダー
st.markdown(f"""
<div style="background:linear-gradient(135deg, #09090b, #1e1b4b);padding:22px;border-radius:14px;border:2px solid #38bdf8;">
    <h2 style="color:#38bdf8;margin:0">⚡ UTAYA AI QUANTUM — 完全自立型タイムトラベラー・リサーチコア ({VERSION})</h2>
    <p style="color:#cbd5e1;margin:6px 0 0 0">XTrend実測ギャップ管理 ｜ 3分先高精度未来予測 ｜ 固定StandardScaler・オンライン自己学習</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ システム・アセット設定")
    symbol = st.selectbox("対象アセット", list(ASSETS.keys()))
    ticker = ASSETS[symbol]
    
    st.markdown("---")
    st.markdown("### 📥 XTrend 実価格登録 (手動 / CSV)")
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
    model_chk, scaler_chk = load_persisted_model()
    n_trained = model_chk.n_iter_ if (model_chk and hasattr(model_chk, 'n_iter_')) else 0
    with connect_db() as c:
        total_learn_cnt = c.execute("SELECT COUNT(*) FROM training_audit").fetchone()[0]
    st.metric("🧠 オンライン学習済レコード数", f"{total_learn_cnt} 件")
    st.caption("※十分な決着データ蓄積後にモデルが自律推論します。")

# 外部市場データの取得 ＆ 決済・学習の自動実行
raw_df = fetch_external_market_data(ticker)

col_l, col_r = st.columns([1.1, 0.9])

with col_l:
    st.markdown("### 👁️ 千里眼・多角性宇宙情報モニター (グローバル基準)")
    if raw_df.empty:
        st.error("🚨 外部市場データの取得に失敗しました。ネットワーク接続を確認してください。")
    else:
        # バックグラウンド自動処理
        settle_pending_predictions(symbol, raw_df)
        train_pending_predictions()
        
        f = compute_features(raw_df)
        state, reason = classify_market_state(f)
        current_global_price = float(raw_df.iloc[-1]["Close"])
        
        # XTrendギャップ計算
        xtrend_df = get_xtrend_quotes(symbol)
        latest_xtrend_price = float(xtrend_df.iloc[-1]["price"]) if not xtrend_df.empty else current_global_price
        price_gap = latest_xtrend_price - current_global_price
        
        m1, m2, m3 = st.columns(3)
        m1.metric("グローバル市場価格", f"{current_global_price:,.5f}")
        m2.metric("XTrend最新価格", f"{latest_xtrend_price:,.5f}", delta=f"Gap: {price_gap:+.5f}")
        m3.metric("相場状態 (State)", state)
        
        st.info(f"🔮 **宇宙情報解析**: {reason}")
        
        feat = {k: float(f.iloc[-1][k]) for k in FEATURES}
        p_up, note = probability_val = predict_probability(feat)
        
        if p_up is not None:
            st.progress(p_up, text=f"AI上昇確率予測: {p_up*100:.1f}% ({note})")
            
            # 厳格な自動売買判断ロジック（根拠不足時は必ずWAIT）
            auto_dir = "WAIT"
            if state in ["HIGH_CEILING_CONFIRMED", "DOWNTREND"] and p_up <= 0.45:
                auto_dir = "LOW"
            elif state in ["LOW_REBOUND_CONFIRMED", "UPTREND"] and p_up >= 0.55:
                auto_dir = "HIGH"
            else:
                auto_dir = "WAIT"
        else:
            st.warning(f"🤖 AIモデル状態: {note} ⇒ 安全のため自動判定は **[WAIT]** です。")
            auto_dir = "WAIT"
            p_up = 0.5

        # 自動3分先予測レコードのスポーン（同一時間帯の重複防止）
        with connect_db() as c:
            recent_pend = c.execute("""
                SELECT COUNT(*) FROM predictions 
                WHERE symbol=? AND status='PENDING' AND created >= ?
            """, (symbol, stamp_utc(now_utc() - timedelta(minutes=2)))).fetchone()[0]
            
        if recent_pend == 0 and len(f) >= 30 and auto_dir != "WAIT":
            pid = str(uuid.uuid4())
            t_create = now_utc()
            due_time = t_create + timedelta(minutes=3)
            with connect_db() as c:
                c.execute("""
                    INSERT INTO predictions(pred_id, symbol, created, due, entry, direction, state, p_up, features, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                """, (pid, symbol, stamp_utc(t_create), stamp_utc(due_time), latest_xtrend_price, auto_dir, state, p_up, json.dumps(feat)))
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
            st.write("条件が揃い次第、次の3分予測レコードを自動生成します（現在はWAIT維持）。")

with col_r:
    st.markdown("### 🏆 実績・パフォーマンス統計サマリー")
    with connect_db() as c:
        stats = pd.read_sql_query("SELECT direction, outcome FROM predictions WHERE symbol=? AND status='RESOLVED'", c, params=(symbol,))
        hist = pd.read_sql_query("SELECT created, direction, entry, settle, outcome FROM predictions WHERE symbol=? ORDER BY created DESC LIMIT 10", c, params=(symbol,))
        
        total_pred = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=?", (symbol,)).fetchone()[0]
        resolved_cnt = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND status='RESOLVED'", (symbol,)).fetchone()[0]
        pending_cnt = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND status='PENDING'", (symbol,)).fetchone()[0]
        wait_cnt = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND direction='WAIT'", (symbol,)).fetchone()[0]
        high_cnt = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND direction='HIGH' AND status='RESOLVED'", (symbol,)).fetchone()[0]
        high_wins = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND direction='HIGH' AND outcome='WIN'", (symbol,)).fetchone()[0]
        low_cnt = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND direction='LOW' AND status='RESOLVED'", (symbol,)).fetchone()[0]
        low_wins = c.execute("SELECT COUNT(*) FROM predictions WHERE symbol=? AND direction='LOW' AND outcome='WIN'", (symbol,)).fetchone()[0]

    st.markdown(f"""
    - **総予測件数**: {total_pred} 件
    - **決着件数**: {resolved_cnt} 件 （未決着 PENDING: {pending_cnt} 件）
    - **見送り (WAIT) 件数**: {wait_cnt} 件
    - **HIGH戦績**: {high_wins}勝 / {high_cnt}件 (勝率: {(high_wins/high_cnt*100) if high_cnt>0 else 0.0:.1f}%)
    - **LOW戦績**: {low_wins}勝 / {low_cnt}件 (勝率: {(low_wins/low_cnt*100) if low_cnt>0 else 0.0:.1f}%)
    """)

    if not stats.empty:
        trades = stats[stats["direction"].isin(["HIGH", "LOW"]) & stats["outcome"].isin(["WIN", "LOSS"])]
        tot = len(trades)
        wins = int((trades["outcome"] == "WIN").sum())
        rate = (wins / tot * 100.0) if tot > 0 else 0.0
        if tot < 20:
            st.warning(f"⚠️ サンプル数不足（現在 {tot} 件）。統計的に不安定なため勝率は参考値です。")
        else:
            st.success(f"📊 総合実績勝率: {rate:.1f}% ({wins}勝 / {tot}件)")

    st.markdown("#### 📋 過去の決着済み予測履歴")
    if not hist.empty:
        st.dataframe(hist, use_container_width=True, hide_index=True)
    else:
        st.write("決着済みの予測履歴はまだありません。")

st.markdown("---")
st.caption(f"UTAYA AI QUANTUM {VERSION} | E-Drive SQLite & Persistent StandardScaler Active")
if st.checkbox("🔄 15秒ごとに画面を自動同期する", value=True):
    time.sleep(15)
    st.rerun()
