import os, json, time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TD_KEY = os.environ.get("TWELVEDATA_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# inv = futures price is quoted inverted (foreign currency in USD)
# dp  = decimals to show, tol = allowed % difference with backup price
ASSETS = {
    "EURUSD":   {"sym": "6E=F",    "shift": 6, "td": "EUR/USD", "inv": False, "dp": 5, "tol": 0.5},
    "GBPUSD":   {"sym": "6B=F",    "shift": 6, "td": "GBP/USD", "inv": False, "dp": 5, "tol": 0.5},
    "AUDUSD":   {"sym": "6A=F",    "shift": 6, "td": "AUD/USD", "inv": False, "dp": 5, "tol": 0.5},
    "NZDUSD":   {"sym": "6N=F",    "shift": 6, "td": "NZD/USD", "inv": False, "dp": 5, "tol": 0.5},
    "JPY(6J)":  {"sym": "6J=F",    "shift": 6, "td": "USD/JPY", "inv": True,  "dp": 7, "tol": 0.5},
    "CAD(6C)":  {"sym": "6C=F",    "shift": 6, "td": "USD/CAD", "inv": True,  "dp": 5, "tol": 0.5},
    "CHF(6S)":  {"sym": "6S=F",    "shift": 6, "td": "USD/CHF", "inv": True,  "dp": 5, "tol": 0.5},
    "GOLD":     {"sym": "GC=F",    "shift": 6, "td": "XAU/USD", "inv": False, "dp": 2, "tol": 1.5},
    "BTC":      {"sym": "BTC-USD", "shift": 0, "td": "BTC/USD", "inv": False, "dp": 2, "tol": 1.0},
}
FILE = "trades.json"
MIN_RR = 1.5
MAX_AGE_MIN = 40
VOL_LOOKBACK = 20


def tg(msg):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  data={"chat_id": CHAT_ID, "text": msg}, timeout=20)


def load():
    try:
        with open(FILE) as f:
            return json.load(f)
    except Exception:
        return {"open": [], "closed": []}


# ---------- Data sources ----------
def yahoo(sym, tries=3):
    for _ in range(tries):
        try:
            df = yf.Ticker(sym).history(period="5d", interval="15m")
            if len(df) > 0:
                return df
        except Exception as e:
            print("Yahoo error:", e)
        time.sleep(3)
    return None


def twelve(symbol, n=100, inv=False):
    if not TD_KEY:
        return None
    try:
        r = requests.get("https://api.twelvedata.com/time_series",
                         params={"symbol": symbol, "interval": "15min",
                                 "outputsize": n, "timezone": "UTC",
                                 "apikey": TD_KEY}, timeout=20).json()
        vals = r.get("values")
        if not vals:
            return None
        df = pd.DataFrame(vals)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)
        df = df.rename(columns={"high": "High", "low": "Low", "close": "Close"})
        if inv:
            hi, lo = 1 / df["Low"], 1 / df["High"]
            df["High"], df["Low"], df["Close"] = hi, lo, 1 / df["Close"]
        return df
    except Exception as e:
        print("TwelveData error:", e)
        return None


def price_ok(cfg, price):
    bp = twelve(cfg["td"], 3, cfg["inv"])
    if bp is None:
        return True, "backup unavailable"
    ref = bp.Close.iloc[-1]
    diff = abs(ref - price) / ref * 100
    return diff < cfg["tol"], f"{diff:.2f}%"


# ---------- AI news filter ----------
def news_check(name, side):
    if not GEMINI_KEY:
        return False, "AI filter off"
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    prompt = (
        f"Now it is {now}. Use Google Search. Is there a high-impact scheduled "
        f"economic release or breaking news (for example US CPI, FOMC, NFP, "
        f"ECB, BoE, BoJ decision, central bank speech, major crypto news) "
        f"affecting the market '{name}' within the last 30 minutes or the "
        f"next 30 minutes? Reply with the first word BLOCK if yes, or OK if no. "
        f"Then give one short reason in English."
    )
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}]}
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            headers={"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"},
            json=body, timeout=60).json()
        parts = r["candidates"][0]["content"]["parts"]
        text = " ".join(p.get("text", "") for p in parts).strip()
        return text.upper().startswith("BLOCK"), text[:200]
    except Exception as e:
        print("Gemini error:", e)
        return False, "AI check failed"


# ---------- Strategy: Pivot + VWAP + Volume + POC ----------
def poc(df, bins=40):
    p = (df.High + df.Low + df.Close) / 3
    h, e = np.histogram(p, bins=bins, weights=df.Volume)
    i = h.argmax()
    return (e[i] + e[i + 1]) / 2


def analyse(cfg):
    df = yahoo(cfg["sym"])
    if df is None:
        return None
    df = df[df.Volume > 0].dropna().copy()
    if len(df) < 60:
        return None
    df = df.iloc[:-1]  # drop unfinished candle

    last_time = df.index[-1]
    age = pd.Timestamp.now(tz=last_time.tz) - last_time
    if age > pd.Timedelta(minutes=MAX_AGE_MIN):
        return None  # market closed or old data

    df["day"] = (df.index + pd.Timedelta(hours=cfg["shift"])).date
    days = sorted(df["day"].unique())
    if len(days) < 2:
        return None
    prev = df[df.day == days[-2]]
    today = df[df.day == days[-1]]
    if len(today) < 15 or len(prev) < 20:
        return None

    prev_poc = poc(prev)

    H, L, C = prev.High.max(), prev.Low.min(), prev.Close.iloc[-1]
    rng = H - L
    P = (H + L + C) / 3
    R1, S1 = 2 * P - L, 2 * P - H
    R2, S2 = P + rng, P - rng
    h3, h4 = C + rng * 1.1 / 4, C + rng * 1.1 / 2
    l3, l4 = C - rng * 1.1 / 4, C - rng * 1.1 / 2

    tp = (today.High + today.Low + today.Close) / 3
    vwap_s = (tp * today.Volume).cumsum() / today.Volume.cumsum()
    vwap = vwap_s.iloc[-1]

    vol_now = df.Volume.iloc[-1]
    vol_avg = df.Volume.iloc[-(VOL_LOOKBACK + 1):-1].mean()
    if vol_now <= vol_avg:
        return None

    close = today.Close.iloc[-1]
    prev_close = today.Close.iloc[-2]
    low, high = today.Low.iloc[-1], today.High.iloc[-1]

    trend_up = vwap > vwap_s.iloc[-7]
    trend_down = vwap < vwap_s.iloc[-7]
    swing_low = today.Low.tail(6).min()
    swing_high = today.High.tail(6).max()

    side, reason = None, ""
    if trend_up and close > vwap:
        if prev_close <= prev_poc < close:
            side, reason = "BUY", "POC breakout"
        elif low <= l3 < close:
            side, reason = "BUY", "L3 bounce"
    if side is None and trend_down and close < vwap:
        if prev_close >= prev_poc > close:
            side, reason = "SELL", "POC breakdown"
        elif high >= h3 > close:
            side, reason = "SELL", "H3 rejection"
    if side is None:
        return None

    if side == "BUY":
        sl = min(vwap, swing_low)
        risk = close - sl
        above = [x for x in (P, R1, R2, h3, h4) if x > close]
        target = min(above) if above else None
        reward = (target - close) if target else 0
    else:
        sl = max(vwap, swing_high)
        risk = sl - close
        below = [x for x in (P, S1, S2, l3, l4) if x < close]
        target = max(below) if below else None
        reward = (close - target) if target else 0

    if target is None or risk <= 0 or reward <= 0 or reward / risk < MIN_RR:
        return None

    return {"side": side, "reason": reason, "price": float(close),
            "sl": float(sl), "target": float(target),
            "rr": float(reward / risk), "vwap": float(vwap),
            "poc": float(prev_poc), "P": float(P), "R1": float(R1),
            "S1": float(S1), "h3": float(h3), "l3": float(l3),
            "vol_ratio": float(vol_now / vol_avg)}


def update_open(name, cfg, d):
    if not any(t["name"] == name for t in d["open"]):
        return
    df = yahoo(cfg["sym"])
    if df is None:
        df = twelve(cfg["td"], 200, cfg["inv"])
        if df is not None:
            print("Using backup data for", name)
    if df is None:
        return
    keep = []
    for t in d["open"]:
        if t["name"] != name:
            keep.append(t)
            continue
        since = df[df.index > pd.Timestamp(t["time"])]
        result = None
        for _, c in since.iterrows():
            if t["side"] == "BUY":
                if c.Low <= t["sl"]:
                    result = -1.0
                    break
                if c.High >= t["target"]:
                    result = t["rr"]
                    break
            else:
                if c.High >= t["sl"]:
                    result = -1.0
                    break
                if c.Low <= t["target"]:
                    result = t["rr"]
                    break
        if result is None:
            keep.append(t)
            continue
        t["r"] = round(result, 2)
        d["closed"].append(t)
        wins = sum(1 for x in d["closed"] if x["r"] > 0)
        total = len(d["closed"])
        total_r = sum(x["r"] for x in d["closed"])
        tg(f"{'✅ WIN' if result > 0 else '❌ LOSS'}: {name} {t['side']}\n"
           f"Result: {result:+.1f}R\n"
           f"Overall: {wins}/{total} ({100 * wins / total:.0f}%), total {total_r:+.1f}R")
    d["open"] = keep


d = load()
for name, cfg in ASSETS.items():
    try:
        update_open(name, cfg, d)
        if any(t["name"] == name for t in d["open"]):
            continue
        res = analyse(cfg)
        if not res:
            print("No signal:", name)
            continue

        ok, diff = price_ok(cfg, res["price"])
        if not ok:
            tg(f"⚠️ {name}: signal skipped, price mismatch with backup ({diff})")
            continue

        block, why = news_check(name, res["side"])
        if block:
            tg(f"⛔ {name} {res['side']} blocked by news filter:\n{why}")
            continue

        dp = cfg["dp"]
        f = lambda x: f"{x:.{dp}f}"
        note = ("\nNote: inverted futures. BUY = foreign currency "
                "strengthens vs USD (USD pair falls)." if cfg["inv"] else "")
        tg(f"{'🟢' if res['side'] == 'BUY' else '🔴'} {res['side']} (paper) {name} - {res['reason']}\n"
           f"Price: {f(res['price'])}\nStop-loss: {f(res['sl'])}\n"
           f"Target: {f(res['target'])}\nReward:Risk = {res['rr']:.1f}\n"
           f"VWAP: {f(res['vwap'])}\nPOC: {f(res['poc'])}\n"
           f"Volume: {res['vol_ratio']:.1f}x average\n"
           f"Pivot: {f(res['P'])} | R1: {f(res['R1'])} | S1: {f(res['S1'])}\n"
           f"H3: {f(res['h3'])} | L3: {f(res['l3'])}\n"
           f"Price check: {diff}\nNews check: {why}{note}")
        d["open"].append({"name": name, "side": res["side"],
                          "price": res["price"], "sl": res["sl"],
                          "target": res["target"], "rr": res["rr"],
                          "time": pd.Timestamp.now(tz="UTC").isoformat()})
    except Exception as e:
        print("Error", name, e)
    time.sleep(1)

with open(FILE, "w") as f:
    json.dump(d, f, indent=2)
