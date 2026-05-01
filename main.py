
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np, yfinance as yf, time
from scipy.stats import t as student_t, jarque_bera, skew, kurtosis
from scipy.optimize import minimize
from scipy.special import gammaln
from datetime import datetime, date, timedelta

app = FastAPI(title="V-Lab DBK.DE")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

CACHE = {}
CACHE_TTL = 3600

def download_data(ticker, start="2021-01-01"):
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Pas de donnees pour {ticker}")
    close = df["Close"]
    if hasattr(close, "squeeze"):
        close = close.squeeze()
    close = close.dropna()
    dates = [str(d.date()) for d in close.index]
    px    = [round(float(v), 2) for v in close.values]
    rets  = []
    for i in range(1, len(close)):
        r = float(np.log(close.iloc[i] / close.iloc[i-1]) * 100)
        rets.append(round(r, 6))
    return dates, px, np.array(rets)

def garch_ll(params, R):
    w, a, b, nu = float(params[0]),float(params[1]),float(params[2]),float(params[3])
    if w<=0 or a<=0 or b<=0 or nu<=2 or a+b>=1: return 1e10
    mu = float(R.mean())
    eps = R - mu
    s2 = w/(1-a-b)
    lc = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log(np.pi*(nu-2))
    ll = 0.0
    for e in eps:
        e = float(e)
        s2 = max(w + a*e**2 + b*s2, 1e-8)
        ll += lc - 0.5*np.log(s2) - ((nu+1)/2)*np.log(1 + e**2/(s2*(nu-2)))
    return -ll

def fit_garch(R):
    res = minimize(garch_ll, [0.1,0.05,0.93,5.0], args=(R,),
                   method="L-BFGS-B",
                   bounds=[(1e-6,2),(1e-6,0.5),(1e-6,0.9999),(2.01,30)],
                   options={"maxiter":1000,"ftol":1e-10})
    w,a,b,nu = [float(x) for x in res.x]
    ll = float(-res.fun)
    n,k = len(R),4
    return dict(omega=w, alpha=a, beta=b, nu=nu,
                mu=float(R.mean()), loglik=ll,
                aic=float(2*k-2*ll), bic=float(k*np.log(n)-2*ll))

def compute_series(R, p):
    w,a,b,nu,mu = p["omega"],p["alpha"],p["beta"],p["nu"],p["mu"]
    q01 = float(student_t.ppf(0.01, df=nu))
    s2 = w/(1-a-b)
    vols, vars_ = [], []
    for e in (R - mu):
        e = float(e)
        s2 = max(w + a*e**2 + b*s2, 1e-6)
        s  = float(np.sqrt(s2))
        vols.append(round(s, 6))
        vars_.append(round(float(-(mu + s*q01)), 6))
    return vols, vars_, float(q01)

def forecast_ts(s0, p, horizons):
    w,a,b = p["omega"],p["alpha"],p["beta"]
    si = float(np.sqrt(w/(1-a-b)))
    result = []
    for h in horizons:
        if h == 0:
            result.append(round(float(s0), 6))
        else:
            v = float(np.sqrt(si**2 + (a+b)**h * (s0**2 - si**2)))
            result.append(round(v, 6))
    return result

def compute_nic(p, s0):
    w,a,b = p["omega"],p["alpha"],p["beta"]
    xs = np.linspace(-10, 10, 200)
    ys = [round(float(np.sqrt(w + a*float(r)**2 + b*float(s0)**2)), 6) for r in xs]
    return {"returns": [round(float(x), 3) for x in xs], "vols": ys}

@app.get("/")
def root(): return {"message": "V-Lab API OK"}

@app.get("/api/health")
def health(): return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/api/volatility/{ticker}")
def get_vol(ticker: str, start: str = Query("2021-01-01"), refit: bool = Query(False)):
    key = f"{ticker}_{start}"
    now = time.time()
    if not refit and key in CACHE:
        t0, data = CACHE[key]
        if now - t0 < CACHE_TTL:
            return JSONResponse(content=data)
    try:
        dates, prices, R = download_data(ticker, start)
        n   = len(R)
        p   = fit_garch(R)
        vols, vars_, q01 = compute_series(R, p)
        s0  = float(vols[-1])
        v0  = float(vars_[-1])
        per = float(p["alpha"] + p["beta"])
        si  = round(float(np.sqrt(p["omega"]/(1-per))), 4)
        hl  = round(float(np.log(0.5)/np.log(per)), 1)
        hz  = [0, 1, 5, 10, 22, 66, 132, 252]
        lb  = ["Auj.","1j","1s","10j","1m","3m","6m","1a"]
        fc  = forecast_ts(s0, p, hz)
        nic = compute_nic(p, s0)
        jbs, jbp = jarque_bera(R)
        vc  = {str(c): round(float(c)*abs(v0)/100, 2)
               for c in [10000,50000,100000,500000,1000000]}
        result = {
            "ticker":      ticker,
            "last_date":   dates[-1],
            "last_price":  float(prices[-1]),
            "n_obs":       int(n+1),
            "start_date":  dates[0],
            "params":      p,
            "q01":         round(q01, 4),
            "sig_inf":     si,
            "halflife":    hl,
            "persistence": round(per, 4),
            "sig_today":   round(s0, 4),
            "var_today":   round(v0, 4),
            "ann_vol":     round(float(R.std()*np.sqrt(252)), 2),
            "var_by_capital": vc,
            "forecast":    {"horizons": hz, "labels": lb, "vols": fc},
            "nic":         nic,
            "series": {
                "dates":   dates[1:],
                "prices":  [float(x) for x in prices[1:]],
                "returns": [float(x) for x in R],
                "vols":    [float(x) for x in vols],
                "vars":    [float(x) for x in vars_],
            },
            "stats": {
                "mean":     round(float(R.mean()), 6),
                "std":      round(float(R.std()), 6),
                "skewness": round(float(skew(R)), 4),
                "kurtosis": round(float(kurtosis(R)), 4),
                "jb_stat":  round(float(jbs), 1),
                "jb_p":     round(float(jbp), 6),
            },
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
        CACHE[key] = (now, result)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
