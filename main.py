
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import yfinance as yf
import time
import random
from scipy.stats import t as student_t, jarque_bera, skew, kurtosis
from scipy.optimize import minimize
from scipy.special import gammaln
from datetime import datetime, date, timedelta

app = FastAPI(title="V-Lab Volatility API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

CACHE = {}
CACHE_TTL = 7200  # 2 heures — réduit les appels Yahoo


def download_data(ticker: str, start: str = "2021-01-01", retries: int = 4):
    """Télécharge avec retry exponentiel si rate-limited."""
    last_err = None
    for attempt in range(retries):
        try:
            if attempt > 0:
                wait = (2 ** attempt) + random.uniform(1, 3)
                time.sleep(wait)
            
            df = yf.download(
                ticker,
                start=start,
                progress=False,
                auto_adjust=True,
                timeout=30,
            )
            if df is None or df.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Extraire Close de manière robuste
            if "Close" not in df.columns:
                raise ValueError("No Close column in data")
            
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.squeeze().dropna().astype(float)
            
            if len(close) < 100:
                raise ValueError(f"Not enough data: {len(close)} rows")
            
            dates  = [str(d.date()) for d in close.index]
            prices = [round(float(v), 2) for v in close.values]
            vals   = close.values.astype(float)
            rets   = np.log(vals[1:] / vals[:-1]) * 100.0
            rets   = np.array([round(float(r), 6) for r in rets])
            
            return dates, prices, rets
        
        except Exception as e:
            last_err = e
            if "Rate" in str(e) or "429" in str(e) or "Too Many" in str(e):
                continue
            raise
    
    raise RuntimeError(f"Failed after {retries} attempts: {last_err}")


def garch11_loglik(params, eps):
    omega, alpha, beta, nu = (float(p) for p in params)
    if omega <= 0 or alpha <= 0 or beta <= 0: return 1e12
    if alpha + beta >= 0.9999: return 1e12
    if nu <= 2.01: return 1e12
    
    log_c = (gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log(np.pi*(nu-2)))
    sig2  = omega / (1.0 - alpha - beta)
    ll    = 0.0
    
    for e in eps:
        sig2 = omega + alpha*e*e + beta*sig2
        if sig2 <= 1e-12: return 1e12
        ll += log_c - 0.5*np.log(sig2) - ((nu+1)/2)*np.log(1.0 + e*e/(sig2*(nu-2)))
    return -ll


def fit_garch(returns: np.ndarray) -> dict:
    mu  = float(returns.mean())
    eps = (returns - mu).astype(float)
    
    # Variance empirique pour initialiser omega
    var_emp = float(eps.var())
    
    best_ll, best_res = np.inf, None
    
    starts = [
        [var_emp * 0.05, 0.08,  0.90, 6.0],
        [var_emp * 0.05, 0.05,  0.93, 5.0],
        [var_emp * 0.10, 0.10,  0.85, 4.0],
        [var_emp * 0.03, 0.06,  0.92, 7.0],
        [var_emp * 0.08, 0.12,  0.82, 5.5],
        [var_emp * 0.02, 0.04,  0.95, 8.0],
    ]
    
    bounds = [
        (1e-8,  var_emp * 2),  # omega
        (1e-6,  0.25),         # alpha
        (0.60,  0.9997),       # beta  — force persistence >= 0.60
        (2.10,  20.0),         # nu
    ]
    
    for x0 in starts:
        try:
            res = minimize(
                garch11_loglik, x0, args=(eps,),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-9},
            )
            if res.fun < best_ll:
                best_ll  = res.fun
                best_res = res
        except Exception:
            continue
    
    if best_res is None:
        raise RuntimeError("GARCH optimization failed")
    
    omega, alpha, beta, nu = (float(x) for x in best_res.x)
    loglik = float(-best_res.fun)
    n, k   = len(eps), 5
    
    return {
        "omega":    round(omega, 6),
        "alpha":    round(alpha, 6),
        "beta":     round(beta,  6),
        "nu":       round(nu,    4),
        "mu":       round(mu,    6),
        "loglik":   round(loglik, 3),
        "aic":      round(2*k - 2*loglik, 3),
        "bic":      round(k*float(np.log(n)) - 2*loglik, 3),
        "converged": bool(best_res.success),
    }


def compute_series(returns, params):
    w, a, b = params["omega"], params["alpha"], params["beta"]
    nu, mu  = params["nu"], params["mu"]
    q01  = float(student_t.ppf(0.01, df=nu))
    sig2 = w / (1.0 - a - b)
    vols, vars_ = [], []
    for e in (returns - mu).astype(float):
        sig2 = max(w + a*e*e + b*sig2, 1e-12)
        s    = float(np.sqrt(sig2))
        vols.append(round(s, 6))
        vars_.append(round(float(-(mu + s*q01)), 6))
    return vols, vars_, float(q01)


def term_structure(s0, params, horizons):
    w, a, b = params["omega"], params["alpha"], params["beta"]
    si = float(np.sqrt(w/(1-a-b)))
    out = []
    for h in horizons:
        if h == 0:
            out.append(round(float(s0), 6))
        else:
            v = float(np.sqrt(si**2 + (a+b)**h * (s0**2 - si**2)))
            out.append(round(v, 6))
    return out


def nic(params, s0):
    w, a, b = params["omega"], params["alpha"], params["beta"]
    xs = np.linspace(-10, 10, 200)
    ys = [round(float(np.sqrt(w + a*float(r)**2 + b*s0**2)), 6) for r in xs]
    return {"returns": [round(float(x), 3) for x in xs], "vols": ys}


@app.get("/")
def root(): return {"message": "V-Lab API OK", "docs": "/docs"}

@app.get("/api/health")
def health(): return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/volatility/{ticker}")
def get_vol(ticker: str,
            start:  str  = Query("2021-01-01"),
            refit:  bool = Query(False)):
    
    key = f"{ticker}|{start}"
    now = time.time()
    
    if not refit and key in CACHE:
        t0, data = CACHE[key]
        if now - t0 < CACHE_TTL:
            return JSONResponse(content=data)
    
    try:
        dates, prices, R = download_data(ticker, start)
        n = len(R)
        
        params = fit_garch(R)
        vols, vars_, q01 = compute_series(R, params)
        
        s0  = float(vols[-1])
        v0  = float(vars_[-1])
        per = round(float(params["alpha"] + params["beta"]), 4)
        si  = round(float(np.sqrt(params["omega"]/(1-per))), 4)
        hl  = round(float(np.log(0.5)/np.log(per)), 1) if per < 1 else 999
        
        hz  = [0, 1, 5, 10, 22, 66, 132, 252]
        lb  = ["Auj.","1j","1s","10j","1m","3m","6m","1a"]
        
        jbs, jbp = jarque_bera(R)
        
        vc = {str(c): round(float(c)*abs(v0)/100, 2)
              for c in [10000, 50000, 100000, 500000, 1000000]}
        
        # Alléger les séries
        cutoff = (date.today() - timedelta(days=730)).isoformat()
        idx2y  = next((i for i,d in enumerate(dates[1:]) if d >= cutoff), 0)
        keep   = sorted(set(list(range(0, max(idx2y,1), 3)) + list(range(idx2y, n))))
        
        def th(lst, off=0):
            return [float(lst[i+off]) for i in keep if i+off < len(lst)]
        
        result = {
            "ticker":      ticker,
            "last_date":   dates[-1],
            "last_price":  float(prices[-1]),
            "n_obs":       int(n+1),
            "start_date":  dates[0],
            "params":      params,
            "q01":         round(q01, 4),
            "sig_inf":     si,
            "halflife":    hl,
            "persistence": per,
            "sig_today":   round(s0, 4),
            "var_today":   round(v0, 4),
            "ann_vol":     round(float(R.std()*np.sqrt(252)), 2),
            "var_by_capital": vc,
            "forecast":    {"horizons": hz, "labels": lb, "vols": term_structure(s0, params, hz)},
            "nic":         nic(params, s0),
            "series": {
                "dates":   [dates[i+1] for i in keep if i+1 < len(dates)],
                "prices":  th(prices, 1),
                "returns": [float(R[i]) for i in keep if i < n],
                "vols":    th(vols),
                "vars":    th(vars_),
            },
            "stats": {
                "mean":     round(float(R.mean()), 6),
                "std":      round(float(R.std()), 6),
                "skewness": round(float(skew(R)), 4),
                "kurtosis": round(float(kurtosis(R)), 4),
                "jb_stat":  round(float(jbs), 1),
                "jb_p":     round(float(jbp), 6),
                "min":      round(float(R.min()), 4),
                "max":      round(float(R.max()), 4),
            },
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        
        CACHE[key] = (now, result)
        return JSONResponse(content=result)
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
