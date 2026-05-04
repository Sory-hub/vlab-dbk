
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import yfinance as yf
import time, random
from scipy.stats import t as student_t, jarque_bera, skew, kurtosis
from scipy.optimize import minimize
from scipy.special import gammaln
from datetime import datetime, date, timedelta

app = FastAPI(title="V-Lab Volatility API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

CACHE = {}
CACHE_TTL = 14400


def download_data(ticker, start="2021-01-01", retries=4):
    last_err = None
    for attempt in range(retries):
        try:
            if attempt > 0:
                time.sleep((2**attempt) + random.uniform(1, 3))
            df = yf.download(ticker, start=start, progress=False, auto_adjust=True, timeout=30)
            if df is None or df.empty:
                raise ValueError(f"No data for {ticker}")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.squeeze().dropna().astype(float)
            if len(close) < 100:
                raise ValueError(f"Not enough data: {len(close)}")
            dates  = [str(d.date()) for d in close.index]
            prices = [round(float(v), 2) for v in close.values]
            vals   = close.values.astype(float)
            rets   = np.log(vals[1:] / vals[:-1]) * 100.0
            return dates, prices, rets.astype(float)
        except Exception as e:
            last_err = e
            if "Rate" in str(e) or "429" in str(e) or "Too Many" in str(e):
                continue
            raise
    raise RuntimeError(f"Failed after {retries} retries: {last_err}")


# ── Filtre GARCH(1,1) ──────────────────────────────────────────────────
def garch_filter(omega, alpha, beta, eps):
    n = len(eps)
    sig2 = np.empty(n)
    sig2[0] = omega / (1.0 - alpha - beta)
    for t in range(1, n):
        sig2[t] = omega + alpha*eps[t-1]**2 + beta*sig2[t-1]
        if sig2[t] <= 0: sig2[t] = 1e-8
    return sig2


# ── Filtre GJR-GARCH(1,1) ─────────────────────────────────────────────
def gjr_filter(omega, alpha, beta, gamma, eps):
    """GJR-GARCH : sigma2_t = omega + alpha*eps2 + gamma*eps2*I(eps<0) + beta*sigma2"""
    n = len(eps)
    sig2 = np.empty(n)
    denom = 1.0 - alpha - beta - 0.5*gamma
    sig2[0] = omega / max(denom, 1e-8)
    for t in range(1, n):
        I = 1.0 if eps[t-1] < 0 else 0.0
        sig2[t] = omega + alpha*eps[t-1]**2 + gamma*eps[t-1]**2*I + beta*sig2[t-1]
        if sig2[t] <= 0: sig2[t] = 1e-8
    return sig2


# ── Log-vraisemblances ────────────────────────────────────────────────
def ll_normal(omega, alpha, beta, eps):
    sig2 = garch_filter(omega, alpha, beta, eps)
    return -0.5 * np.sum(np.log(2*np.pi*sig2) + eps**2/sig2)

def ll_student(omega, alpha, beta, nu, eps):
    sig2  = garch_filter(omega, alpha, beta, eps)
    log_c = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log(np.pi*(nu-2))
    return np.sum(log_c - 0.5*np.log(sig2) - ((nu+1)/2)*np.log(1+eps**2/(sig2*(nu-2))))

def ll_gjr_student(omega, alpha, beta, gamma, nu, eps):
    sig2  = gjr_filter(omega, alpha, beta, gamma, eps)
    log_c = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log(np.pi*(nu-2))
    return np.sum(log_c - 0.5*np.log(sig2) - ((nu+1)/2)*np.log(1+eps**2/(sig2*(nu-2))))


# ── Estimation GARCH(1,1) Student (Nelder-Mead 2 etapes) ─────────────
def fit_garch(returns):
    mu  = float(returns.mean())
    eps = (returns - mu).astype(float)
    var = float(eps.var())
    
    # Etape 1 : Normale
    def neg_ll_n(p):
        w,a,b = p
        if w<=0 or a<=0 or b<=0 or a+b>=1: return 1e12
        return -ll_normal(w, a, b, eps)
    best_n = None
    for x0 in [[var*0.05,0.08,0.90],[var*0.03,0.05,0.93]]:
        try:
            r = minimize(neg_ll_n, x0, method="Nelder-Mead",
                         options={"maxiter":5000,"xatol":1e-8,"fatol":1e-8})
            if best_n is None or r.fun < best_n.fun: best_n = r
        except: pass
    w0,a0,b0 = best_n.x if best_n else [var*0.05,0.08,0.90]
    
    # Etape 2 : Student
    def neg_ll_s(p):
        w,a,b,nu = p
        if w<=0 or a<=0 or b<=0 or nu<=2 or a+b>=1: return 1e12
        return -ll_student(w, a, b, nu, eps)
    best_s, best_ll = None, np.inf
    for nu0 in [4.0, 5.0, 6.0, 7.0]:
        for x0 in [[w0,a0,b0,nu0],[w0*0.8,a0*0.9,b0*1.01,nu0]]:
            try:
                r = minimize(neg_ll_s, x0, method="Nelder-Mead",
                             options={"maxiter":10000,"xatol":1e-8,"fatol":1e-8,"adaptive":True})
                w,a,b,nu = r.x
                if w>0 and a>0 and b>0 and nu>2 and a+b<1 and r.fun < best_ll:
                    best_ll = r.fun; best_s = r
            except: pass
    
    if best_s is None: raise RuntimeError("GARCH optimization failed")
    w,a,b,nu = (float(x) for x in best_s.x)
    ll = float(-best_s.fun); n,k = len(eps),5
    return dict(omega=round(w,6),alpha=round(a,6),beta=round(b,6),nu=round(nu,4),
                mu=round(mu,6),loglik=round(ll,3),
                aic=round(2*k-2*ll,3),bic=round(k*float(np.log(n))-2*ll,3),
                converged=bool(best_s.success))


# ── Estimation GJR-GARCH(1,1) Student ────────────────────────────────
def fit_gjr_garch(returns, garch_params):
    """Estime GJR-GARCH en partant des parametres GARCH comme initialisation."""
    mu  = float(returns.mean())
    eps = (returns - mu).astype(float)
    w0  = garch_params["omega"]
    a0  = garch_params["alpha"]
    b0  = garch_params["beta"]
    nu0 = garch_params["nu"]
    
    def neg_ll_gjr(p):
        w,a,b,g,nu = p
        if w<=0 or a<=0 or b<=0 or g<0 or nu<=2: return 1e12
        if a+b+0.5*g >= 1.0: return 1e12
        return -ll_gjr_student(w, a, b, g, nu, eps)
    
    best_s, best_ll = None, np.inf
    # gamma initialise a une fraction de alpha (effet levier modere)
    for g0 in [a0*0.5, a0*0.8, a0*1.0, a0*1.5]:
        for x0 in [
            [w0, a0*0.6, b0, g0, nu0],
            [w0, a0*0.5, b0, g0*1.2, nu0],
            [w0, a0*0.7, b0, g0*0.8, nu0],
        ]:
            try:
                r = minimize(neg_ll_gjr, x0, method="Nelder-Mead",
                             options={"maxiter":15000,"xatol":1e-9,"fatol":1e-9,"adaptive":True})
                w,a,b,g,nu = r.x
                if w>0 and a>0 and b>0 and g>=0 and nu>2 and a+b+0.5*g<1 and r.fun<best_ll:
                    best_ll = r.fun; best_s = r
            except: pass
    
    if best_s is None: return None
    w,a,b,g,nu = (float(x) for x in best_s.x)
    ll = float(-best_s.fun); n,k = len(eps),6
    return dict(omega=round(w,6),alpha=round(a,6),beta=round(b,6),gamma=round(g,6),
                nu=round(nu,4),mu=round(mu,6),loglik=round(ll,3),
                aic=round(2*k-2*ll,3),bic=round(k*float(np.log(n))-2*ll,3))


# ── Series conditionnelles ────────────────────────────────────────────
def compute_series(returns, params):
    w,a,b,nu,mu = params["omega"],params["alpha"],params["beta"],params["nu"],params["mu"]
    q01  = float(student_t.ppf(0.01, df=nu))
    sig2 = garch_filter(w, a, b, (returns-mu).astype(float))
    vols  = [round(float(np.sqrt(s)),6) for s in sig2]
    vars_ = [round(float(-(mu+np.sqrt(s)*q01)),6) for s in sig2]
    return vols, vars_, float(q01)


# ── Term structure ────────────────────────────────────────────────────
def term_structure(s0, params, horizons):
    w,a,b = params["omega"],params["alpha"],params["beta"]
    si = float(np.sqrt(w/(1-a-b)))
    return [round(float(s0) if h==0
                  else float(np.sqrt(si**2+(a+b)**h*(s0**2-si**2))),6)
            for h in horizons]


# ── NIC GARCH(1,1) — symetrique ───────────────────────────────────────
def nic_garch(params, s0):
    w,a,b = params["omega"],params["alpha"],params["beta"]
    xs = np.linspace(-10, 10, 200)
    ys = [round(float(np.sqrt(w+a*float(r)**2+b*s0**2)),6) for r in xs]
    return {"returns":[round(float(x),3) for x in xs],"vols":ys,"type":"symmetric"}


# ── NIC GJR-GARCH — asymetrique ───────────────────────────────────────
def nic_gjr(gjr_params, s0):
    """NIC asymetrique : les chocs negatifs font plus de volatilite."""
    w = gjr_params["omega"]
    a = gjr_params["alpha"]
    b = gjr_params["beta"]
    g = gjr_params["gamma"]
    sig2_t = s0**2
    xs = np.linspace(-10, 10, 200)
    ys = []
    for r in xs:
        I = 1.0 if r < 0 else 0.0
        v = w + a*r**2 + g*r**2*I + b*sig2_t
        ys.append(round(float(np.sqrt(max(v,0))),6))
    return {"returns":[round(float(x),3) for x in xs],"vols":ys,"type":"asymmetric",
            "gamma":gjr_params["gamma"]}


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"message":"V-Lab API OK"}

@app.get("/api/health")
def health(): return {"status":"ok","time":datetime.utcnow().isoformat()}


@app.get("/api/volatility/{ticker}")
def get_vol(ticker:str, start:str=Query("2021-01-01"), refit:bool=Query(False)):
    key = f"{ticker}|{start}"; now = time.time()
    if not refit and key in CACHE:
        t0,data = CACHE[key]
        if now-t0 < CACHE_TTL: return JSONResponse(content=data)
    try:
        dates, prices, R = download_data(ticker, start)
        n = len(R)
        
        # 1. GARCH(1,1) Student
        params = fit_garch(R)
        vols, vars_, q01 = compute_series(R, params)
        s0  = float(vols[-1]); v0 = float(vars_[-1])
        per = round(float(params["alpha"]+params["beta"]),4)
        si  = round(float(np.sqrt(params["omega"]/(1-per))),4)
        hl  = round(float(np.log(0.5)/np.log(per)),1) if 0<per<1 else 999
        
        # 2. GJR-GARCH Student (pour la NIC asymetrique)
        gjr_params = None
        try: gjr_params = fit_gjr_garch(R, params)
        except: pass
        
        hz = [0,1,5,10,22,66,132,252]
        lb = ["Auj.","1j","1s","10j","1m","3m","6m","1a"]
        jbs,jbp = jarque_bera(R)
        vc = {str(c):round(float(c)*abs(v0)/100,2)
              for c in [10000,50000,100000,500000,1000000]}
        
        # Alleger les series
        cutoff = (date.today()-timedelta(days=730)).isoformat()
        idx2y  = next((i for i,d in enumerate(dates[1:]) if d>=cutoff),0)
        keep   = sorted(set(list(range(0,max(idx2y,1),3))+list(range(idx2y,n))))
        def th(lst,off=0): return [float(lst[i+off]) for i in keep if i+off<len(lst)]
        
        result = {
            "ticker":ticker,"last_date":dates[-1],"last_price":float(prices[-1]),
            "n_obs":int(n+1),"start_date":dates[0],
            "params":params,"q01":round(q01,4),
            "sig_inf":si,"halflife":hl,"persistence":per,
            "sig_today":round(s0,4),"var_today":round(v0,4),
            "ann_vol":round(float(R.std()*np.sqrt(252)),2),
            "var_by_capital":vc,
            "forecast":{"horizons":hz,"labels":lb,"vols":term_structure(s0,params,hz)},
            # NIC GARCH symetrique
            "nic": nic_garch(params, s0),
            # NIC GJR asymetrique (si disponible)
            "nic_gjr": nic_gjr(gjr_params, s0) if gjr_params else None,
            "gjr_params": gjr_params,
            "series":{
                "dates":[dates[i+1] for i in keep if i+1<len(dates)],
                "prices":th(prices,1),
                "returns":[float(R[i]) for i in keep if i<n],
                "vols":th(vols),"vars":th(vars_),
            },
            "stats":{
                "mean":round(float(R.mean()),6),"std":round(float(R.std()),6),
                "skewness":round(float(skew(R)),4),"kurtosis":round(float(kurtosis(R)),4),
                "jb_stat":round(float(jbs),1),"jb_p":round(float(jbp),6),
                "min":round(float(R.min()),4),"max":round(float(R.max()),4),
            },
            "updated_at":datetime.utcnow().isoformat()+"Z",
        }
        CACHE[key] = (now,result)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500,detail=str(exc))
