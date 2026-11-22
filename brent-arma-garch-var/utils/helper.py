"""
Funzioni di supporto per l'analisi ARMA, GARCH e VAR dinamico sul petrolio Brent.

Ogni funzione replica un passaggio del notebook `notebooks/Progetto_1_Brent.ipynb`,
così che l'intera analisi possa essere eseguita anche come script (`main.py`).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss


# ---------------------------------------------------------------------------
# Dati
# ---------------------------------------------------------------------------

def scarica_prezzi(ticker="BZ=F", start="2010-01-01", percorso_csv=None):
    """Scarica i prezzi di chiusura da Yahoo Finance e, se richiesto, li salva in CSV."""
    dati = yf.download(ticker, start=start, progress=False)
    prezzi = dati["Close"].squeeze().dropna()   # da DataFrame a una colonna a Series
    prezzi.name = ticker

    if percorso_csv is not None:
        Path(percorso_csv).parent.mkdir(parents=True, exist_ok=True)
        prezzi.to_csv(percorso_csv)

    return prezzi


def carica_prezzi(percorso_csv):
    """Carica i prezzi salvati in precedenza con `scarica_prezzi`."""
    prezzi = pd.read_csv(percorso_csv, index_col=0, parse_dates=True).squeeze("columns")
    return prezzi.dropna()


def rendimenti_log(prezzi):
    """Rendimenti logaritmici giornalieri: ln(P_t / P_t-1)."""
    return np.log(prezzi / prezzi.shift(1)).dropna()


# ---------------------------------------------------------------------------
# 1. Stazionarietà
# ---------------------------------------------------------------------------

def test_stazionarieta(serie):
    """
    ADF  -> H0: radice unitaria (serie NON stazionaria)
    KPSS -> H0: serie stazionaria
    Il p-value del KPSS è troncato tra 0.01 e 0.10.
    """
    adf_p = adfuller(serie)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # warning di interpolazione del KPSS
        kpss_p = kpss(serie, regression="c", nlags="auto")[1]
    return {"adf_pvalue": adf_p, "kpss_pvalue": kpss_p}


# ---------------------------------------------------------------------------
# 2. Autocorrelazione ed eteroschedasticità
# ---------------------------------------------------------------------------

def ljung_box(serie, lags=10):
    """Test di Ljung-Box. H0: nessuna autocorrelazione fino al lag k."""
    return acorr_ljungbox(serie, lags=lags)


def test_arch(serie, nlags=10):
    """Test ARCH di Engle. H0: varianza costante. Restituisce il p-value."""
    return het_arch(serie, nlags=nlags)[1]


# ---------------------------------------------------------------------------
# 3. ARMA
# ---------------------------------------------------------------------------

def griglia_arma(rendimenti, max_p=3, max_q=3):
    """Stima tutti gli ARMA(p, q) con p <= max_p e q <= max_q e restituisce AIC e BIC."""
    risultati = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                modello = ARIMA(rendimenti, order=(p, 0, q)).fit()
            risultati.append({"p": p, "q": q, "aic": modello.aic, "bic": modello.bic})
    return pd.DataFrame(risultati)


def ordine_migliore(griglia, criterio="aic"):
    """Restituisce l'ordine (p, 0, q) che minimizza il criterio scelto ('aic' o 'bic')."""
    riga = griglia.loc[griglia[criterio].idxmin()]
    return (int(riga["p"]), 0, int(riga["q"]))


def stima_arma(rendimenti, ordine):
    """Stima un ARMA con l'ordine indicato."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(rendimenti, order=ordine).fit()


# ---------------------------------------------------------------------------
# 4. GARCH(1,1)
# ---------------------------------------------------------------------------

def stima_garch(rendimenti, dist="normal"):
    """GARCH(1,1) a media costante sui rendimenti in percentuale (x100)."""
    return arch_model(rendimenti * 100, vol="Garch", p=1, q=1, dist=dist).fit(disp="off")


def statistiche_garch(modello):
    """Persistenza, volatilità di lungo periodo e half-life di uno shock."""
    omega = modello.params["omega"]
    alpha = modello.params["alpha[1]"]
    beta = modello.params["beta[1]"]

    persistenza = alpha + beta
    var_lungo_periodo = omega / (1 - persistenza)            # in %^2

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistenza": persistenza,
        "vol_giornaliera_lungo_periodo_%": np.sqrt(var_lungo_periodo),
        "vol_annua_lungo_periodo_%": np.sqrt(var_lungo_periodo) * np.sqrt(252),
        "half_life_giorni": np.log(0.5) / np.log(persistenza),
    }


def residui_standardizzati(modello):
    """z_t = epsilon_t / sigma_t: se il GARCH è adeguato sono rumore bianco."""
    return modello.resid / modello.conditional_volatility


# ---------------------------------------------------------------------------
# 5. VAR al 99%: statico, GARCH, EWMA
# ---------------------------------------------------------------------------

def calcola_var(rendimenti, modello_garch, livello=0.99, lam=0.94):
    """
    VAR_t = mu + z * sigma_t, con z quantile della normale.

    - statico: sigma costante (deviazione standard di tutto il campione)
    - GARCH:   sigma_t = volatilità condizionata (già calcolata con dati fino a t-1)
    - EWMA:    sigma_t = media mobile esponenziale, traslata di un giorno
               per usare solo informazioni disponibili fino a ieri
    """
    z = norm.ppf(1 - livello)                                 # -2.326 al 99%
    mu = rendimenti.mean()
    sigma = rendimenti.std()

    vol_garch = modello_garch.conditional_volatility / 100    # da % a decimali
    vol_ewma = rendimenti.ewm(alpha=1 - lam).std().shift(1)

    return pd.DataFrame({
        "rendimenti": rendimenti,
        "var_statico": mu + z * sigma,
        "var_garch": mu + z * vol_garch,
        "var_ewma": mu + z * vol_ewma,
    }).dropna()


def conta_violazioni(df_var, livello=0.99):
    """Numero e percentuale di giorni in cui il rendimento è inferiore al VAR."""
    n = len(df_var)
    modelli = {"Statico": "var_statico", "GARCH": "var_garch", "EWMA": "var_ewma"}

    righe = []
    for nome, colonna in modelli.items():
        violazioni = int((df_var["rendimenti"] < df_var[colonna]).sum())
        righe.append({"Modello": nome, "Violazioni": violazioni,
                      "Percentuale": round(violazioni / n * 100, 2)})

    tabella = pd.DataFrame(righe).set_index("Modello")
    attese = (1 - livello) * n
    return tabella, attese, n


def violazioni_per_anno(df_var, livello=0.99):
    """Violazioni per anno solare dei tre VAR, con le violazioni attese."""
    viol = pd.DataFrame({
        "Statico": df_var["rendimenti"] < df_var["var_statico"],
        "GARCH": df_var["rendimenti"] < df_var["var_garch"],
        "EWMA": df_var["rendimenti"] < df_var["var_ewma"],
    })
    per_anno = viol.groupby(viol.index.year).sum()
    per_anno["Attese"] = (viol.groupby(viol.index.year).size() * (1 - livello)).round(1)
    per_anno.index.name = "Anno"
    return per_anno
