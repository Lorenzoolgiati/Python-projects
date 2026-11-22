"""
PROGETTO 1: ARMA, GARCH e VAR dinamico sul petrolio Brent.

Esegue l'intera analisi del notebook in un solo comando:

    python main.py                # scarica i dati aggiornati da Yahoo Finance
    python main.py --offline      # usa i dati già salvati in data/brent.csv

I risultati vengono stampati a schermo e salvati in reports/ (tabelle CSV e grafici PNG).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # salva i grafici su file senza aprire finestre
import matplotlib.pyplot as plt
import numpy as np

from utils import helper

CARTELLA = Path(__file__).resolve().parent
PERCORSO_DATI = CARTELLA / "data" / "brent.csv"
CARTELLA_REPORT = CARTELLA / "reports"
CARTELLA_FIGURE = CARTELLA_REPORT / "figures"

TICKER = "BZ=F"
INIZIO = "2010-01-01"
LIVELLO_VAR = 0.99


def titolo(testo):
    print("\n" + "=" * 70)
    print(testo)
    print("=" * 70)


def salva_figura(nome):
    plt.tight_layout()
    plt.savefig(CARTELLA_FIGURE / nome, dpi=120)
    plt.close()


def main(offline=False):
    CARTELLA_FIGURE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ dati
    if offline:
        prezzi = helper.carica_prezzi(PERCORSO_DATI)
    else:
        prezzi = helper.scarica_prezzi(TICKER, INIZIO, percorso_csv=PERCORSO_DATI)
    rendimenti = helper.rendimenti_log(prezzi)
    print(f"Dati: {prezzi.index[0].date()} -> {prezzi.index[-1].date()} | {len(rendimenti)} rendimenti")

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    axes[0].plot(prezzi.index, prezzi)
    axes[0].set_title("Prezzo del Brent (USD/barile)")
    axes[1].plot(rendimenti.index, rendimenti * 100)
    axes[1].set_title("Rendimenti logaritmici giornalieri (%)")
    axes[2].plot(rendimenti.index, rendimenti**2 * 100)
    axes[2].set_title("Rendimenti al quadrato - varianza")
    salva_figura("prezzi_rendimenti.png")

    # ------------------------------------------------------- 1. stazionarietà
    titolo("1. Stazionarietà (ADF: H0 non stazionaria | KPSS: H0 stazionaria)")
    for nome, serie in [("Prezzi", prezzi), ("Rendimenti", rendimenti)]:
        test = helper.test_stazionarieta(serie)
        print(f"{nome:<11} ADF p-value: {test['adf_pvalue']:.4f} | KPSS p-value: {test['kpss_pvalue']:.4f}")

    # -------------------------------------------------- 2. autocorrelazione
    titolo("2. Autocorrelazione nella media e nella varianza (Ljung-Box, 10 lag)")
    lb_rend = helper.ljung_box(rendimenti)
    lb_quad = helper.ljung_box(rendimenti**2)
    print(f"Rendimenti:           p-value min {lb_rend['lb_pvalue'].min():.4f}")
    print(f"Rendimenti al quadrato: p-value min {lb_quad['lb_pvalue'].min():.2e}")
    print(f"Test ARCH sui rendimenti: p-value {helper.test_arch(rendimenti):.2e}")

    # ---------------------------------------------------------------- 3. ARMA
    titolo("3. Selezione del modello ARMA")
    griglia = helper.griglia_arma(rendimenti)
    griglia.to_csv(CARTELLA_REPORT / "griglia_arma.csv", index=False)
    ordine_aic = helper.ordine_migliore(griglia, "aic")
    ordine_bic = helper.ordine_migliore(griglia, "bic")
    print(f"Ordine scelto dall'AIC: {ordine_aic} | dal BIC: {ordine_bic}")

    for ordine in sorted({ordine_aic, ordine_bic}):
        modello = helper.stima_arma(rendimenti, ordine)
        lb = helper.ljung_box(modello.resid)
        print(f"ARMA{ordine[0], ordine[2]}: Ljung-Box residui, p-value min sui 10 lag = "
              f"{lb['lb_pvalue'].min():.4f}")

    # --------------------------------------------------------------- 4. GARCH
    titolo("4. GARCH(1,1)")
    garch = helper.stima_garch(rendimenti)
    print(garch.summary())

    stats = helper.statistiche_garch(garch)
    print(f"\nPersistenza alpha + beta:          {stats['persistenza']:.4f}")
    print(f"Volatilità di lungo periodo:       {stats['vol_giornaliera_lungo_periodo_%']:.2f} % "
          f"giornaliera | {stats['vol_annua_lungo_periodo_%']:.2f} % annua")
    print(f"Half-life di uno shock:            {stats['half_life_giorni']:.1f} giorni")

    z = helper.residui_standardizzati(garch)
    print(f"\nDiagnostica sui residui standardizzati:")
    print(f"Ljung-Box z   (p-value min): {helper.ljung_box(z)['lb_pvalue'].min():.4f}")
    print(f"Ljung-Box z^2 (p-value min): {helper.ljung_box(z**2)['lb_pvalue'].min():.4f}")
    print(f"Test ARCH su z (p-value):    {helper.test_arch(z):.4f}")

    plt.figure(figsize=(14, 5))
    plt.plot(garch.conditional_volatility.index, garch.conditional_volatility * np.sqrt(252))
    plt.title("Volatilità condizionata GARCH(1,1) annualizzata - Brent (%)")
    salva_figura("volatilita_garch.png")

    # ---------------------------------------------------------------- 5. VAR
    titolo("5. VAR al 99%: statico, GARCH ed EWMA")
    df_var = helper.calcola_var(rendimenti, garch, livello=LIVELLO_VAR)
    tabella, attese, n = helper.conta_violazioni(df_var, livello=LIVELLO_VAR)
    per_anno = helper.violazioni_per_anno(df_var, livello=LIVELLO_VAR)

    print(f"Giorni analizzati: {n} | violazioni attese: {attese:.0f}\n")
    print(tabella)
    print("\nViolazioni per anno:")
    print(per_anno)

    tabella.to_csv(CARTELLA_REPORT / "violazioni_totali.csv")
    per_anno.to_csv(CARTELLA_REPORT / "violazioni_per_anno.csv")

    plt.figure(figsize=(14, 6))
    plt.plot(df_var.index, df_var["rendimenti"] * 100, label="Rendimenti", color="lightgray", linewidth=0.7)
    plt.plot(df_var.index, df_var["var_statico"] * 100, label="VAR statico", color="black", linestyle="--")
    plt.plot(df_var.index, df_var["var_garch"] * 100, label="VAR GARCH")
    plt.plot(df_var.index, df_var["var_ewma"] * 100, label="VAR EWMA", alpha=0.7)
    plt.title("Brent: VAR al 99% statico, GARCH ed EWMA (%)")
    plt.legend()
    salva_figura("var_confronto.png")

    per_anno[["Statico", "GARCH", "EWMA"]].plot(kind="bar", figsize=(14, 6), width=0.8)
    plt.axhline((1 - LIVELLO_VAR) * 252, color="red", linestyle="--", label="Attese per anno (≈ 2.5)")
    plt.title("Brent: violazioni del VAR al 99% per anno")
    plt.xlabel("Anno")
    plt.ylabel("Numero di violazioni")
    plt.legend()
    salva_figura("violazioni_per_anno.png")

    print(f"\nTabelle e grafici salvati in: {CARTELLA_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARMA, GARCH e VAR dinamico sul petrolio Brent")
    parser.add_argument("--offline", action="store_true",
                        help="usa i dati salvati in data/brent.csv invece di scaricarli")
    args = parser.parse_args()
    main(offline=args.offline)
