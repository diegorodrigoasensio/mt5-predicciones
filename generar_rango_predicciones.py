import argparse
import os
import sys
from datetime import datetime

import pytz
import pandas as pd
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestRegressor
from dateutil.easter import easter

# pandas_market_calendars no funciona en Python < 3.10. Si falla el import,
# se usa el calendario NYSE propio definido mas abajo.
try:
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar('NYSE')
except Exception as e:
    nyse = None
    print("AVISO: pandas_market_calendars no disponible (", type(e).__name__,
          "). Se usa calendario NYSE interno.")

# Cierres extraordinarios de NYSE (duelos nacionales, etc.)
CIERRES_ESPECIALES = {
    pd.Timestamp("2018-12-05"),  # G. H. W. Bush
    pd.Timestamp("2025-01-09"),  # J. Carter
    pd.Timestamp("2012-10-29"), pd.Timestamp("2012-10-30"),  # Sandy
}

def _nth_weekday(year, month, weekday, n):
    d = pd.Timestamp(year, month, 1)
    d += pd.Timedelta(days=(weekday - d.weekday()) % 7)
    return d + pd.Timedelta(weeks=n - 1)

def _last_weekday(year, month, weekday):
    d = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() - weekday) % 7)

def _observado(d):
    # Sabado -> viernes anterior, domingo -> lunes siguiente
    if d.weekday() == 5:
        return d - pd.Timedelta(days=1)
    if d.weekday() == 6:
        return d + pd.Timedelta(days=1)
    return d

def festivos_nyse(year):
    h = []
    ny = pd.Timestamp(year, 1, 1)
    if ny.weekday() == 6:
        h.append(ny + pd.Timedelta(days=1))
    elif ny.weekday() != 5:          # si cae en sabado NYSE no cierra el viernes
        h.append(ny)
    h.append(_nth_weekday(year, 1, 0, 3))                          # MLK
    h.append(_nth_weekday(year, 2, 0, 3))                          # Presidents
    h.append(pd.Timestamp(easter(year)) - pd.Timedelta(days=2))    # Viernes Santo
    h.append(_last_weekday(year, 5, 0))                            # Memorial Day
    if year >= 2022:
        h.append(_observado(pd.Timestamp(year, 6, 19)))            # Juneteenth
    h.append(_observado(pd.Timestamp(year, 7, 4)))                 # Independence
    h.append(_nth_weekday(year, 9, 0, 1))                          # Labor Day
    h.append(_nth_weekday(year, 11, 3, 4))                         # Thanksgiving
    h.append(_observado(pd.Timestamp(year, 12, 25)))               # Navidad
    return h

def sesiones_nyse(ini, fin):
    """Sesiones NYSE entre ini y fin (inclusive), sin zona horaria."""
    if nyse is not None:
        s = nyse.schedule(start_date=ini, end_date=fin).index
        return pd.DatetimeIndex(s).tz_localize(None).normalize()
    festivos = set(CIERRES_ESPECIALES)
    for y in range(ini.year, fin.year + 1):
        festivos.update(festivos_nyse(y))
    return pd.DatetimeIndex([d for d in pd.bdate_range(ini, fin) if d not in festivos])

# =====================================================
# CONFIGURACION
# =====================================================

# Rango de fechas de SEÑAL (la fecha que saldra en AddPred, es decir,
# la sesion en la que operara el EA). Formato "YYYY-MM-DD".
#   - Ambas en None  -> modo normal: solo la proxima sesion (como antes)
#   - Solo DESDE     -> un unico dia
#   - DESDE y HASTA  -> todas las sesiones NYSE del rango (inclusive)
# Tambien se pueden pasar por linea de comandos (tienen prioridad):
#   python generar_predicciones.py --desde 2026-06-01 --hasta 2026-07-10
FECHA_DESDE = "2021-01-01"
FECHA_HASTA = "2026-11-02"

TICKERS = {
    "tecnologicas": ['AMD']
}

ATR_SL_MULT = 1.2
RISK_REWARD_BASE = 1.5
ADX_TREND_THRESHOLD = 25

multiplicadores = {
    "AAPL": 1.00042,
    "AMD": 1.00042,
    "BBVA.MC": 1.008605,
    "GOOG": 1.000348,
    "IAG.MC": 0.998109,
    "META": 0.999911,
    "NVDA": 1.000032,
    "SAB.MC": 1.006444,
    "SAN.MC": 1.002492,
    "SMCI": 0.999054
}

accion_a_cfd = {
    "AAPL": "AAPL",
    "NVDA": "NVDA",
    "AMZN": "AMAZN",
    "GOOG": "GOOG",
    "MSFT": "MSFT",
    "AMD": "AMD",
    "META": "META.US",
    "TSLA": "TSLA",
    "SMCI": "SMCI.US",
    "BBVA.MC": "BBVA",
    "SAN.MC": "SANES",
    "SAB.MC": "SAB2",
    "IAG.MC": "IAG.ES"
}

# =====================================================
# ARGUMENTOS
# =====================================================

parser = argparse.ArgumentParser()
parser.add_argument("--desde", default=FECHA_DESDE, help="Fecha de señal inicial YYYY-MM-DD")
parser.add_argument("--hasta", default=FECHA_HASTA, help="Fecha de señal final YYYY-MM-DD")
parser.add_argument("--tickers", default=None, help="Lista separada por comas (ej: AMD,AAPL). Sustituye a TICKERS")
parser.add_argument("--salida", default="predicciones_addpred.txt", help="Archivo con las lineas AddPred")
parser.add_argument("--csv", action="store_true", help="Escribe tambien predicciones.csv")
args, _ = parser.parse_known_args()  # parse_known_args: ignora los argumentos internos de Jupyter

if args.hasta and not args.desde:
    print("ERROR: si indicas --hasta tienes que indicar tambien --desde.")
    sys.exit(1)

modo_rango = args.desde is not None
desde = pd.Timestamp(args.desde) if args.desde else None
hasta = pd.Timestamp(args.hasta) if args.hasta else desde

if modo_rango and hasta < desde:
    print("ERROR: --hasta no puede ser anterior a --desde.")
    sys.exit(1)

if args.tickers:
    TICKERS = {"tecnologicas": [t.strip() for t in args.tickers.split(",") if t.strip()]}

# =====================================================
# CONTROL MERCADO ABIERTO (solo modo normal)
# =====================================================

ahora_ny = datetime.now(pytz.timezone("America/New_York"))
es_dia_laborable = ahora_ny.weekday() < 5

mercado_abierto = (
    es_dia_laborable and
    ((ahora_ny.hour > 9 or (ahora_ny.hour == 9 and ahora_ny.minute >= 30))
     and ahora_ny.hour < 16)
)

if not modo_rango:
    csv_vacio = False
    if os.path.exists("predicciones.csv"):
        try:
            csv_vacio = pd.read_csv("predicciones.csv").empty
        except Exception:
            csv_vacio = True
    else:
        csv_vacio = True

    if mercado_abierto and not csv_vacio:
        print("Mercado USA abierto. No se actualiza el CSV.")
        sys.exit(0)

    if mercado_abierto and csv_vacio:
        print("Mercado abierto, pero el CSV está vacío. Se regenerará.")

# =====================================================
# CALENDARIO NYSE
# =====================================================

hoy = pd.Timestamp(ahora_ny.date())
ini_cal = (desde if modo_rango else hoy) - pd.Timedelta(days=15)
fin_cal = (hasta if modo_rango else hoy) + pd.Timedelta(days=20)

sesiones = sesiones_nyse(ini_cal, fin_cal)

def siguiente_sesion(fecha):
    """Primera sesion NYSE estrictamente posterior a 'fecha'."""
    pos = sesiones.searchsorted(pd.Timestamp(fecha), side="right")
    if pos >= len(sesiones):
        return None
    return sesiones[pos]

# =====================================================
# INDICADORES
# =====================================================

def add_indicators(df):

    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    df['SMA_100'] = df['Close'].rolling(window=100).mean()

    df['ATR'] = ta.volatility.average_true_range(
        df['High'], df['Low'], df['Close'], window=14
    )

    df['ADX'] = ta.trend.ADXIndicator(
        df['High'], df['Low'], df['Close'], window=14
    ).adx()

    return df

# =====================================================
# PREDICCION
# =====================================================

def calcular_prediccion_en_fecha(df_total, fecha_idx, ticker, grupo, fecha_signal):

    # Solo se usan datos hasta fecha_idx (sin mirar al futuro)
    df = df_total.iloc[:fecha_idx + 1].copy()

    if len(df) < 300:
        return None

    train = df.iloc[:-1]
    last_row = df.iloc[-1]

    X_pred = last_row.drop(['Open', 'High', 'Low', 'Close', 'Volume']).to_frame().T

    # ---------------- MODELO CLOSE ----------------
    dfc = train.copy()
    dfc['Target'] = dfc['Close'].shift(-1)
    dfc.dropna(inplace=True)

    X = dfc.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target'], axis=1)
    y = dfc['Target']

    model_close = RandomForestRegressor(
        n_estimators=100, max_depth=20, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model_close.fit(X, y)
    close_pred = float(model_close.predict(X_pred)[0])

    # ---------------- MODELO ATR ----------------
    dfv = train.copy()
    dfv['Target'] = dfv['ATR'].shift(-1)
    dfv.dropna(inplace=True)

    Xv = dfv.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target'], axis=1)
    yv = dfv['Target']

    model_vol = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model_vol.fit(Xv, yv)
    atr_pred = float(model_vol.predict(X_pred)[0])

    # ---------------- REGIMEN ----------------
    if last_row['ADX'] > ADX_TREND_THRESHOLD:
        ATR_ENTRY_MULT = 0.9
        RISK_REWARD = 2.0
    else:
        ATR_ENTRY_MULT = 0.5
        RISK_REWARD = RISK_REWARD_BASE

    # ---------------- NIVELES ----------------
    entry_long = close_pred - ATR_ENTRY_MULT * atr_pred
    sl_long = entry_long - ATR_SL_MULT * atr_pred
    tp_long = entry_long + RISK_REWARD * (entry_long - sl_long)

    entry_short = close_pred + ATR_ENTRY_MULT * atr_pred
    sl_short = entry_short + ATR_SL_MULT * atr_pred
    tp_short = entry_short - RISK_REWARD * (sl_short - entry_short)

    # ---------------- MULTIPLICADOR ----------------
    mult = multiplicadores.get(ticker, 1.0)

    entry_long *= mult
    sl_long *= mult
    tp_long *= mult
    entry_short *= mult
    sl_short *= mult
    tp_short *= mult

    ticker_cfd = accion_a_cfd.get(ticker, ticker)

    return {
        "fecha": fecha_signal.strftime("%Y-%m-%d"),
        "grupo": grupo,
        "ticker": ticker_cfd,
        "entry_long": round(entry_long, 5),
        "tp_long": round(tp_long, 5),
        "sl_long": round(sl_long, 5),
        "entry_short": round(entry_short, 5),
        "tp_short": round(tp_short, 5),
        "sl_short": round(sl_short, 5),
    }

# =====================================================
# FORMATO AddPred
# =====================================================

def fila_a_addpred(f):
    return (
        f'AddPred("{f["fecha"]}","{f["grupo"]}","{f["ticker"]}",'
        f'{f["entry_long"]:.5f},{f["tp_long"]:.5f},{f["sl_long"]:.5f},'
        f'{f["entry_short"]:.5f},{f["tp_short"]:.5f},{f["sl_short"]:.5f});'
    )

# =====================================================
# MAIN
# =====================================================

import time

CACHE_HORAS = 12      # si el cache tiene menos horas, no se vuelve a llamar a Yahoo
REINTENTOS = 4
ESPERA_BASE_SEG = 30  # 30s, 60s, 90s...

def descargar_datos(ticker):
    """Descarga con cache local y reintentos para evitar el 'Too Many Requests'."""
    ruta = f"cache_{ticker.replace('.', '_')}.csv"

    def leer_cache():
        d = pd.read_csv(ruta, index_col=0)
        d.index = pd.to_datetime(d.index, utc=True).tz_convert("America/New_York")
        return d

    if os.path.exists(ruta):
        edad_h = (time.time() - os.path.getmtime(ruta)) / 3600
        if edad_h < CACHE_HORAS:
            print(f"{ticker}: usando cache local ({edad_h:.1f} h)")
            return leer_cache()

    ultimo_error = None
    for intento in range(1, REINTENTOS + 1):
        try:
            d = yf.Ticker(ticker).history(start="2000-01-01")
            if len(d) > 0:
                d.to_csv(ruta)
                return d
            ultimo_error = "sin datos"
        except Exception as e:
            ultimo_error = e
        espera = ESPERA_BASE_SEG * intento
        print(f"{ticker}: intento {intento}/{REINTENTOS} fallido ({ultimo_error}). Esperando {espera}s...")
        time.sleep(espera)

    if os.path.exists(ruta):
        print(f"AVISO: {ticker}: no se pudo descargar, uso cache ANTIGUO. Los datos pueden estar desactualizados.")
        return leer_cache()

    raise RuntimeError(f"No se pudo descargar {ticker}: {ultimo_error}")

resultados = []

for grupo, lista in TICKERS.items():

    for ticker in lista:

        print("===================================")
        print("Descargando:", ticker)

        try:
            data = descargar_datos(ticker)

            if len(data) == 0:
                print("Sin datos:", ticker)
                continue

            data = add_indicators(data)
            data.dropna(inplace=True)

            fechas_datos = pd.DatetimeIndex(
                [d.date() for d in data.index]
            )

            # ---- Seleccion de indices a procesar ----
            if modo_rango:
                candidatos = [
                    i for i, d in enumerate(fechas_datos)
                    if (desde - pd.Timedelta(days=15)) <= d <= hasta
                ]
            else:
                candidatos = [len(data) - 1]

            pendientes = []
            for i in candidatos:
                fs = siguiente_sesion(fechas_datos[i])
                if fs is None:
                    continue
                if modo_rango and not (desde <= fs <= hasta):
                    continue
                pendientes.append((i, fs))

            print(f"{ticker}: {len(pendientes)} fecha(s) a calcular")

            for n, (i, fs) in enumerate(pendientes, start=1):
                print(f"  [{n}/{len(pendientes)}] datos hasta {fechas_datos[i].date()} -> señal {fs.date()}")

                fila = calcular_prediccion_en_fecha(data, i, ticker, grupo, fs)

                if fila:
                    resultados.append(fila)

        except Exception as e:
            print("ERROR:", ticker, e)

# =====================================================
# GUARDAR
# =====================================================

if len(resultados) == 0:
    print("ERROR: No se ha podido generar ninguna predicción.")
    sys.exit(1)

df_resultados = pd.DataFrame(resultados)

# Orden: por ticker (en el orden configurado) y despues por fecha
orden_tickers = [accion_a_cfd.get(t, t) for lista in TICKERS.values() for t in lista]
df_resultados["_o"] = df_resultados["ticker"].map({t: k for k, t in enumerate(orden_tickers)})
df_resultados = df_resultados.sort_values(["_o", "fecha"]).drop(columns="_o").reset_index(drop=True)

lineas = [fila_a_addpred(f) for f in df_resultados.to_dict("records")]

with open(args.salida, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lineas) + "\n")

if args.csv or not modo_rango:
    df_resultados.to_csv("predicciones.csv", index=False)

print("\n===================================")
print("Total señales:", len(lineas))
print("Archivo AddPred:", args.salida)
if args.csv or not modo_rango:
    print("CSV: predicciones.csv")
print("===================================\n")

print("\n".join(lineas))
