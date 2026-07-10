import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import ta
import pandas_market_calendars as mcal
import os

nyse = mcal.get_calendar('NYSE')

# =====================================================
# CONFIGURACION
# =====================================================
from datetime import datetime
from zoneinfo import ZoneInfo
import sys

# Hora de Nueva York
ahora_ny = datetime.now(ZoneInfo("America/New_York"))

# Lunes=0 ... Viernes=4
es_dia_laborable = ahora_ny.weekday() < 5

# Mercado regular USA: 09:30 - 16:00
mercado_abierto = (
    es_dia_laborable and
    (
        (ahora_ny.hour > 9 or (ahora_ny.hour == 9 and ahora_ny.minute >= 30))
        and
        ahora_ny.hour < 16
    )
)

csv_vacio = False

if os.path.exists("predicciones.csv"):
    try:
        df_csv = pd.read_csv("predicciones.csv")
        csv_vacio = df_csv.empty
    except Exception:
        csv_vacio = True
else:
    csv_vacio = True

if mercado_abierto and not csv_vacio:
    print("Mercado USA abierto. No se actualiza el CSV.")
    sys.exit(0)

if mercado_abierto and csv_vacio:
    print("Mercado abierto, pero el CSV está vacío. Se regenerará.")
TICKERS = {
    "tecnologicas": ['TSM', 'AAPL','NVDA','TSLA']
}

ATR_SL_MULT = 1.2
RISK_REWARD_BASE = 1.5
ADX_TREND_THRESHOLD = 25

multiplicadores = {
    "AAPL": 1.00042,
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
# INDICADORES
# =====================================================

def add_indicators(df):

    df['RSI'] = ta.momentum.RSIIndicator(
        df['Close'],
        window=14
    ).rsi()

    df['SMA_100'] = df['Close'].rolling(
        window=100
    ).mean()

    df['ATR'] = ta.volatility.average_true_range(
        df['High'],
        df['Low'],
        df['Close'],
        window=14
    )

    df['ADX'] = ta.trend.ADXIndicator(
        df['High'],
        df['Low'],
        df['Close'],
        window=14
    ).adx()

    return df

# =====================================================
# PREDICCION
# =====================================================

def calcular_prediccion_en_fecha(df_total, fecha_idx, ticker, grupo):

    df = df_total.iloc[:fecha_idx+1].copy()

    if len(df) < 300:
        return None

    train = df.iloc[:-1]
    last_row = df.iloc[-1]

    X_pred = last_row.drop(
        ['Open','High','Low','Close','Volume']
    ).to_frame().T

    # =================================================
    # MODELO CLOSE
    # =================================================

    dfc = train.copy()

    dfc['Target'] = dfc['Close'].shift(-1)

    dfc.dropna(inplace=True)

    X = dfc.drop(
        ['Open','High','Low','Close','Volume','Target'],
        axis=1
    )

    y = dfc['Target']

    model_close = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42
    )

    model_close.fit(X, y)

    close_pred = float(
        model_close.predict(X_pred)[0]
    )

    # =================================================
    # MODELO ATR
    # =================================================

    dfv = train.copy()

    dfv['Target'] = dfv['ATR'].shift(-1)

    dfv.dropna(inplace=True)

    Xv = dfv.drop(
        ['Open','High','Low','Close','Volume','Target'],
        axis=1
    )

    yv = dfv['Target']

    model_vol = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42
    )

    model_vol.fit(Xv, yv)

    atr_pred = float(
        model_vol.predict(X_pred)[0]
    )

    # =================================================
    # REGIMEN
    # =================================================

    if last_row['ADX'] > ADX_TREND_THRESHOLD:

        ATR_ENTRY_MULT = 0.9
        RISK_REWARD = 2.0

    else:

        ATR_ENTRY_MULT = 0.5
        RISK_REWARD = RISK_REWARD_BASE

    # =================================================
    # NIVELES
    # =================================================

    entry_long = close_pred - ATR_ENTRY_MULT * atr_pred

    sl_long = entry_long - ATR_SL_MULT * atr_pred

    tp_long = entry_long + RISK_REWARD * (
        entry_long - sl_long
    )

    entry_short = close_pred + ATR_ENTRY_MULT * atr_pred

    sl_short = entry_short + ATR_SL_MULT * atr_pred

    tp_short = entry_short - RISK_REWARD * (
        sl_short - entry_short
    )

    # =================================================
    # MULTIPLICADOR
    # =================================================

    mult = multiplicadores.get(ticker, 1.0)

    entry_long *= mult
    sl_long *= mult
    tp_long *= mult

    entry_short *= mult
    sl_short *= mult
    tp_short *= mult

    # =================================================
    # TICKER CFD
    # =================================================

    ticker_cfd = accion_a_cfd.get(
        ticker,
        ticker
    )

    fecha_actual = df_total.index[fecha_idx].date()
    
    schedule = nyse.schedule(
        start_date=fecha_actual,
        end_date=fecha_actual + pd.Timedelta(days=10)
    )
    
    fecha_signal = schedule.index[1].strftime("%Y-%m-%d")
    # =================================================
    # FILA CSV
    # =================================================

    fila = {

        "fecha": fecha_signal,

        "grupo": grupo,

        "ticker": ticker_cfd,

        "entry_long": round(entry_long, 5),

        "tp_long": round(tp_long, 5),

        "sl_long": round(sl_long, 5),

        "entry_short": round(entry_short, 5),

        "tp_short": round(tp_short, 5),

        "sl_short": round(sl_short, 5)
    }

    return fila

# =====================================================
# =====================================================
# MAIN
# =====================================================

resultados = []

for grupo, lista in TICKERS.items():

    for ticker in lista:

        print("===================================")
        print("Descargando:", ticker)

        try:

            data = yf.Ticker(ticker).history(
                start="2000-01-01"
            )

            print("Ahora NY:", ahora_ny)

            if len(data) == 0:

                print("Sin datos:", ticker)

                continue

            print("Últimas 5 filas descargadas:")
            print(data.tail(5))

            print("Último índice bruto:", data.index[-1])
            print("Última fecha descargada:", data.index[-1].date())

            data = add_indicators(data)

            data.dropna(inplace=True)

            i = len(data) - 1

            print("i =", i)
            print(
                "Procesando:",
                ticker,
                data.index[i].date()
            )

            print("Fecha usada para predecir:", data.index[i])
            fecha_actual = data.index[i].date()
            
            schedule = nyse.schedule(
                start_date=fecha_actual,
                end_date=fecha_actual + pd.Timedelta(days=10)
            )
            
            print(
                "Fecha siguiente sesión NYSE:",
                schedule.index[1]
            )

            fila = calcular_prediccion_en_fecha(
                data,
                i,
                ticker,
                grupo
            )

            if fila:
                print("Fila generada:", fila)
                resultados.append(fila)

        except Exception as e:

            print("ERROR:", ticker, e)

# =====================================================
# DATAFRAME
# =====================================================

df_resultados = pd.DataFrame(resultados)

# =====================================================
# GUARDAR CSV
# =====================================================
if len(resultados) == 0:
    print("ERROR: No se ha podido generar ninguna predicción.")
    print("Se mantiene el CSV anterior.")
    sys.exit(1)
df_resultados.to_csv(
    "predicciones.csv",
    index=False
)

print("\n===================================")
print("CSV generado correctamente")
print("Total señales:", len(df_resultados))
print("===================================")

print(df_resultados)
