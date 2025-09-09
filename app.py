#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime
import logging
from collections import deque
from copy import deepcopy
import random
import requests

# scikit-learn (opcional)
SKLEARN_AVAILABLE = True
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score
except Exception:
    SKLEARN_AVAILABLE = False

MARKET_TICKER_BINANCE = "BTCUSDT"
MARKET_NAME = "BTC/USDT (Binance) | Crypto"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

DEFAULT_PARAMS = {
    'ml_enabled': True,
    'timeframes': [1, 5, 15],
    'weights': {'ml': 0.5, 'rule': 0.3, 'mtf': 0.2},
    'thresholds': {'buy': 60, 'sell': 60, 'strong': 75, 'rule_min_conditions': 3},
    'vol_filters': {
        'atr_window': 14,
        'atr_percentile_min': 20,
        'atr_percentile_max': 90,
        'volume_ratio_min': 0.9,
        'volume_ratio_max': 3.0,
        'realized_vol_window': 30,
        'realized_vol_min': 0.0004,
        'realized_vol_max': 0.02
    },
    'risk': {
        'atr_stop_mult': 1.5,
        'atr_target_mult': 2.5,
        'min_minutes_between_trades': 30,
        'max_trades_per_day': 6,
        'daily_target_pct_of_initial': 0.5
    },
    'ml': {'horizon': 3, 'n_estimators': 150, 'max_depth': 6, 'min_samples_leaf': 5, 'random_state': 42},
    'optimization': {'trials': 20, 'objective': 'sharpe'},
    'data': {'base_interval': '1m', 'max_hist_days': 7, 'allow_simulation': False}
}

def parse_iso(ts):
    try: return datetime.fromisoformat(ts.replace('Z', ''))
    except: return datetime.now()

def as_iso(dt):
    try: return dt.isoformat()
    except: return datetime.now().isoformat()

def parse_period_days(period_str, default_days=7):
    try:
        if isinstance(period_str, (int, float)): return int(period_str)
        s = str(period_str).strip().lower()
        if s.endswith('d'): return int(s[:-1])
        if s.endswith('w'): return int(s[:-1]) * 7
        if s.endswith('m'): return int(s[:-1]) * 30
        if s.endswith('y'): return int(s[:-1]) * 365
        return int(s)
    except: return default_days

# ---------------- Binance Provider ----------------
class BinanceProvider:
    def __init__(self, symbol='BTCUSDT', base_url='https://api.binance.com'):
        self.symbol = symbol
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (TradingBot)"})

    def _fetch_klines(self, interval='1m', start_ms=None, end_ms=None, limit=1000):
        url = f"{self.base_url}/api/v3/klines"
        params = {'symbol': self.symbol, 'interval': interval, 'limit': limit}
        if start_ms is not None: params['startTime'] = int(start_ms)
        if end_ms is not None: params['endTime'] = int(end_ms)
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_historical_df(self, days=7, interval='1m'):
        try:
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - int(days * 24 * 60 * 60 * 1000)
            klines, cur = [], start_ms
            step_ms = {'1m': 60_000, '3m': 180_000, '5m': 300_000, '15m': 900_000}.get(interval, 60_000)
            safety = 0
            while cur < end_ms and safety < 6000:
                batch = self._fetch_klines(interval=interval, start_ms=cur, end_ms=end_ms, limit=1000)
                if not batch: break
                klines.extend(batch)
                next_open = batch[-1][0] + step_ms
                if next_open <= cur: next_open = cur + step_ms
                cur = int(next_open)
                safety += 1
                time.sleep(0.03)
            if not klines: return pd.DataFrame()
            ot = [int(k[0]) for k in klines]
            o = [float(k[1]) for k in klines]
            h = [float(k[2]) for k in klines]
            l = [float(k[3]) for k in klines]
            c = [float(k[4]) for k in klines]
            v = [float(k[7]) for k in klines]  # quote volume (USDT aprox)
            idx = pd.to_datetime(ot, unit='ms', utc=True).tz_convert(None)
            df = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}, index=idx)
            df = df[~df.index.duplicated(keep='last')].sort_index()
            return df
        except Exception as e:
            logger.error(f"Erro histórico Binance: {e}")
            return pd.DataFrame()

    def get_latest_df(self, interval='1m'):
        try:
            data = self._fetch_klines(interval=interval, limit=1)
            if not data: return pd.DataFrame()
            k = data[-1]
            idx = pd.to_datetime([int(k[0])], unit='ms', utc=True).tz_convert(None)
            return pd.DataFrame({
                'open': [float(k[1])], 'high': [float(k[2])], 'low': [float(k[3])],
                'close': [float(k[4])], 'volume': [float(k[7])]
            }, index=idx)
        except Exception as e:
            logger.warning(f"Erro último kline Binance: {e}")
            return pd.DataFrame()

# ---------------- Data Collector ----------------
class OptimizedDataCollector:
    def __init__(self, provider=None, base_interval='1m', max_hist_days=7):
        self.provider = provider or BinanceProvider(symbol=MARKET_TICKER_BINANCE)
        self.base_interval = base_interval
        self.max_hist_days = max_hist_days
        self.data_source = "Binance"
        self.current_data = {}
        self.historical_data = deque(maxlen=12000)
        self.chart_data = deque(maxlen=500)
        self.is_running = False
        self.thread = None
        self.last_update = None
        self.last_fetch_ok = False
        self._last_bar_time = None
        self._initialize_historical_data()

    def _initialize_historical_data(self):
        logger.info(f"Inicializando dados (Binance) {MARKET_TICKER_BINANCE} {self.max_hist_days}d {self.base_interval}...")
        df = self.provider.get_historical_df(days=self.max_hist_days, interval=self.base_interval)
        if df is None or df.empty:
            logger.error("Histórico inicial vazio (Binance).")
            return
        for i, (idx, row) in enumerate(df.iterrows()):
            dp = {'timestamp': idx.isoformat(), 'price': float(row['close']),
                  'open': float(row['open']), 'high': float(row['high']),
                  'low': float(row['low']), 'close': float(row['close']),
                  'volume': float(row['volume'])}
            self.historical_data.append(dp)
            if i >= len(df) - self.chart_data.maxlen: self.chart_data.append(dp)
            self._last_bar_time = idx
        if self.historical_data:
            self.current_data = self.historical_data[-1]
            self.last_fetch_ok = True
        logger.info(f"Histórico pronto: {len(self.historical_data)}")

    def start_collection(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._collect_data_loop, name="DataCollector", daemon=True)
            self.thread.start()
            logger.info("Coleta (1m) iniciada")

    def stop_collection(self):
        self.is_running = False
        if self.thread: self.thread.join(timeout=2)
        logger.info("Coleta parada")

    def _collect_data_loop(self):
        while self.is_running:
            try:
                df_latest = self.provider.get_latest_df(interval=self.base_interval)
                if df_latest is not None and not df_latest.empty:
                    idx, row = df_latest.index[-1], df_latest.iloc[-1]
                    if self._last_bar_time is None or idx > self._last_bar_time:
                        data = {'timestamp': idx.isoformat(), 'price': float(row['close']),
                                'open': float(row['open']), 'high': float(row['high']),
                                'low': float(row['low']), 'close': float(row['close']),
                                'volume': float(row['volume'])}
                        self.current_data = data
                        self.historical_data.append(data)
                        self.chart_data.append(data)
                        self.last_update = datetime.now()
                        self.last_fetch_ok = True
                        self._last_bar_time = idx
                        logger.info(f"[{MARKET_TICKER_BINANCE}] {data['price']:.2f}")
                time.sleep(15)
            except Exception as e:
                self.last_fetch_ok = False
                logger.error(f"Loop coleta: {e}")
                time.sleep(30)

    def get_current_data(self): return self.current_data
    def get_historical_data(self): return list(self.historical_data)
    def get_chart_data(self): return list(self.chart_data)

    def get_historical_dataframe(self):
        if not self.historical_data: return pd.DataFrame()
        df = pd.DataFrame(self.historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df[['open','high','low','close','volume']].astype(float)

    def get_extended_historical_dataframe(self, period="7d", interval="1m"):
        days = parse_period_days(period, DEFAULT_PARAMS['data']['max_hist_days'])
        return self.provider.get_historical_df(days=days, interval=interval)

# ---------------- Indicators ----------------
class TechnicalIndicators:
    @staticmethod
    def calculate_indicators(data_list):
        if len(data_list) < 15: return {}
        try:
            prices = np.array([d['close'] for d in data_list], dtype=float)
            highs = np.array([d['high'] for d in data_list], dtype=float)
            lows  = np.array([d['low']  for d in data_list], dtype=float)
            volumes = np.array([d['volume'] for d in data_list], dtype=float)
            rsi = TechnicalIndicators._rsi(prices, 7)
            bb_period, bb_std_dev = 15, 1.8
            mid = np.mean(prices[-bb_period:]); std = np.std(prices[-bb_period:])
            upper, lower = mid + std*bb_std_dev, mid - std*bb_std_dev
            bb_pos = (prices[-1]-lower)/(upper-lower) if upper!=lower else 0.5
            ema = TechnicalIndicators._ema(prices, 8)
            p_vs_ema = (prices[-1]-ema)/ema if ema!=0 else 0.0
            momentum = (prices[-1]-prices[-4])/prices[-4] if len(prices)>3 else 0.0
            macd, macd_sig = TechnicalIndicators._macd(prices)
            stoch_k, stoch_d = TechnicalIndicators._stoch(highs, lows, prices, 14)
            vol_sma = np.mean(volumes[-10:]) if len(volumes)>=10 else volumes[-1]
            vol_ratio = volumes[-1]/vol_sma if vol_sma!=0 else 1.0
            atr = TechnicalIndicators._atr(highs, lows, prices, 14)
            ema_slope = TechnicalIndicators._ema_slope(prices, 8)
            return {
                'rsi': float(rsi), 'bb_upper': float(upper), 'bb_middle': float(mid), 'bb_lower': float(lower),
                'bb_position': float(bb_pos), 'ema': float(ema), 'price_vs_ema': float(p_vs_ema),
                'momentum': float(momentum), 'macd': float(macd), 'macd_signal': float(macd_sig),
                'macd_histogram': float(macd-macd_sig), 'stoch_k': float(stoch_k), 'stoch_d': float(stoch_d),
                'volume_ratio': float(vol_ratio), 'atr': float(atr), 'ema_slope': float(ema_slope)
            }
        except Exception as e:
            logger.error(f"Indicadores: {e}")
            return {}

    @staticmethod
    def _rsi(prices, period):
        if len(prices)<period+1: return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas>0, deltas, 0); losses = np.where(deltas<0, -deltas, 0)
        avg_gain = np.mean(gains[-period:]); avg_loss = np.mean(losses[-period:])
        if avg_loss==0: return 100.0
        rs = avg_gain/avg_loss
        return float(100 - (100/(1+rs)))

    @staticmethod
    def _ema(prices, period):
        if len(prices)<period: return float(np.mean(prices))
        mult = 2/(period+1); ema = prices[0]
        for p in prices[1:]: ema = p*mult + ema*(1-mult)
        return float(ema)

    @staticmethod
    def _macd(prices, fast=12, slow=26, signal=9):
        if len(prices)<slow: return 0.0, 0.0
        ef = TechnicalIndicators._ema(prices, fast)
        es = TechnicalIndicators._ema(prices, slow)
        line = ef - es
        sig = line if len(prices)<slow+signal else line*0.9
        return float(line), float(sig)

    @staticmethod
    def _stoch(highs, lows, closes, period=14):
        if len(closes)<period: return 50.0, 50.0
        hh = np.max(highs[-period:]); ll = np.min(lows[-period:]); c = closes[-1]
        if hh==ll: return 50.0, 50.0
        k = 100*(c-ll)/(hh-ll); d = 0.9*k
        return float(k), float(d)

    @staticmethod
    def _atr(highs, lows, closes, period=14):
        if len(closes)<2: return float(closes[-1]*0.01)
        tr = []
        for i in range(1,len(closes)):
            tr.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
        if len(tr)<period: return float(np.mean(tr)) if tr else float(closes[-1]*0.01)
        return float(np.mean(tr[-period:]))

    @staticmethod
    def _ema_slope(prices, period=8):
        if len(prices)<period+2: return 0.0
        mult = 2/(period+1); ema = prices[0]; vals=[]
        for p in prices[1:]:
            ema = p*mult + ema*(1-mult); vals.append(ema)
        if len(vals)<2: return 0.0
        return float((vals[-1]-vals[-2])/max(1e-9, vals[-2]))

# ---------------- MTF Features ----------------
class MultiTimeframeFeatureBuilder:
    def resample_ohlc(self, df, minutes):
        if df is None or df.empty: return pd.DataFrame()
        rule = f'{minutes}T'
        ohlc = df[['open','high','low','close']].resample(rule).agg({'open':'first','high':'max','low':'min','close':'last'})
        vol = df['volume'].resample(rule).sum()
        out = pd.concat([ohlc, vol.to_frame('volume')], axis=1).dropna(how='all')
        return out

    def indicators_df(self, df, rsi_period=7, bb_period=15, ema_period=8, stoch_period=14, atr_period=14, vol_sma_period=10):
        if df is None or df.empty: return pd.DataFrame()
        out = pd.DataFrame(index=df.index.copy())
        c = df['close'].values; h = df['high'].values; l = df['low'].values; v = df['volume'].values
        # Vetorizado com min_periods para não vaziar
        s = pd.Series(c, index=df.index)
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(rsi_period, min_periods=1).mean()
        loss = -delta.clip(upper=0).rolling(rsi_period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        out['rsi'] = (100 - (100/(1+rs))).fillna(50.0)
        mid = s.rolling(bb_period, min_periods=2).mean()
        std = s.rolling(bb_period, min_periods=2).std()
        upper = mid + std*1.8; lower = mid - std*1.8
        out['bb_pos'] = ((s - lower) / (upper - lower)).clip(0,1).fillna(0.5)
        ema = s.ewm(span=ema_period, adjust=False, min_periods=1).mean()
        out['ema'] = ema.values
        out['price_vs_ema'] = ((s - ema) / ema.replace(0, np.nan)).fillna(0.0)
        ema_fast = s.ewm(span=12, adjust=False, min_periods=1).mean()
        ema_slow = s.ewm(span=26, adjust=False, min_periods=1).mean()
        macd = ema_fast - ema_slow; sig = macd.ewm(span=9, adjust=False, min_periods=1).mean()
        out['macd_hist'] = (macd - sig).values
        hh = pd.Series(h, index=df.index).rolling(stoch_period, min_periods=2).max()
        ll = pd.Series(l, index=df.index).rolling(stoch_period, min_periods=2).min()
        out['stoch_k'] = (100*(s-ll)/(hh-ll)).replace([np.inf,-np.inf], np.nan).fillna(50.0)
        vol_sma = pd.Series(v, index=df.index).rolling(vol_sma_period, min_periods=1).mean()
        out['volume_ratio'] = (pd.Series(v, index=df.index) / vol_sma.replace(0, np.nan)).fillna(1.0)
        # ATR série
        hs = pd.Series(h, index=df.index); ls = pd.Series(l, index=df.index)
        hl = hs - ls; hc = (hs - s.shift(1)).abs(); lc = (ls - s.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        out['atr'] = tr.rolling(atr_period, min_periods=2).mean().fillna(method='bfill')
        out['realized_vol'] = s.pct_change().rolling(30, min_periods=5).std()
        out['ema_slope'] = ema.pct_change().fillna(0.0)
        # Limpeza branda
        out = out.replace([np.inf,-np.inf], np.nan).ffill().bfill()
        return out

    def build_mtf_features(self, df, timeframes=[1,5,15]):
        if df is None or df.empty: return pd.DataFrame()
        f1 = self.indicators_df(df).add_prefix('f1m_')
        if f1.empty: return pd.DataFrame()
        merged = f1.reset_index().rename(columns={'index':'timestamp'})
        for tf in timeframes:
            if tf==1: continue
            dfr = self.resample_ohlc(df, tf)
            if dfr is None or dfr.empty: continue
            fi = self.indicators_df(dfr).add_prefix(f'f{tf}m_')
            if fi is None or fi.empty: continue
            merged = pd.merge_asof(
                merged.sort_values('timestamp'),
                fi.reset_index().rename(columns={'index':'timestamp'}).sort_values('timestamp'),
                on='timestamp', direction='backward'
            )
        features_df = merged.set_index('timestamp').sort_index()
        # Preencher NaNs agressivamente
        features_df = features_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        # Se ainda houver NaN, usar mediana da coluna
        for col in features_df.columns:
            if features_df[col].isna().any():
                median = features_df[col].median()
                features_df[col] = features_df[col].fillna(median if pd.notna(median) else 0.0)
        return features_df.loc[:, ~features_df.columns.duplicated()]

    def build_ml_dataset(self, df, horizon=3, timeframes=[1,5,15]):
        feats = self.build_mtf_features(df, timeframes=timeframes)
        if feats is None or feats.empty:
            # Fallback: apenas 1m
            feats = self.indicators_df(df).add_prefix('f1m_')
            feats = feats.replace([np.inf,-np.inf], np.nan).ffill().bfill()
            for col in feats.columns:
                if feats[col].isna().any():
                    feats[col] = feats[col].fillna(feats[col].median() if pd.notna(feats[col].median()) else 0.0)
            if feats.empty:
                return np.empty((0,0)), np.array([], dtype=int), [], pd.DataFrame(), pd.DataFrame()
        base = df.loc[feats.index].copy()
        future = base['close'].shift(-horizon)
        y = (future > base['close']).astype(int)
        aligned = pd.concat([base, feats], axis=1)
        aligned = aligned.replace([np.inf,-np.inf], np.nan).ffill().bfill().dropna()
        if aligned.empty:
            return np.empty((0,0)), np.array([], dtype=int), [], pd.DataFrame(), pd.DataFrame()
        y = y.loc[aligned.index].astype(int)
        X = aligned[feats.columns].values
        return X, y.values, list(feats.columns), feats, aligned

    def latest_feature_row(self, df, timeframes=[1,5,15]):
        feats = self.build_mtf_features(df, timeframes=timeframes)
        if feats is None or feats.empty: return None, None
        return feats.iloc[-1].to_dict(), feats.columns.tolist()

# ---------------- ML Model ----------------
class MLPatternModel:
    def __init__(self, params=None):
        self.params = params or DEFAULT_PARAMS['ml']
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.trained = False
        self.lock = threading.Lock()
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn não disponível. ML desativado.")

    def train(self, X, y, feature_names):
        if not SKLEARN_AVAILABLE:
            self.trained = False
            return {'sklearn_available': False}
        with self.lock:
            try:
                if X is None or len(X)==0 or y is None or len(y)==0:
                    self.trained = False
                    return {'trained': False, 'error': 'Dataset vazio'}
                if len(np.unique(y))<2:
                    self.trained = False
                    return {'trained': False, 'error': 'Labels insuficientes'}
                self.feature_names = feature_names
                self.scaler = StandardScaler()
                Xs = self.scaler.fit_transform(X)
                self.model = RandomForestClassifier(
                    n_estimators=self.params.get('n_estimators',150),
                    max_depth=self.params.get('max_depth',6),
                    min_samples_leaf=self.params.get('min_samples_leaf',5),
                    random_state=self.params.get('random_state',42),
                    n_jobs=-1
                )
                n = len(Xs)
                n_splits = 5 if n>=300 else (4 if n>=150 else (3 if n>=100 else (2 if n>=60 else 0)))
                accs, aucs = [], []
                if n_splits>=2:
                    tss = TimeSeriesSplit(n_splits=n_splits)
                    for tr, te in tss.split(Xs):
                        self.model.fit(Xs[tr], y[tr])
                        preds = self.model.predict(Xs[te])
                        proba = self.model.predict_proba(Xs[te])[:,1]
                        accs.append(accuracy_score(y[te], preds))
                        try: aucs.append(roc_auc_score(y[te], proba))
                        except: aucs.append(np.nan)
                self.model.fit(Xs, y)
                self.trained = True
                return {'trained': True, 'acc_mean': float(np.nanmean(accs)) if accs else None,
                        'auc_mean': float(np.nanmean(aucs)) if aucs else None, 'splits': len(accs), 'n_samples': int(n)}
            except Exception as e:
                logger.error(f"Treino ML: {e}")
                self.trained = False
                return {'trained': False, 'error': str(e)}

    def predict_proba(self, feature_row_dict):
        if not SKLEARN_AVAILABLE or not self.trained or self.model is None or self.scaler is None:
            return 0.5
        with self.lock:
            try:
                x = np.array([feature_row_dict.get(f, 0.0) for f in self.feature_names], dtype=float).reshape(1,-1)
                xs = self.scaler.transform(x)
                return float(self.model.predict_proba(xs)[0,1])
            except Exception as e:
                logger.warning(f"Predição ML: {e}")
                return 0.5

# ---------------- Filters ----------------
class VolatilityVolumeFilter:
    @staticmethod
    def evaluate(df_1m, params):
        if df_1m is None or df_1m.empty:
            return True, 1.0, ["Filtro: sem dados"]
        p = params.get('vol_filters', DEFAULT_PARAMS['vol_filters'])
        win_atr = p.get('atr_window',14)
        min_len = max(20, win_atr+2, p.get('realized_vol_window',30)+2, 12)
        if len(df_1m)<min_len:
            return True, 1.0, ["Filtro: dados insuficientes"]
        h, l, c = df_1m['high'].values, df_1m['low'].values, df_1m['close'].values
        # ATR série segura
        s = pd.Series(c, index=df_1m.index)
        hs, ls = pd.Series(h, index=df_1m.index), pd.Series(l, index=df_1m.index)
        hl = hs - ls; hc = (hs - s.shift(1)).abs(); lc = (ls - s.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr_ser = tr.rolling(win_atr, min_periods=2).mean().fillna(method='bfill')
        if atr_ser.empty: return True, 1.0, ["Filtro: ATR indisponível"]
        atr_cur = float(atr_ser.iloc[-1])
        atr_clean = atr_ser.dropna()
        if len(atr_clean)<5: return True, 1.0, ["Filtro: ATR insuficiente"]
        atr_min = np.percentile(atr_clean, p['atr_percentile_min'])
        atr_max = np.percentile(atr_clean, p['atr_percentile_max'])
        vol_sma = df_1m['volume'].rolling(10, min_periods=1).mean()
        den = vol_sma.iloc[-1] if not np.isnan(vol_sma.iloc[-1]) else 1.0
        vol_ratio = float(df_1m['volume'].iloc[-1]/max(1e-9, den))
        rv = df_1m['close'].pct_change().rolling(p.get('realized_vol_window',30), min_periods=5).std().iloc[-1]
        pass_vol = (atr_cur>=atr_min) and (atr_cur<=atr_max)
        pass_volume = (vol_ratio>=p.get('volume_ratio_min',0.9)) and (vol_ratio<=p.get('volume_ratio_max',3.0))
        pass_rv = (rv>=p.get('realized_vol_min',0.0004)) and (rv<=p.get('realized_vol_max',0.02))
        reasons=[]; mult=1.0
        if not pass_vol: reasons.append("ATR fora do range"); mult*=0.6
        if not pass_volume: reasons.append("Volume ratio fora do range"); mult*=0.7
        if not pass_rv: reasons.append("Vol. realizada fora do range"); mult*=0.8
        if pass_vol and pass_volume and pass_rv: reasons.append("Filtros OK")
        return pass_vol and pass_volume and pass_rv, float(np.clip(mult,0.2,1.0)), reasons

# ---------------- Signal Generator ----------------
class AdvancedSignalGenerator:
    def __init__(self, params=None, feature_builder=None, ml_model=None):
        self.params = deepcopy(DEFAULT_PARAMS) if params is None else deepcopy(params)
        self.feature_builder = feature_builder or MultiTimeframeFeatureBuilder()
        self.ml_model = ml_model or MLPatternModel(self.params.get('ml', {}))
        self.last_signal = 'HOLD'
        self.signal_strength = 0
        self.lock = threading.Lock()

    def set_params(self, new_params):
        with self.lock: self.params = deepcopy(new_params)
    def get_params(self):
        with self.lock: return deepcopy(self.params)

    def _rule_scores(self, ind):
        buy, sell, br, sr = [], [], [], []
        try:
            if ind['rsi']<35: buy.append(1); br.append(f"RSI {ind['rsi']:.1f}")
            if ind['rsi']>65: sell.append(1); sr.append(f"RSI {ind['rsi']:.1f}")
            if ind['bb_position']<0.25: buy.append(1); br.append(f"BB low {ind['bb_position']:.2f}")
            if ind['bb_position']>0.75: sell.append(1); sr.append(f"BB high {ind['bb_position']:.2f}")
            if ind['price_vs_ema']<-0.005: buy.append(1); br.append(f"<EMA {ind['price_vs_ema']:.3f}")
            if ind['price_vs_ema']>0.005: sell.append(1); sr.append(f">EMA {ind['price_vs_ema']:.3f}")
            if ind['macd']>ind['macd_signal']: buy.append(1); br.append("MACD+")
            if ind['macd']<ind['macd_signal']: sell.append(1); sr.append("MACD-")
            if ind['stoch_k']<25: buy.append(1); br.append(f"Stoch {ind['stoch_k']:.1f}")
            if ind['stoch_k']>75: sell.append(1); sr.append(f"Stoch {ind['stoch_k']:.1f}")
            if ind['momentum']>0: buy.append(1)
            else: sell.append(1)
            maxc=6.0
            return min(1.0, sum(buy)/maxc), min(1.0, sum(sell)/maxc), br[:3], sr[:3]
        except:
            return 0.0, 0.0, [], []

    def _mtf_conf(self, df, tfs):
        if df is None or df.empty: return 0.0, 0.0, ["MTF: sem dados"]
        cl, cs, total, used = 0.0, 0.0, 0.0, []
        for tf in tfs:
            dfr = df if tf==1 else self.feature_builder.resample_ohlc(df, tf)
            if dfr is None or dfr.empty or len(dfr)<20: continue
            feats = self.feature_builder.indicators_df(dfr)
            if feats is None or feats.empty: continue
            last = feats.iloc[-1]
            if last.get('price_vs_ema',0)>0: cl += 1
            elif last.get('price_vs_ema',0)<0: cs += 1
            if last.get('macd_hist',0)>0: cl += 1
            elif last.get('macd_hist',0)<0: cs += 1
            r= float(last.get('rsi',50))
            if 45<=r<=70: cl += 0.5
            if 30<=r<=55: cs += 0.5
            total += 2.5
            used.append(tf)
        if total==0 or not used: return 0.0, 0.0, ["MTF indisponível"]
        return float(np.clip(cl/total,0,1)), float(np.clip(cs/total,0,1)), [f"MTF usando {', '.join(f'{x}m' for x in used)}",
                                                                             f"MTF long={cl/total:.2f} short={cs/total:.2f}"]

    def generate_signal(self, current_data, indicators, historical_data, df_1m=None):
        try:
            if not indicators or len(historical_data)<30:
                return {'signal':'HOLD','strength':0,'reason':'Dados insuficientes','entry_price':None,'stop_loss':None,'take_profit':None}
            p = self.get_params()
            w = p['weights']; th = p['thresholds']
            if df_1m is None:
                df_1m = pd.DataFrame(historical_data)
                df_1m['timestamp']=pd.to_datetime(df_1m['timestamp'])
                df_1m=df_1m.set_index('timestamp').sort_index()[['open','high','low','close','volume']].astype(float)
            rb_l, rb_s, rb_lr, rb_sr = self._rule_scores(indicators)
            tf_l, tf_s, mtf_r = self._mtf_conf(df_1m, p['timeframes'])
            if p['ml_enabled'] and SKLEARN_AVAILABLE and self.ml_model.trained:
                feat_row, _ = self.feature_builder.latest_feature_row(df_1m, timeframes=p['timeframes'])
                p_up = self.ml_model.predict_proba(feat_row) if feat_row is not None else 0.5
            else: p_up = 0.5
            pass_f, mult_f, filt_r = VolatilityVolumeFilter.evaluate(df_1m.tail(500), p)
            long_score = 100*(w['rule']*rb_l + w['mtf']*tf_l + w['ml']*p_up)*mult_f
            short_score= 100*(w['rule']*rb_s + w['mtf']*tf_s + w['ml']*(1-p_up))*mult_f
            price = current_data['price']; atr = indicators.get('atr', price*0.01)
            reasons = mtf_r + filt_r
            if long_score>=th['buy'] and long_score>short_score:
                return {'signal':'BUY','strength':float(np.clip(long_score,0,100)),
                        'reason':'; '.join((rb_lr+reasons)[:6]),
                        'entry_price':price,'stop_loss':price-atr*DEFAULT_PARAMS['risk']['atr_stop_mult'],
                        'take_profit':price+atr*DEFAULT_PARAMS['risk']['atr_target_mult'],
                        'details':{'rb_long':rb_l,'rb_short':rb_s,'tf_long':tf_l,'tf_short':tf_s,'ml_p_up':p_up,
                                   'filter_ok':pass_f,'filter_mult':mult_f}}
            if short_score>=th['sell'] and short_score>long_score:
                return {'signal':'SELL','strength':float(np.clip(short_score,0,100)),
                        'reason':'; '.join((rb_sr+reasons)[:6]),
                        'entry_price':price,'stop_loss':price+atr*DEFAULT_PARAMS['risk']['atr_stop_mult'],
                        'take_profit':price-atr*DEFAULT_PARAMS['risk']['atr_target_mult'],
                        'details':{'rb_long':rb_l,'rb_short':rb_s,'tf_long':tf_l,'tf_short':tf_s,'ml_p_up':p_up,
                                   'filter_ok':pass_f,'filter_mult':mult_f}}
            return {'signal':'HOLD','strength':0,'reason':'Aguardando condições favoráveis; ' + '; '.join(reasons),
                    'entry_price':None,'stop_loss':None,'take_profit':None,
                    'details':{'rb_long':rb_l,'rb_short':rb_s,'tf_long':tf_l,'tf_short':tf_s,'ml_p_up':p_up,
                               'filter_ok':pass_f,'filter_mult':mult_f}}
        except Exception as e:
            logger.error(f"Sinal: {e}")
            return {'signal':'HOLD','strength':0,'reason':f'Erro: {str(e)}','entry_price':None,'stop_loss':None,'take_profit':None}

# ---------------- Trading Engine ----------------
class TradingEngine:
    def __init__(self, initial_capital=500):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.trades_history = []
        self.current_pnl = 0
        self.total_pnl = 0
        self.trades_today = 0
        self.trade_points = []
        self.position_size = 0
        self.last_trade_time = None
        self.daily_target_reached = False

    def execute_trade(self, signal, current_data, indicators, params=None):
        try:
            params = params or DEFAULT_PARAMS
            price = current_data['price']; ts = current_data['timestamp']
            atr = indicators.get('atr', price*0.01)
            if self.total_pnl >= self.initial_capital*(params['risk']['daily_target_pct_of_initial']/100.0):
                if not self.daily_target_reached:
                    logger.info(f"🎯 Meta diária atingida! P&L: ${self.total_pnl:.2f}")
                    self.daily_target_reached = True
                return
            if self.position!=0:
                if self._check_exit_conditions(price, atr, signal):
                    self._close_position(price, ts, "Condições de saída")
            if self.position==0:
                if self.last_trade_time:
                    delta = parse_iso(ts) - parse_iso(self.last_trade_time)
                    if delta.total_seconds()<params['risk']['min_minutes_between_trades']*60: return
                if self.trades_today>=params['risk']['max_trades_per_day']: return
                if signal['signal']=='BUY' and signal['strength']>=50: self._open_long(price, ts, signal, atr)
                elif signal['signal']=='SELL' and signal['strength']>=50: self._open_short(price, ts, signal, atr)
            if self.position!=0: self._update_current_pnl(price)
        except Exception as e:
            logger.error(f"Exec trade: {e}")

    def _open_long(self, price, ts, signal, atr):
        self.position=1; self.entry_price=price; self.entry_time=ts; self.last_trade_time=ts
        self.position_size = self.current_capital/price
        sl = price - atr*DEFAULT_PARAMS['risk']['atr_stop_mult']
        tp = price + atr*DEFAULT_PARAMS['risk']['atr_target_mult']
        self.trades_history.append({'type':'BUY','entry_price':price,'entry_time':ts,'stop_loss':sl,'take_profit':tp,
                                    'signal_strength':signal['strength'],'reason':signal['reason'],
                                    'position_size':self.position_size,'status':'open'})
        self.trades_today += 1
        self.trade_points.append({'timestamp':ts,'price':price,'type':'BUY','strength':signal['strength']})
        logger.info(f"🟢 BUY {MARKET_TICKER_BINANCE}: {price:.2f} | qty: {self.position_size:.6f}")

    def _open_short(self, price, ts, signal, atr):
        self.position=-1; self.entry_price=price; self.entry_time=ts; self.last_trade_time=ts
        self.position_size = self.current_capital/price
        sl = price + atr*DEFAULT_PARAMS['risk']['atr_stop_mult']
        tp = price - atr*DEFAULT_PARAMS['risk']['atr_target_mult']
        self.trades_history.append({'type':'SELL','entry_price':price,'entry_time':ts,'stop_loss':sl,'take_profit':tp,
                                    'signal_strength':signal['strength'],'reason':signal['reason'],
                                    'position_size':self.position_size,'status':'open'})
        self.trades_today += 1
        self.trade_points.append({'timestamp':ts,'price':price,'type':'SELL','strength':signal['strength']})
        logger.info(f"🔴 SELL {MARKET_TICKER_BINANCE}: {price:.2f} | qty: {self.position_size:.6f}")

    def _close_position(self, price, ts, reason):
        if self.position==0: return
        pnl_abs = (price - self.entry_price)*self.position_size if self.position==1 else (self.entry_price - price)*self.position_size
        capital_used = self.entry_price*self.position_size
        pnl_pct = (pnl_abs/capital_used)*100 if capital_used!=0 else 0
        self.current_capital += pnl_abs; self.total_pnl += pnl_abs
        if self.trades_history:
            self.trades_history[-1].update({'exit_price':price,'exit_time':ts,'pnl_absolute':pnl_abs,'pnl_percent':pnl_pct,
                                            'capital_after':self.current_capital,'exit_reason':reason,'status':'closed'})
        exit_type = 'SELL_EXIT' if self.position==1 else 'BUY_EXIT'
        self.trade_points.append({'timestamp':ts,'price':price,'type':exit_type,'pnl':pnl_abs,'pnl_percent':pnl_pct})
        logger.info(f"✅ EXIT {MARKET_TICKER_BINANCE} @ {price:.2f} | P&L: ${pnl_abs:.2f} ({pnl_pct:.2f}%) | Capital: ${self.current_capital:.2f} | {reason}")
        self.position=0; self.entry_price=0; self.entry_time=None; self.current_pnl=0; self.position_size=0

    def _check_exit_conditions(self, price, atr, signal):
        if self.position==0 or not self.trades_history: return False
        last = self.trades_history[-1]
        if self.position==1:
            if price<=last['stop_loss'] or price>=last['take_profit']: return True
        else:
            if price>=last['stop_loss'] or price<=last['take_profit']: return True
        if signal['strength']>=67:
            if self.position==1 and signal['signal']=='SELL': return True
            if self.position==-1 and signal['signal']=='BUY': return True
        if self.entry_time:
            try:
                if (datetime.now()-parse_iso(self.entry_time)).total_seconds()>10800: return True
            except: pass
        return False

    def _update_current_pnl(self, price):
        self.current_pnl = (price - self.entry_price)*self.position_size if self.position==1 else (self.entry_price - price)*self.position_size

    def get_trade_points(self): return self.trade_points[-50:]

    def get_performance(self):
        closed=[t for t in self.trades_history if t.get('status')=='closed']
        if not closed:
            return {'total_trades': len(self.trades_history), 'closed_trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'best_trade': 0, 'worst_trade': 0}
        wins=[t for t in closed if t.get('pnl_absolute',0)>0]
        pnls=[t.get('pnl_absolute',0) for t in closed]
        return {'total_trades': len(self.trades_history),'closed_trades': len(closed),
                'win_rate': len(wins)/len(closed)*100,'avg_pnl': float(np.mean(pnls)),
                'best_trade': float(np.max(pnls)),'worst_trade': float(np.min(pnls))}

# ---------------- Backtest & Optimize ----------------
class Backtester:
    def __init__(self, feature_builder, signal_generator_class, params):
        self.feature_builder = feature_builder
        self.signal_generator_class = signal_generator_class
        self.params = deepcopy(params)

    def run(self, df, walkforward=False, retrain_every=500, initial_train_ratio=0.6, initial_capital=500):
        if df is None or df.empty or len(df)<400:
            return {'success': False, 'error': 'Dados insuficientes (>=400 barras)'}
        eng = TradingEngine(initial_capital=initial_capital)
        sig = self.signal_generator_class(params=self.params, feature_builder=self.feature_builder, ml_model=MLPatternModel(self.params.get('ml', {})))
        tfs = self.params.get('timeframes',[1,5,15]); horizon = self.params.get('ml',{}).get('horizon',3)
        X, y, fcols, feats, aligned = self.feature_builder.build_ml_dataset(df, horizon=horizon, timeframes=tfs)
        rep={'trained': False}
        if self.params.get('ml_enabled',True) and SKLEARN_AVAILABLE and len(y)>0 and len(np.unique(y))>=2:
            cutoff=int(len(X)*initial_train_ratio)
            sig.ml_model.train(X[:cutoff], y[:cutoff], fcols); rep={'trained': True}
        start=int(len(aligned)*initial_train_ratio)
        times=aligned.index
        last_train=start
        for i in range(start, len(aligned)-1):
            if walkforward and self.params.get('ml_enabled',True) and SKLEARN_AVAILABLE:
                if (i-last_train)>=retrain_every:
                    Xw,yw,_,_,_ = self.feature_builder.build_ml_dataset(df.loc[:times[i]], horizon=horizon, timeframes=tfs)
                    if len(yw)>0 and len(np.unique(yw))>=2: sig.ml_model.train(Xw, yw, fcols)
                    last_train=i
            hist_slice=df.loc[:times[i]].copy()
            if len(hist_slice)<30: continue
            row=aligned.iloc[i]
            cur={'timestamp': as_iso(times[i].to_pydatetime()), 'price': float(row['close']),
                 'open': float(row['open']), 'high': float(row['high']), 'low': float(row['low']),
                 'close': float(row['close']), 'volume': float(row['volume'])}
            hlist=self._df_to_list(hist_slice)
            ind=TechnicalIndicators.calculate_indicators(hlist)
            if not ind: continue
            signal=sig.generate_signal(cur, ind, hlist, df_1m=hist_slice)
            eng.execute_trade(signal, cur, ind, params=self.params)
            if i==len(aligned)-2 and eng.position!=0:
                eng._close_position(cur['price'], cur['timestamp'], "Fechamento Backtest")
        perf=eng.get_performance()
        eq=[t.get('capital_after', eng.initial_capital) for t in eng.trades_history if t.get('status')=='closed']
        eq=[eng.initial_capital]+eq
        ret=pd.Series(eq).pct_change().dropna()
        sharpe=float(np.sqrt(252*24*60)*ret.mean()/(ret.std()+1e-9)) if not ret.empty else 0.0
        mdd=self._max_drawdown(eq)
        return {'success': True, 'train_report': rep, 'performance': perf, 'final_capital': float(eng.current_capital),
                'total_pnl': float(eng.total_pnl), 'sharpe': sharpe, 'max_drawdown': float(mdd), 'trades': eng.trades_history}

    def _df_to_list(self, df):
        out=[]
        for idx,row in df.iterrows():
            out.append({'timestamp': as_iso(idx.to_pydatetime()), 'price': float(row['close']),
                        'open': float(row['open']), 'high': float(row['high']),
                        'low': float(row['low']), 'close': float(row['close']),
                        'volume': float(row['volume'])})
        return out

    def _max_drawdown(self, equity):
        arr=np.array(equity, dtype=float)
        if len(arr)<2: return 0.0
        roll=np.maximum.accumulate(arr); dd=(arr-roll)/roll
        return float(np.min(dd))

class Optimizer:
    def __init__(self, backtester): self.backtester = backtester
    def random_search(self, df, trials=20, objective='sharpe'):
        best=None; best_score=-1e9; cands=[]
        for _ in range(trials):
            p=deepcopy(self.backtester.params)
            w_ml=random.choice([0.3,0.4,0.5,0.6]); w_mtf=random.choice([0.1,0.2,0.3]); w_rule=float(np.clip(1.0-w_ml-w_mtf,0.1,0.8))
            p['weights']={'ml':w_ml,'mtf':w_mtf,'rule':w_rule}
            p['thresholds']['buy']=random.choice([55,60,65,70]); p['thresholds']['sell']=random.choice([55,60,65,70])
            p['vol_filters']['volume_ratio_min']=random.choice([0.8,0.9,1.0,1.1])
            p['vol_filters']['volume_ratio_max']=random.choice([2.5,3.0,3.5])
            p['ml']['n_estimators']=random.choice([100,150,200]); p['ml']['max_depth']=random.choice([4,6,8]); p['ml']['min_samples_leaf']=random.choice([3,5,8])
            bt=Backtester(self.backtester.feature_builder, self.backtester.signal_generator_class, p)
            res=bt.run(df, walkforward=False, initial_train_ratio=0.6)
            if not res.get('success'): continue
            score = res.get('sharpe',0.0) if objective=='sharpe' else res.get('total_pnl',0.0)
            cands.append({'params':deepcopy(p),'result':res,'score':score})
            if score>best_score: best_score=score; best=cands[-1]; logger.info(f"Novo melhor {objective}: {best_score:.4f}")
        return {'best':best,'candidates':cands}

# ---------------- Bot ----------------
class OptimizedTradingBot:
    def __init__(self):
        self.data_collector = OptimizedDataCollector(BinanceProvider(MARKET_TICKER_BINANCE),
                                                     DEFAULT_PARAMS['data']['base_interval'],
                                                     DEFAULT_PARAMS['data']['max_hist_days'])
        self.indicators_calculator = TechnicalIndicators()
        self.feature_builder = MultiTimeframeFeatureBuilder()
        self.ml_model = MLPatternModel(DEFAULT_PARAMS['ml'])
        self.signal_generator = AdvancedSignalGenerator(params=DEFAULT_PARAMS, feature_builder=self.feature_builder, ml_model=self.ml_model)
        self.trading_engine = TradingEngine()
        self.is_running=False; self.trading_thread=None

    def start_bot(self):
        if not self.is_running:
            self.is_running=True
            self.data_collector.start_collection()
            self.trading_thread=threading.Thread(target=self._trading_loop, name="TradingLoop", daemon=True)
            self.trading_thread.start()
            logger.info("Bot iniciado"); return True
        return False

    def stop_bot(self):
        if self.is_running:
            self.is_running=False
            self.data_collector.stop_collection()
            if self.trading_thread: self.trading_thread.join(timeout=2)
            logger.info("Bot parado"); return True
        return False

    def _trading_loop(self):
        while self.is_running:
            try:
                cur=self.data_collector.get_current_data()
                hist=self.data_collector.get_historical_data()
                df1=self.data_collector.get_historical_dataframe()
                if cur and len(hist)>=30 and df1 is not None and not df1.empty:
                    ind=self.indicators_calculator.calculate_indicators(hist)
                    if ind:
                        sig=self.signal_generator.generate_signal(cur, ind, hist, df_1m=df1)
                        self.trading_engine.execute_trade(sig, cur, ind, params=self.signal_generator.get_params())
                time.sleep(10)
            except Exception as e:
                logger.error(f"Loop trading: {e}")
                time.sleep(20)

    def get_status(self):
        cur=self.data_collector.get_current_data()
        hist=self.data_collector.get_historical_data()
        df1=self.data_collector.get_historical_dataframe()
        ind={}
        if hist and len(hist)>=15: ind=self.indicators_calculator.calculate_indicators(hist)
        sig={'signal':'HOLD','strength':0,'reason':'Sem dados'}
        if cur and ind and df1 is not None and not df1.empty:
            sig=self.signal_generator.generate_signal(cur, ind, hist, df_1m=df1)
        perf=self.trading_engine.get_performance()
        return {'is_running': self.is_running, 'market': MARKET_NAME, 'ticker': MARKET_TICKER_BINANCE,
                'data_source': self.data_collector.data_source, 'last_fetch_ok': self.data_collector.last_fetch_ok,
                'current_data': cur, 'indicators': ind, 'signal': sig, 'position': self.trading_engine.position,
                'entry_price': self.trading_engine.entry_price, 'current_pnl': self.trading_engine.current_pnl,
                'total_pnl': self.trading_engine.total_pnl, 'current_capital': self.trading_engine.current_capital,
                'trades_today': self.trading_engine.trades_today, 'data_points': len(hist),
                'performance': perf,
                'ml': {'enabled': self.signal_generator.get_params().get('ml_enabled',False) and SKLEARN_AVAILABLE,
                       'trained': self.ml_model.trained, 'sklearn_available': SKLEARN_AVAILABLE},
                'params': self.signal_generator.get_params()}

    def get_chart_data(self):
        candles=self.data_collector.get_chart_data(); tps=self.trading_engine.get_trade_points()
        out=[]
        for d in candles:
            out.append({'timestamp': d['timestamp'],'open': d['open'],'high': d['high'],
                        'low': d['low'],'close': d['close'],'volume': d['volume']})
        return {'candlesticks': out, 'trade_points': tps}

    # Fallbacks de dataset + treino robusto
    def _build_dataset_fallback(self, df, base_horizon, base_tfs):
        tf_candidates = [base_tfs, [1,5], [1,15], [5,15], [1], [5], [15]]
        horizons = [base_horizon, 1]
        for tfs in tf_candidates:
            for hz in horizons:
                X,y,fn,_,aligned = self.feature_builder.build_ml_dataset(df, horizon=hz, timeframes=tfs)
                if X.size>0 and len(y)>=60 and len(np.unique(y))>=2:
                    return X,y,fn,aligned.index, hz, tfs, 'direction'
        # Limiar por retorno (descarta neutros)
        tfs = base_tfs
        X,y,fn,feats,aligned = self.feature_builder.build_ml_dataset(df, horizon=base_horizon, timeframes=tfs)
        if X.size==0 or aligned is None or aligned.empty:
            return np.empty((0,0)), np.array([]), [], [], base_horizon, tfs, 'failed'
        ret_fwd = aligned['close'].shift(-base_horizon)/aligned['close'] - 1.0
        abs_ret = ret_fwd.abs().dropna()
        if abs_ret.empty: return np.empty((0,0)), np.array([]), [], [], base_horizon, tfs, 'failed'
        thr = float(np.percentile(abs_ret, 60))
        keep = abs_ret >= max(thr, 1e-6)
        idx = abs_ret.index[keep]
        aligned2 = aligned.loc[idx].dropna()
        if len(aligned2)<60: return np.empty((0,0)), np.array([]), [], [], base_horizon, tfs, 'failed'
        y2 = (ret_fwd.loc[aligned2.index] > 0).astype(int).values
        X2 = aligned2[fn].values
        if len(np.unique(y2))<2: return np.empty((0,0)), np.array([]), [], [], base_horizon, tfs, 'failed'
        return X2, y2, fn, aligned2.index, base_horizon, tfs, 'threshold'

    def train_ml(self, period="7d", interval="1m"):
        attempts=[('7d', interval), ('14d', interval)]
        last_err="Desconhecido"
        for per, inter in attempts:
            df=self.data_collector.get_extended_historical_dataframe(period=per, interval=inter)
            if df is None or df.empty or len(df)<300:
                last_err=f"Histórico insuficiente ({per}/{inter})"; continue
            tfs=self.signal_generator.get_params().get('timeframes',[1,5,15])
            base_hz=self.signal_generator.get_params().get('ml',{}).get('horizon',3)
            X,y,fn,idx,hz_used,tfs_used,method = self._build_dataset_fallback(df, base_hz, tfs)
            if X.size==0 or len(y)<60 or len(np.unique(y))<2:
                last_err=f"Dataset vazio ({per}, horizon={hz_used})"; continue
            rep=self.ml_model.train(X,y,fn)
            if rep.get('trained'):
                return {'success': True, 'report': {**rep,'period':per,'interval':inter,'horizon':hz_used,'timeframes_used':tfs_used,'method':method}}
            last_err=rep.get('error','Falha treino ML')
        return {'success': False, 'error': last_err}

    def backtest(self, period="7d", interval="1m", walkforward=False):
        df=self.data_collector.get_extended_historical_dataframe(period=period, interval=interval)
        if df is None or df.empty: return {'success': False, 'error': 'Histórico indisponível'}
        bt=Backtester(self.feature_builder, AdvancedSignalGenerator, self.signal_generator.get_params())
        return bt.run(df, walkforward=walkforward, initial_train_ratio=0.6)

    def optimize(self, period="7d", interval="1m", trials=20, objective='sharpe'):
        df=self.data_collector.get_extended_historical_dataframe(period=period, interval=interval)
        if df is None or df.empty: return {'success': False, 'error': 'Histórico indisponível'}
        bt=Backtester(self.feature_builder, AdvancedSignalGenerator, self.signal_generator.get_params())
        opt=Optimizer(bt); res=opt.random_search(df, trials=trials, objective=objective)
        if res.get('best'):
            self.signal_generator.set_params(res['best']['params'])
            return {'success': True, 'best': res['best'], 'count': len(res['candidates'])}
        return {'success': False, 'error': 'Otimização sem solução válida'}

trading_bot = OptimizedTradingBot()

# ---------------- Routes ----------------
@app.route('/')
def index(): return send_from_directory('.', 'interface_advanced.html')

@app.route('/api/bot/start', methods=['POST'])
def start_bot(): return jsonify({'success': trading_bot.start_bot()})

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot(): return jsonify({'success': trading_bot.stop_bot()})

@app.route('/api/bot/status')
def get_bot_status(): return jsonify(trading_bot.get_status())

@app.route('/api/bot/chart')
def get_chart_data(): return jsonify(trading_bot.get_chart_data())

@app.route('/api/bot/reset', methods=['POST'])
def reset_bot():
    global trading_bot
    trading_bot.stop_bot()
    time.sleep(1)
    trading_bot = OptimizedTradingBot()
    return jsonify({'success': True})

@app.route('/api/bot/train_ml', methods=['POST'])
def api_train_ml():
    payload = request.get_json(force=True, silent=True) or {}
    period = payload.get('period', '7d'); interval = payload.get('interval', '1m')
    return jsonify(trading_bot.train_ml(period=period, interval=interval))

@app.route('/api/bot/backtest', methods=['POST'])
def api_backtest():
    payload = request.get_json(force=True, silent=True) or {}
    period = payload.get('period', '7d'); interval = payload.get('interval', '1m')
    walkforward = bool(payload.get('walkforward', False))
    return jsonify(trading_bot.backtest(period=period, interval=interval, walkforward=walkforward))

@app.route('/api/bot/optimize', methods=['POST'])
def api_optimize():
    payload = request.get_json(force=True, silent=True) or {}
    period = payload.get('period', '7d'); interval = payload.get('interval', '1m')
    trials = int(payload.get('trials', DEFAULT_PARAMS['optimization']['trials']))
    objective = payload.get('objective', DEFAULT_PARAMS['optimization']['objective'])
    return jsonify(trading_bot.optimize(period=period, interval=interval, trials=trials, objective=objective))

@app.route('/api/bot/params', methods=['GET'])
def api_params(): return jsonify(trading_bot.signal_generator.get_params())

@app.route('/api/bot/set_params', methods=['POST'])
def api_set_params():
    payload = request.get_json(force=True, silent=True) or {}
    current = trading_bot.signal_generator.get_params()
    def deep_update(d,u):
        for k,v in u.items():
            if isinstance(v,dict) and k in d: deep_update(d[k],v)
            else: d[k]=v
    deep_update(current, payload)
    trading_bot.signal_generator.set_params(current)
    return jsonify({'success': True, 'params': trading_bot.signal_generator.get_params()})

if __name__=='__main__':
    print("🚀 Trading Bot Avançado - BTC/USDT (Binance)")
    print("✅ Dados reais (Binance) 1m | MTF + ML robusto")
    print(f"🧠 scikit-learn disponível: {SKLEARN_AVAILABLE}")
    print("🌐 Interface: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)