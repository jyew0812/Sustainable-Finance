# Backend portfolio logic (data loading, analytics, and chart helpers)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter
import yfinance as yf
import logging
import streamlit as st
from pathlib import Path

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 16,
    'axes.labelsize': 11.5,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 9.5,
})
COLORS = {
    'primary': '#2d5c88',
    'secondary': '#5a6f8c',
    'accent': '#2f7a59',
    'warning': '#c77d2b',
    'danger': '#b14444',
    'ink': '#1f2933',
    'muted': '#64748b',
    'line_soft': '#dde5ed',
    'panel': '#ffffff',
}
APP_BG = '#f0f7f2'
TICKER_ALIASES = {
    # Renamed / re-listed symbols
    'CDAY': 'DAY',
    'WRK': 'SW',
    # Known delisted / unavailable symbols (skip in universe batch)
    'CTLT': None,
    'WBA': None,
}
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').propagate = False


def classify_risk(gamma):
    if gamma >= 6.5:
        return 'Defensive'
    elif gamma >= 4.5:
        return 'Balanced'
    return 'Growth-Oriented'

def classify_esg(lambda_raw_avg):
    if lambda_raw_avg >= 3.5:
        return 'Sustainability-Led'
    elif lambda_raw_avg >= 2.5:
        return 'ESG-Aware'
    return 'Low ESG Priority'
SIN_INDUSTRIES = ['Tobacco', 'Gambling', 'Resorts & Casinos', 'Oil & Gas E&P', 'Oil & Gas Integrated', 'Oil & Gas Midstream', 'Oil & Gas Refining & Marketing', 'Oil & Gas Equipment & Services', 'Oil & Gas Drilling', 'Thermal Coal']
ESG_DEFAULT_SOURCE = 'https://raw.githubusercontent.com/jyew0812/Sustainable-Finance/2c5bfb9bdef7e7c6aa2007c3783ef4170b30827b/finalized/clean_esg_app_data.csv'

def _normalise_ticker(value):
    if value is None:
        return ''
    return str(value).strip().upper().replace('$', '').replace(' ', '')

def _ticker_variants(ticker):
    t = _normalise_ticker(ticker)
    if not t:
        return []
    variants = [t]
    if '-' in t:
        variants.append(t.replace('-', '.'))
    if '.' in t:
        variants.append(t.replace('.', '-'))
    return list(dict.fromkeys(variants))

def _mapped_ticker(ticker):
    t = _normalise_ticker(ticker)
    if not t:
        return ''
    return TICKER_ALIASES.get(t, t)

def _download_candidates(ticker):
    t = _normalise_ticker(ticker)
    if not t:
        return []
    mapped = _mapped_ticker(t)
    candidates = []
    if mapped:
        candidates.extend(_ticker_variants(mapped))
    if mapped != t:
        candidates.extend(_ticker_variants(t))
    if not candidates:
        candidates.extend(_ticker_variants(t))
    return list(dict.fromkeys([c for c in candidates if c]))

def weighted_esg(e, s, g, w_e, w_s, w_g):
    return w_e * e + w_s * s + w_g * g

def _normalise_score_series(series):
    s = pd.to_numeric(series, errors='coerce')
    valid = s.dropna()
    # Keep ESG values on a 0-100 scale for all downstream calculations/UI.
    if len(valid) > 0 and valid.max() <= 1.0:
        s = s * 100.0
    return s.clip(lower=0.0, upper=100.0)

def _to_raw_github_url(source):
    if not isinstance(source, str):
        return source
    if 'github.com' in source and '/blob/' in source:
        source = source.replace('https://github.com/', 'https://raw.githubusercontent.com/')
        source = source.replace('/blob/', '/')
    return source

def _load_esg_data_impl(path_or_file):
    required_strict = ['ticker', 'fieldid', 'valuescore']
    date_col_candidates = ['valuedate', 'year']
    all_accepted = set(required_strict) | set(date_col_candidates)
    source = _to_raw_github_url(path_or_file)
    source_name = getattr(path_or_file, 'name', str(source)).lower()
    if source_name.endswith('.xlsx') or source_name.endswith('.xls'):
        try:
            esg_raw = pd.read_excel(source, dtype=str)
        except Exception as exc:
            raise ValueError(f'Unable to read ESG Excel file: {exc}')
        esg_raw.columns = [str(c).strip().lower() for c in esg_raw.columns]
        missing = [c for c in required_strict if c not in esg_raw.columns]
        if missing:
            raise ValueError(f'Missing required ESG columns in Excel file: {missing}')
        date_col = next((c for c in date_col_candidates if c in esg_raw.columns), None)
        cols_to_load = required_strict + ([date_col] if date_col else [])
        esg_filtered = esg_raw[cols_to_load].copy()
        esg_filtered = esg_filtered[esg_filtered['fieldid'].isin(['4', '5', '6'])]
    else:
        chunks = []
        last_error = None
        for enc in ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']:
            try:
                reader = pd.read_csv(source, usecols=lambda c: str(c).strip().lower() in all_accepted, dtype=str, chunksize=300000, encoding=enc)
                for chunk in reader:
                    chunk.columns = [str(c).strip().lower() for c in chunk.columns]
                    missing_cols = [c for c in required_strict if c not in chunk.columns]
                    if missing_cols:
                        continue
                    chunk = chunk[chunk['fieldid'].isin(['4', '5', '6'])]
                    chunks.append(chunk)
                last_error = None
                break
            except Exception as exc:
                chunks = []
                last_error = exc
        if last_error is not None:
            raise ValueError(f'Unable to read ESG CSV file: {last_error}')
        if not chunks:
            # Fallback: support cleaned/wide ESG CSVs (e.g. EnvironmentPillarScore columns).
            wide_accepted = {
                'ticker', 'year',
                'environmentpillarscore', 'socialpillarscore', 'governancepillarscore',
                'esgcombinedscore'
            }
            try:
                wide = pd.read_csv(
                    source,
                    usecols=lambda c: str(c).strip().lower() in wide_accepted,
                    dtype=str,
                )
                wide.columns = [str(c).strip().lower() for c in wide.columns]
            except Exception as exc:
                raise ValueError(f'Unable to read ESG CSV file: {exc}')

            required_wide = ['ticker', 'environmentpillarscore', 'socialpillarscore', 'governancepillarscore']
            missing_wide = [c for c in required_wide if c not in wide.columns]
            if missing_wide:
                raise ValueError('No ESG rows were found for fieldid 4, 5, 6 in the ESG CSV file.')

            wide['ticker'] = wide['ticker'].map(_normalise_ticker)
            for c in ['environmentpillarscore', 'socialpillarscore', 'governancepillarscore', 'esgcombinedscore']:
                if c in wide.columns:
                    wide[c] = pd.to_numeric(wide[c], errors='coerce')
            if 'year' in wide.columns:
                wide['year'] = pd.to_numeric(wide['year'], errors='coerce')
                wide = wide.sort_values(['ticker', 'year'])
                latest = wide.drop_duplicates(subset=['ticker'], keep='last')
            else:
                latest = wide.drop_duplicates(subset=['ticker'], keep='last')

            pivot = latest.rename(columns={
                'environmentpillarscore': 'E',
                'socialpillarscore': 'S',
                'governancepillarscore': 'G',
                'esgcombinedscore': 'ESG',
            })[['ticker', 'E', 'G', 'S']].copy()
            pivot['ESG'] = pd.to_numeric(latest.get('esgcombinedscore'), errors='coerce')
            if pivot['ESG'].isna().all():
                pivot['ESG'] = (pivot['E'] + pivot['G'] + pivot['S']) / 3
            pivot = pivot.dropna(subset=['ticker'])
            for col in ['E', 'G', 'S', 'ESG']:
                pivot[col] = _normalise_score_series(pivot[col])

            long_for_compat = pivot[['ticker', 'E', 'G', 'S']].melt(
                id_vars=['ticker'],
                value_vars=['E', 'G', 'S'],
                var_name='pillar',
                value_name='valuescore',
            )
            long_for_compat['fieldid'] = long_for_compat['pillar'].map({'E': 4, 'G': 5, 'S': 6})
            long_for_compat['valuedate'] = pd.NaT
            esg_lookup = pivot.set_index('ticker')[['E', 'G', 'S', 'ESG']].to_dict(orient='index')
            return (long_for_compat[['ticker', 'fieldid', 'valuescore', 'valuedate']], pivot[['ticker', 'E', 'G', 'S', 'ESG']], esg_lookup)
        esg_filtered = pd.concat(chunks, ignore_index=True)
    if 'valuedate' not in esg_filtered.columns and 'year' in esg_filtered.columns:
        esg_filtered = esg_filtered.rename(columns={'year': 'valuedate'})
    esg_filtered['ticker'] = esg_filtered['ticker'].map(_normalise_ticker)
    esg_filtered['fieldid'] = pd.to_numeric(esg_filtered['fieldid'], errors='coerce').astype('Int64')
    esg_filtered['valuescore'] = pd.to_numeric(esg_filtered['valuescore'], errors='coerce')
    if 'valuedate' in esg_filtered.columns:
        sample = esg_filtered['valuedate'].dropna().head(20)
        if all((str(v).strip().isdigit() and len(str(v).strip()) == 4 for v in sample)):
            esg_filtered['valuedate'] = pd.to_datetime(esg_filtered['valuedate'].astype(str).str.strip() + '-12-31', errors='coerce')
        else:
            esg_filtered['valuedate'] = pd.to_datetime(esg_filtered['valuedate'], errors='coerce')
    else:
        esg_filtered['valuedate'] = pd.NaT
    valid_scores = esg_filtered['valuescore'].dropna()
    if len(valid_scores) > 0 and valid_scores.max() <= 1.0:
        esg_filtered['valuescore'] = esg_filtered['valuescore'] * 100.0
    esg_filtered = esg_filtered.dropna(subset=['ticker', 'fieldid', 'valuescore'])
    esg_filtered = esg_filtered.sort_values(['ticker', 'fieldid', 'valuedate'])
    latest_by_ticker = esg_filtered.drop_duplicates(subset=['ticker', 'fieldid'], keep='last')
    pivot = latest_by_ticker.pivot(index='ticker', columns='fieldid', values='valuescore').reset_index()
    pivot = pivot.rename(columns={4: 'E', 5: 'G', 6: 'S'})
    for col in ['E', 'G', 'S']:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot['ESG'] = 0.33 * pivot['E'] + 0.33 * pivot['G'] + 0.33 * pivot['S']
    for col in ['E', 'G', 'S', 'ESG']:
        pivot[col] = _normalise_score_series(pivot[col])
    esg_lookup = pivot.set_index('ticker')[['E', 'G', 'S', 'ESG']].to_dict(orient='index')
    return (esg_filtered, pivot[['ticker', 'E', 'G', 'S', 'ESG']], esg_lookup)

@st.cache_data(show_spinner=False)
def load_esg_data_from_path(path):
    return _load_esg_data_impl(path)

def load_esg_data_from_uploaded(uploaded_file):
    uploaded_file.seek(0)
    return _load_esg_data_impl(uploaded_file)

@st.cache_data(show_spinner=False)
def fetch_ticker_profile(ticker):
    if not ticker:
        return None
    candidates = _download_candidates(ticker)
    best_profile = None
    for candidate in candidates:
        try:
            ticker_obj = yf.Ticker(candidate)
        except Exception:
            continue
        info = {}
        try:
            info = ticker_obj.get_info() or {}
        except Exception:
            info = {}
        if not info:
            try:
                info = ticker_obj.info or {}
            except Exception:
                info = {}
        sector = info.get('sector') or info.get('sectorDisp') or info.get('sectorKey') or 'Unavailable'
        industry = info.get('industry') or info.get('industryDisp') or info.get('industryKey') or 'Unavailable'
        short_name = info.get('shortName') or info.get('longName') or info.get('displayName') or candidate
        if sector == 'Unavailable' or industry == 'Unavailable':
            try:
                search = yf.Search(candidate, max_results=1)
                quotes = getattr(search, 'quotes', []) or []
            except Exception:
                quotes = []
            if quotes:
                quote = quotes[0]
                short_name = quote.get('shortname') or quote.get('longname') or quote.get('dispSecIndFlag') or short_name
                sector = quote.get('sector') or quote.get('sectorDisp') or quote.get('sectorKey') or sector
                industry = quote.get('industry') or quote.get('industryDisp') or quote.get('industryKey') or industry
        if sector == 'Unavailable' or industry == 'Unavailable':
            try:
                quote_type = ticker_obj.get_history_metadata() or {}
            except Exception:
                quote_type = {}
            short_name = quote_type.get('shortName') or short_name
        profile = {'name': short_name, 'sector': sector, 'industry': industry}
        if sector != 'Unavailable' or industry != 'Unavailable':
            return profile
        if best_profile is None:
            best_profile = profile
    return best_profile or {'name': ticker, 'sector': 'Unavailable', 'industry': 'Unavailable'}

@st.cache_data(show_spinner=False)
def fetch_market_data(tickers, ticker2=None, period='3y'):
    if isinstance(tickers, (list, tuple, pd.Index, np.ndarray)):
        clean_tickers = [_normalise_ticker(t) for t in tickers if _normalise_ticker(t)]
    else:
        clean_tickers = [_normalise_ticker(tickers), _normalise_ticker(ticker2)]
    clean_tickers = [t for t in clean_tickers if t]
    clean_tickers = list(dict.fromkeys(clean_tickers))
    if len(clean_tickers) < 2:
        raise ValueError('Please provide at least two valid ticker symbols.')

    def _download_close_series(raw_ticker):
        candidates = _download_candidates(raw_ticker)
        for candidate in candidates:
            try:
                data = yf.download(
                    candidate,
                    period=period,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            except Exception:
                continue
            if data is None or data.empty:
                continue
            if 'Close' in data.columns:
                close = data['Close']
            elif isinstance(data, pd.Series):
                close = data
            else:
                continue
            if isinstance(close, pd.DataFrame):
                if close.shape[1] == 1:
                    close = close.iloc[:, 0]
                else:
                    matching_cols = [c for c in close.columns if str(c).upper() == candidate]
                    close = close[matching_cols[0]] if matching_cols else close.iloc[:, 0]
            close = pd.to_numeric(close, errors='coerce').dropna()
            if len(close) >= 30:
                return close
        tried = ", ".join(candidates) if candidates else raw_ticker
        raise ValueError(
            f'No recent Yahoo price data for ticker "{raw_ticker}" (tried: {tried}). '
            'The ticker may be valid, but Yahoo Finance returned no recent history for this request. '
            'This is often caused by temporary Yahoo throttling or unavailable cloud responses.'
        )

    price_series = [_download_close_series(ticker).rename(ticker) for ticker in clean_tickers]
    prices = pd.concat(price_series, axis=1).dropna()
    if prices.empty or len(prices) < 30:
        raise ValueError('Insufficient overlapping price history between the selected tickers. Choose a longer period or different symbols.')
    returns = prices.pct_change(fill_method=None).dropna()
    mean_returns = returns.mean() * 252
    volatilities = returns.std() * np.sqrt(252)
    covariance = returns.cov() * 252
    correlation_matrix = returns.corr()
    market_data = {
        'tickers': clean_tickers,
        'prices': prices,
        'returns': returns,
        'mean_returns': mean_returns,
        'covariance': covariance,
        'volatilities': volatilities,
        'correlation_matrix': correlation_matrix,
    }
    if len(clean_tickers) == 2:
        ticker1, ticker2 = clean_tickers
        market_data.update({
            'r1': float(mean_returns[ticker1]),
            'r2': float(mean_returns[ticker2]),
            'sd1': float(volatilities[ticker1]),
            'sd2': float(volatilities[ticker2]),
            'corr': float(correlation_matrix.loc[ticker1, ticker2]),
        })
    return market_data

@st.cache_data(show_spinner=False)
def fetch_universe_returns(tickers, period):
    clean_tickers = [_normalise_ticker(t) for t in tickers if isinstance(t, str) and t]
    clean_tickers = [t for t in clean_tickers if t and (len(t) <= 8)]
    clean_tickers = list(dict.fromkeys(clean_tickers))[:150]
    if not clean_tickers:
        return pd.DataFrame(columns=['ticker', 'Expected_Return', 'Volatility'])

    source_to_mapped = {}
    for src in clean_tickers:
        mapped = _mapped_ticker(src)
        if mapped:
            source_to_mapped[src] = mapped
    if not source_to_mapped:
        return pd.DataFrame(columns=['ticker', 'Expected_Return', 'Volatility'])

    mapped_download_list = list(dict.fromkeys(source_to_mapped.values()))
    all_rows = []
    chunk_size = 40
    for i in range(0, len(mapped_download_list), chunk_size):
        chunk = mapped_download_list[i:i + chunk_size]
        try:
            data = yf.download(
                chunk,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if data.empty:
                continue
            prices = data['Close'].copy() if 'Close' in data.columns else data.copy()
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=chunk[0])
            prices = prices.dropna(how='all')
            returns = prices.pct_change(fill_method=None).dropna(how='all')
            if returns.empty:
                continue
            mean_returns = returns.mean() * 252
            volatilities = returns.std() * np.sqrt(252)
            mapped_index = pd.Index(mean_returns.index.astype(str).str.upper().str.strip())
            mean_returns.index = mapped_index
            volatilities.index = mapped_index

            rows = []
            for src, mapped in source_to_mapped.items():
                if mapped in mean_returns.index and mapped in volatilities.index:
                    rows.append(
                        {
                            'ticker': src,
                            'Expected_Return': float(mean_returns.loc[mapped]),
                            'Volatility': float(volatilities.loc[mapped]),
                        }
                    )
            if rows:
                all_rows.append(pd.DataFrame(rows))
        except Exception:
            continue
    if not all_rows:
        return pd.DataFrame(columns=['ticker', 'Expected_Return', 'Volatility'])
    out = pd.concat(all_rows, ignore_index=True)
    out['ticker'] = out['ticker'].astype(str).str.upper().str.strip()
    out = out.drop_duplicates(subset=['ticker'], keep='last')
    return out

def find_sustainable_alternatives(base_ticker, base_return, base_esg, candidates_df, top_n=5):
    filtered = candidates_df[(candidates_df['ticker'] != base_ticker) & (candidates_df['Expected_Return'] >= base_return) & (candidates_df['ESG'] >= base_esg)].copy()
    if filtered.empty:
        return filtered
    filtered = filtered.sort_values(['Expected_Return', 'ESG'], ascending=[False, False])
    return filtered.head(top_n)

def _coerce_weight_vector(weights, second_value=None):
    if np.isscalar(weights):
        w1 = float(weights)
        return np.array([w1, 1 - w1], dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    return weights

def _coerce_vector(values, second_value=None):
    if second_value is not None and np.isscalar(values):
        return np.array([float(values), float(second_value)], dtype=float)
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)

def _weight_columns(tickers):
    return [f'Weight_{ticker}' for ticker in tickers]

def get_weight_vector(portfolio_like, tickers):
    if isinstance(portfolio_like, dict):
        weights = portfolio_like.get('Weights')
        if isinstance(weights, dict):
            return np.array([float(weights.get(ticker, 0.0)) for ticker in tickers], dtype=float)
    if isinstance(portfolio_like, pd.Series):
        weights = portfolio_like.get('Weights')
        if isinstance(weights, dict):
            return np.array([float(weights.get(ticker, 0.0)) for ticker in tickers], dtype=float)
    return np.array([float(portfolio_like.get(f'Weight_{ticker}', 0.0)) for ticker in tickers], dtype=float)

def portfolio_return(weights, expected_returns, r2=None):
    w = _coerce_weight_vector(weights)
    mu = _coerce_vector(expected_returns, r2)
    return float(np.dot(w, mu))

def portfolio_variance(weights, covariance_or_sd1, sd2=None, corr=None):
    w = _coerce_weight_vector(weights)
    if corr is not None and sd2 is not None and np.isscalar(covariance_or_sd1):
        sd1 = float(covariance_or_sd1)
        cov = np.array([
            [sd1 ** 2, corr * sd1 * float(sd2)],
            [corr * sd1 * float(sd2), float(sd2) ** 2],
        ], dtype=float)
    else:
        cov = np.asarray(covariance_or_sd1, dtype=float)
    return float(w @ cov @ w)

def portfolio_esg(weights, esg_scores, esg2=None):
    w = _coerce_weight_vector(weights)
    scores = _coerce_vector(esg_scores, esg2)
    total = float(np.sum(w))
    return float(np.dot(w, scores) / total) if abs(total) > 1e-12 else 0.0

def esg_utility_function(port_return_value, port_variance, port_esg_score, gamma, lambda_esg):
    return port_return_value - 0.5 * gamma * port_variance + lambda_esg * port_esg_score

def utility_function(expected_complete_return, complete_variance, gamma):
    return expected_complete_return - 0.5 * gamma * complete_variance

def _build_composition_samples(n_assets, sample_count=8000):
    if n_assets == 2:
        weights = np.linspace(0, 1, 1001)
        return np.column_stack([weights, 1 - weights])
    rng = np.random.default_rng(42)
    sample_count = max(sample_count, n_assets * 600)
    samples = rng.dirichlet(np.ones(n_assets), size=sample_count)
    basis = np.eye(n_assets)
    equal = np.full((1, n_assets), 1.0 / n_assets)
    return np.vstack([basis, equal, samples])


def _portfolio_payload_from_row(row, tickers, use_complete=True):
    payload = row.copy()
    if use_complete:
        weights = {ticker: float(row.get(f'Weight_{ticker}', 0.0)) for ticker in tickers}
        payload['Weights'] = weights
        payload['Expected_Return'] = float(row['Expected_Return'])
        payload['Variance'] = float(row['Variance'])
        payload['Risk_SD'] = float(row['Risk_SD'])
        payload['Sharpe_Ratio'] = float(row['Sharpe_Ratio'])
        payload['ESG_Score'] = float(row['ESG_Score'])
        payload['Utility'] = float(row['Utility'])
        payload['y'] = float(row.get('y', sum(weights.values())))
        payload['weight_risk_free'] = float(row.get('weight_risk_free', 1.0 - payload['y']))
        payload['Risky_Weights'] = {ticker: float(row.get(f'Risky_Weight_{ticker}', 0.0)) for ticker in tickers}
        payload['Risky_Expected_Return'] = float(row.get('Risky_Expected_Return', payload['Expected_Return']))
        payload['Risky_Variance'] = float(row.get('Risky_Variance', payload['Variance']))
        payload['Risky_Risk_SD'] = float(row.get('Risky_Risk_SD', payload['Risk_SD']))
        payload['Risky_ESG_Score'] = float(row.get('Risky_ESG_Score', payload['ESG_Score']))
        payload['Risky_Sharpe_Ratio'] = float(row.get('Risky_Sharpe_Ratio', payload['Sharpe_Ratio']))
    else:
        weights = {ticker: float(row.get(f'Risky_Weight_{ticker}', 0.0)) for ticker in tickers}
        payload['Weights'] = weights
        payload['Expected_Return'] = float(row['Risky_Expected_Return'])
        payload['Variance'] = float(row['Risky_Variance'])
        payload['Risk_SD'] = float(row['Risky_Risk_SD'])
        payload['Sharpe_Ratio'] = float(row['Risky_Sharpe_Ratio'])
        payload['ESG_Score'] = float(row['Risky_ESG_Score'])
        payload['Utility'] = float(row['Utility'])
        payload['y'] = 1.0
        payload['weight_risk_free'] = 0.0
        payload['Risky_Weights'] = weights.copy()
        payload['Risky_Expected_Return'] = payload['Expected_Return']
        payload['Risky_Variance'] = payload['Variance']
        payload['Risky_Risk_SD'] = payload['Risk_SD']
        payload['Risky_ESG_Score'] = payload['ESG_Score']
        payload['Risky_Sharpe_Ratio'] = payload['Sharpe_Ratio']
    for idx, ticker in enumerate(tickers, start=1):
        payload[f'Weight_Asset{idx}'] = float(payload['Weights'].get(ticker, 0.0))
        payload[f'Weight_{ticker}'] = float(payload['Weights'].get(ticker, 0.0))
        payload[f'Risky_Weight_Asset{idx}'] = float(payload['Risky_Weights'].get(ticker, 0.0))
        payload[f'Risky_Weight_{ticker}'] = float(payload['Risky_Weights'].get(ticker, 0.0))
    return payload

def build_portfolio_table(expected_returns=None, covariance_matrix=None, rf=None, esg_scores=None, gamma=None, lambda_esg=None, tickers=None, sample_count=8000, **legacy_kwargs):
    if expected_returns is None:
        expected_returns = [legacy_kwargs['r1'], legacy_kwargs['r2']]
        covariance_matrix = np.array([
            [legacy_kwargs['sd1'] ** 2, legacy_kwargs['corr'] * legacy_kwargs['sd1'] * legacy_kwargs['sd2']],
            [legacy_kwargs['corr'] * legacy_kwargs['sd1'] * legacy_kwargs['sd2'], legacy_kwargs['sd2'] ** 2],
        ], dtype=float)
        esg_scores = [legacy_kwargs['esg1'], legacy_kwargs['esg2']]
        rf = legacy_kwargs['rf']
        gamma = legacy_kwargs['gamma']
        lambda_esg = legacy_kwargs['lambda_esg']
        tickers = legacy_kwargs.get('tickers') or ['Asset1', 'Asset2']
    mu = _coerce_vector(expected_returns)
    cov = np.asarray(covariance_matrix, dtype=float)
    esg_vector = _coerce_vector(esg_scores)
    tickers = list(tickers or [f'Asset{i + 1}' for i in range(len(mu))])
    compositions = _build_composition_samples(len(mu), sample_count=sample_count)
    rows = []
    weight_cols = _weight_columns(tickers)
    mu_excess = mu - float(rf)
    gamma = float(gamma)
    lambda_esg = float(lambda_esg)
    for composition in compositions:
        risky_return = float(np.dot(composition, mu))
        risky_variance = float(composition @ cov @ composition)
        risky_risk = float(np.sqrt(max(risky_variance, 0.0)))
        risky_esg = float(np.dot(composition, esg_vector))
        risky_sharpe = (risky_return - rf) / risky_risk if risky_risk > 0 else np.nan
        if risky_variance <= 0 or gamma <= 0:
            y = 0.0
        else:
            y = max(0.0, float(np.dot(composition, mu_excess)) / (gamma * risky_variance))
        complete_weights = composition * y
        expected_complete_return = float(rf + np.dot(complete_weights, mu_excess))
        complete_variance = float(complete_weights @ cov @ complete_weights)
        complete_risk = float(np.sqrt(max(complete_variance, 0.0)))
        sharpe = (expected_complete_return - rf) / complete_risk if complete_risk > 0 else np.nan
        esg_score = risky_esg if y > 0 else 0.0
        utility = float(np.dot(complete_weights, mu_excess) - 0.5 * gamma * complete_variance + lambda_esg * esg_score)
        row = {
            'Weights': {ticker: float(weight) for ticker, weight in zip(tickers, complete_weights)},
            'Expected_Return': expected_complete_return,
            'Variance': complete_variance,
            'Risk_SD': complete_risk,
            'ESG_Score': esg_score,
            'Sharpe_Ratio': sharpe,
            'Utility': utility,
            'y': float(y),
            'weight_risk_free': float(1.0 - y),
            'Risky_Weights': {ticker: float(weight) for ticker, weight in zip(tickers, composition)},
            'Risky_Expected_Return': risky_return,
            'Risky_Variance': risky_variance,
            'Risky_Risk_SD': risky_risk,
            'Risky_ESG_Score': risky_esg,
            'Risky_Sharpe_Ratio': risky_sharpe,
        }
        for col_name, weight in zip(weight_cols, complete_weights):
            row[col_name] = float(weight)
        for idx, (ticker, weight) in enumerate(zip(tickers, complete_weights), start=1):
            row[f'Weight_Asset{idx}'] = float(weight)
            row[f'Weight_{ticker}'] = float(weight)
        for idx, (ticker, weight) in enumerate(zip(tickers, composition), start=1):
            row[f'Risky_Weight_Asset{idx}'] = float(weight)
            row[f'Risky_Weight_{ticker}'] = float(weight)
        rows.append(row)
    return pd.DataFrame(rows)

def select_recommended_portfolio(df):
    row = df.loc[df['Utility'].idxmax()].copy()
    tickers = [col.replace('Weight_', '') for col in df.columns if col.startswith('Weight_') and not col.startswith('Weight_Asset')]
    return _portfolio_payload_from_row(row, tickers, use_complete=True)

def select_max_sharpe_portfolio(df):
    row = df.loc[df['Risky_Sharpe_Ratio'].idxmax()].copy()
    tickers = [col.replace('Weight_', '') for col in df.columns if col.startswith('Weight_') and not col.startswith('Weight_Asset')]
    return _portfolio_payload_from_row(row, tickers, use_complete=False)

def build_tangency_portfolio(expected_returns, covariance_matrix, rf, tickers=None, esg_scores=None):
    mu = _coerce_vector(expected_returns)
    cov = np.asarray(covariance_matrix, dtype=float)
    tickers = list(tickers or [f'Asset{i + 1}' for i in range(len(mu))])
    excess = mu - float(rf)
    z = np.linalg.pinv(cov) @ excess
    denom = float(np.sum(z))
    if abs(denom) <= 1e-12:
        raise ValueError('Tangency portfolio is undefined because the excess-return weights sum to zero.')
    risky_weights = z / denom
    expected_return = float(np.dot(risky_weights, mu))
    variance = float(risky_weights @ cov @ risky_weights)
    risk = float(np.sqrt(max(variance, 0.0)))
    sharpe = (expected_return - rf) / risk if risk > 0 else np.nan
    esg_score = float(portfolio_esg(risky_weights, esg_scores)) if esg_scores is not None else np.nan
    payload = {
        'Weights': {ticker: float(weight) for ticker, weight in zip(tickers, risky_weights)},
        'Expected_Return': expected_return,
        'Variance': variance,
        'Risk_SD': risk,
        'Sharpe_Ratio': float(sharpe),
        'ESG_Score': esg_score,
        'Utility': np.nan,
        'y': 1.0,
        'weight_risk_free': 0.0,
        'Risky_Weights': {ticker: float(weight) for ticker, weight in zip(tickers, risky_weights)},
        'Risky_Expected_Return': expected_return,
        'Risky_Variance': variance,
        'Risky_Risk_SD': risk,
        'Risky_ESG_Score': esg_score,
        'Risky_Sharpe_Ratio': float(sharpe),
    }
    for idx, (ticker, weight) in enumerate(zip(tickers, risky_weights), start=1):
        payload[f'Weight_Asset{idx}'] = float(weight)
        payload[f'Weight_{ticker}'] = float(weight)
        payload[f'Risky_Weight_Asset{idx}'] = float(weight)
        payload[f'Risky_Weight_{ticker}'] = float(weight)
    return payload

def build_complete_portfolio(tangency_portfolio, rf, gamma, expected_returns=None, covariance_matrix=None, tickers=None, esg_scores=None):
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError('Gamma must be positive to compute the complete tangency portfolio.')
    if expected_returns is None or covariance_matrix is None:
        raise ValueError('Expected returns and covariance matrix are required to compute the complete tangency portfolio.')
    mu = _coerce_vector(expected_returns)
    cov = np.asarray(covariance_matrix, dtype=float)
    tickers = list(tickers or tangency_portfolio.get('Weights', {}).keys() or [f'Asset{i + 1}' for i in range(len(mu))])
    excess = mu - float(rf)
    risky_holdings = (np.linalg.pinv(cov) @ excess) / gamma
    scaled_weights = {ticker: float(weight) for ticker, weight in zip(tickers, risky_holdings)}
    risky_total = float(np.sum(risky_holdings))
    expected_complete_return = float(rf + np.dot(risky_holdings, excess))
    complete_variance = float(risky_holdings @ cov @ risky_holdings)
    complete_risk = float(np.sqrt(max(complete_variance, 0.0)))
    utility = utility_function(expected_complete_return, complete_variance, gamma)
    esg_score = float(portfolio_esg(risky_holdings, esg_scores)) if esg_scores is not None else float(tangency_portfolio.get('ESG_Score', np.nan))
    payload = {
        'y': risky_total,
        'weight_risk_free': float(1.0 - risky_total),
        'Expected_Return': expected_complete_return,
        'Variance': complete_variance,
        'Risk_SD': complete_risk,
        'Utility': utility,
        'Weights': scaled_weights,
        'ESG_Score': esg_score,
        'Sharpe_Ratio': tangency_portfolio['Sharpe_Ratio'],
        'Risky_Weights': {ticker: float(weight) for ticker, weight in tangency_portfolio.get('Weights', {}).items()},
        'Risky_Expected_Return': tangency_portfolio['Expected_Return'],
        'Risky_Variance': tangency_portfolio['Variance'],
        'Risky_Risk_SD': tangency_portfolio['Risk_SD'],
        'Risky_ESG_Score': tangency_portfolio['ESG_Score'],
        'Risky_Sharpe_Ratio': tangency_portfolio['Sharpe_Ratio'],
    }
    for idx, (ticker, weight) in enumerate(scaled_weights.items(), start=1):
        payload[f'Weight_Asset{idx}'] = float(weight)
        payload[f'Weight_{ticker}'] = float(weight)
    return payload

def compute_portfolio_compatibility(gamma, lambda_raw_avg, w_e, w_s, w_g, recommended, esg_data_list=None, weights=None, esg1_data=None, esg2_data=None, w1=None, w2=None):
    import numpy as np
    vol = recommended.get('Risk_SD', 0)
    if vol < 0.12:
        vol_level = 1
    elif vol < 0.22:
        vol_level = 2
    else:
        vol_level = 3
    if gamma >= 6.5:
        preferred_vol = 1
    elif gamma >= 4.5:
        preferred_vol = 2
    else:
        preferred_vol = 3
    risk_compatibility = max(0, 100 - abs(vol_level - preferred_vol) * 40)
    if esg_data_list is None:
        esg_data_list = [esg1_data, esg2_data]
    if weights is None:
        weights = [w1, w2]
    clean_pairs = [(float(weight), data or {}) for weight, data in zip(weights, esg_data_list) if weight is not None]
    blended_esg = sum(weight * float(data.get('ESG', 50)) for weight, data in clean_pairs)
    expected_esg = (lambda_raw_avg - 1) / 3 * 100
    esg_compatibility = max(0, 100 - abs(blended_esg - expected_esg))
    bE = sum(weight * float(data.get('E', 33)) for weight, data in clean_pairs) / 100
    bS = sum(weight * float(data.get('S', 33)) for weight, data in clean_pairs) / 100
    bG = sum(weight * float(data.get('G', 33)) for weight, data in clean_pairs) / 100
    inv = np.array([w_e, w_s, w_g])
    stock = np.array([bE, bS, bG])
    denom = np.linalg.norm(inv) * np.linalg.norm(stock)
    pillar_alignment = float(np.dot(inv, stock) / denom) * 100 if denom > 0 else 50.0
    overall = 0.4 * risk_compatibility + 0.35 * esg_compatibility + 0.25 * pillar_alignment
    return {'risk_compatibility': round(risk_compatibility, 1), 'esg_compatibility': round(esg_compatibility, 1), 'pillar_alignment': round(pillar_alignment, 1), 'overall_compatibility': round(overall, 1)}


def style_axis(ax, x_percent=False, y_percent=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d5dde7')
    ax.spines['bottom'].set_color('#d5dde7')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.set_axisbelow(True)
    ax.set_facecolor(APP_BG)
    ax.grid(False)
    ax.tick_params(colors='#56637a', labelsize=10.5)
    ax.xaxis.label.set_color('#415067')
    ax.yaxis.label.set_color('#415067')
    ax.xaxis.label.set_size(11.5)
    ax.yaxis.label.set_size(11.5)
    ax.title.set_color('#1f2f46')
    ax.title.set_fontweight('700')
    ax.title.set_fontsize(17)
    if x_percent:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))
    if y_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))


def add_end_label(ax, x_value, y_value, label, color, offset=(8, 0), boxed=False):
    text = ax.annotate(
        label,
        (x_value, y_value),
        textcoords='offset points',
        xytext=offset,
        va='center',
        fontsize=10,
        fontweight='600',
        color=color,
        annotation_clip=False,
        zorder=10,
    )
    text.set_clip_on(False)
    text.set_path_effects([pe.withStroke(linewidth=3.2, foreground=APP_BG)])


def _chart_title(ax, title):
    ax.set_title(title, loc='left', pad=12)

def _style_legend(ax, **kwargs):
    ax.legend(frameon=False, fancybox=False, framealpha=0.0, edgecolor='none', facecolor='none', labelcolor='#415067', **kwargs)

def _asset_palette(n_assets):
    cmap = plt.cm.get_cmap('tab10', max(n_assets, 1))
    return [cmap(i) for i in range(n_assets)]

def _efficient_frontier_curve(df):
    x_col = 'Risky_Risk_SD' if 'Risky_Risk_SD' in df.columns else 'Risk_SD'
    y_col = 'Risky_Expected_Return' if 'Risky_Expected_Return' in df.columns else 'Expected_Return'
    curve = df[[x_col, y_col]].dropna().sort_values([x_col, y_col]).rename(columns={x_col: 'Risk_SD', y_col: 'Expected_Return'})
    if curve.empty:
        return curve
    running_max = curve['Expected_Return'].cummax()
    curve = curve[curve['Expected_Return'] >= (running_max - 1e-12)]
    curve = curve.groupby('Risk_SD', as_index=False)['Expected_Return'].max()
    return curve.sort_values('Risk_SD')

def _upper_envelope(curve, x_col, y_col):
    if curve.empty:
        return curve
    points = curve[[x_col, y_col]].dropna().sort_values([x_col, y_col]).to_numpy()
    hull = []
    for point in points:
        while len(hull) >= 2:
            x1, y1 = hull[-2]
            x2, y2 = hull[-1]
            x3, y3 = point
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross >= -1e-12:
                hull.pop()
            else:
                break
        hull.append(point)
    return pd.DataFrame(hull, columns=[x_col, y_col])

def _esg_return_curve(df):
    x_col = 'Risky_ESG_Score' if 'Risky_ESG_Score' in df.columns else 'ESG_Score'
    y_col = 'Risky_Expected_Return' if 'Risky_Expected_Return' in df.columns else 'Expected_Return'
    curve = df[[x_col, y_col]].dropna().sort_values([x_col, y_col]).rename(columns={x_col: 'ESG_Score', y_col: 'Expected_Return'})
    if curve.empty:
        return curve
    curve = curve.groupby('ESG_Score', as_index=False)['Expected_Return'].max()
    curve = curve.sort_values('ESG_Score')
    return _upper_envelope(curve, 'ESG_Score', 'Expected_Return')

def _esg_sharpe_curve(df):
    x_col = 'Risky_ESG_Score' if 'Risky_ESG_Score' in df.columns else 'ESG_Score'
    y_col = 'Risky_Sharpe_Ratio' if 'Risky_Sharpe_Ratio' in df.columns else 'Sharpe_Ratio'
    curve = df[[x_col, y_col]].dropna().sort_values([x_col, y_col]).rename(columns={x_col: 'ESG_Score', y_col: 'Sharpe_Ratio'})
    if curve.empty:
        return curve
    curve = curve.groupby('ESG_Score', as_index=False)['Sharpe_Ratio'].max()
    curve = curve.sort_values('ESG_Score')
    return _upper_envelope(curve, 'ESG_Score', 'Sharpe_Ratio')


def make_frontier_figure(df, tangency, recommended, tickers, mean_returns, volatilities):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(APP_BG)
    frontier_color = '#3a4f73'
    recommended_color = '#1f8a70'
    tangency_color = '#8b5e34'
    curve = _efficient_frontier_curve(df)

    ax.plot(curve['Risk_SD'], curve['Expected_Return'], linewidth=2.8, color=frontier_color)
    ax.scatter(
        tangency['Risk_SD'],
        tangency['Expected_Return'],
        s=118,
        marker='X',
        color=tangency_color,
        linewidths=1.2,
        zorder=4,
    )
    rec_risk = float(recommended.get('Risky_Risk_SD', recommended['Risk_SD']))
    rec_return = float(recommended.get('Risky_Expected_Return', recommended['Expected_Return']))
    ax.scatter(
        rec_risk,
        rec_return,
        s=112,
        marker='o',
        color=recommended_color,
        edgecolors='white',
        linewidths=1.2,
        zorder=4,
    )
    if not curve.empty:
        add_end_label(ax, curve['Risk_SD'].iloc[-1], curve['Expected_Return'].iloc[-1], 'Frontier', frontier_color, offset=(9, -10))
    add_end_label(ax, tangency['Risk_SD'], tangency['Expected_Return'], 'Tangency', tangency_color, offset=(-42, -18))
    info = ax.annotate(
        f'Recommended ESG Portfolio\nExp. Return: {recommended["Expected_Return"]*100:.1f}%\nVolatility: {recommended["Risk_SD"]*100:.1f}%',
        (rec_risk, rec_return),
        textcoords='offset points',
        xytext=(20, 8),
        ha='left',
        va='center',
        fontsize=10,
        color='#1f2f46',
        annotation_clip=False,
        zorder=11,
    )
    info.set_clip_on(False)
    info.set_path_effects([pe.withStroke(linewidth=3.8, foreground=APP_BG)])
    ax.set_xlabel('Portfolio Volatility')
    ax.set_ylabel('Expected Return')
    _chart_title(ax, 'Efficient Frontier & ESG-Efficient Portfolio')
    style_axis(ax, x_percent=True, y_percent=True)
    fig.tight_layout()
    return fig


def make_cml_figure(df, tangency, complete_portfolio, rf, tickers, mean_returns, volatilities):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(APP_BG)
    frontier_color = '#3a4f73'
    cml_color = '#5f9c84'
    tangency_color = '#8b5e34'
    complete_color = '#7c3f8c'
    risk_free_color = '#7a8da8'
    curve = _efficient_frontier_curve(df)

    frontier_max_risk = float(curve['Risk_SD'].max()) if not curve.empty else 0.0
    cml_risk = np.linspace(0, max(frontier_max_risk, complete_portfolio['Risk_SD']) * 1.15, 300)
    cml_return = rf + tangency['Sharpe_Ratio'] * cml_risk
    ax.plot(curve['Risk_SD'], curve['Expected_Return'], linewidth=2.8, color=frontier_color, label='Efficient frontier')
    ax.plot(cml_risk, cml_return, linewidth=2.0, color=cml_color, linestyle='--', dashes=(4, 3), label='Capital market line')
    ax.scatter(tangency['Risk_SD'], tangency['Expected_Return'], s=110, marker='X', color=tangency_color, zorder=4)
    ax.scatter(complete_portfolio['Risk_SD'], complete_portfolio['Expected_Return'], s=105, marker='o', color=complete_color, edgecolors='white', linewidths=1.0, zorder=4)
    ax.scatter(0, rf, s=75, color=risk_free_color, zorder=4)

    if not curve.empty:
        add_end_label(ax, curve['Risk_SD'].iloc[-1], curve['Expected_Return'].iloc[-1], 'Frontier', frontier_color, offset=(9, -10))
    add_end_label(ax, cml_risk[-1], cml_return[-1], 'CML', cml_color, offset=(8, -12))
    add_end_label(ax, tangency['Risk_SD'], tangency['Expected_Return'], 'Tangency', tangency_color, offset=(10, 12))
    add_end_label(ax, complete_portfolio['Risk_SD'], complete_portfolio['Expected_Return'], 'Complete', complete_color, offset=(10, -14))
    rf_label = ax.annotate(
        'Risk-free',
        (0, rf),
        textcoords='offset points',
        xytext=(8, 8),
        fontsize=9.8,
        color='#5b6578',
        annotation_clip=False,
        zorder=10,
    )
    rf_label.set_clip_on(False)
    rf_label.set_path_effects([pe.withStroke(linewidth=3.2, foreground=APP_BG)])
    ax.set_xlabel('Portfolio Volatility')
    ax.set_ylabel('Expected Return')
    _chart_title(ax, 'Efficient Frontier & Capital Market Line')
    style_axis(ax, x_percent=True, y_percent=True)
    fig.tight_layout()
    return fig


def make_esg_tradeoff_figure(df, recommended):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(APP_BG)
    tradeoff_color = '#2e8b57'
    recommended_color = '#0f766e'
    curve = _esg_return_curve(df)

    ax.plot(curve['ESG_Score'], curve['Expected_Return'], linewidth=2.8, color=tradeoff_color)
    rec_esg = float(recommended.get('Risky_ESG_Score', recommended['ESG_Score']))
    rec_return = float(recommended.get('Risky_Expected_Return', recommended['Expected_Return']))
    ax.scatter(rec_esg, rec_return, s=112, marker='o', color=recommended_color, edgecolors='white', linewidths=1.0, zorder=4)
    if not curve.empty:
        add_end_label(ax, curve['ESG_Score'].iloc[-1], curve['Expected_Return'].iloc[-1], 'Trade-off', tradeoff_color, offset=(10, -6))
    add_end_label(ax, rec_esg, rec_return, 'Recommended ESG Portfolio', recommended_color, offset=(12, -20), boxed=False)
    ax.set_xlabel('Portfolio ESG Score')
    ax.set_ylabel('Expected Return')
    _chart_title(ax, 'ESG Frontier & Recommended ESG Portfolio')
    style_axis(ax, y_percent=True)
    fig.tight_layout()
    return fig


def make_esg_efficient_frontier_figure(df, tangency, recommended):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(APP_BG)
    curve = _esg_sharpe_curve(df)
    frontier_color = '#3a4f73'
    tangency_color = '#8b5e34'
    recommended_color = '#0f766e'
    guide_color = '#f2c94c'

    ax.plot(curve['ESG_Score'], curve['Sharpe_Ratio'], linewidth=2.8, color=frontier_color)
    ax.scatter(tangency['ESG_Score'], tangency['Sharpe_Ratio'], s=110, marker='X', color=tangency_color, zorder=4)
    rec_esg = float(recommended.get('Risky_ESG_Score', recommended['ESG_Score']))
    rec_sharpe = float(recommended.get('Risky_Sharpe_Ratio', recommended['Sharpe_Ratio']))
    ax.scatter(rec_esg, rec_sharpe, s=110, marker='o', color=recommended_color, edgecolors='white', linewidths=1.1, zorder=4)
    x_left = min(tangency['ESG_Score'], rec_esg)
    x_right = max(tangency['ESG_Score'], rec_esg)
    y_bottom = min(tangency['Sharpe_Ratio'], rec_sharpe)
    y_top = max(tangency['Sharpe_Ratio'], rec_sharpe)
    ax.hlines(
        tangency['Sharpe_Ratio'],
        xmin=x_left,
        xmax=x_right,
        colors=guide_color,
        linewidth=3.0,
        alpha=0.98,
    )
    ax.vlines(
        rec_esg,
        ymin=y_bottom,
        ymax=y_top,
        colors=guide_color,
        linewidth=3.0,
        alpha=0.98,
    )

    add_end_label(ax, tangency['ESG_Score'], tangency['Sharpe_Ratio'], 'Tangency', tangency_color, offset=(14, -2))
    add_end_label(ax, rec_esg, rec_sharpe, 'Recommended ESG Portfolio', recommended_color, offset=(14, -20))

    sharpe_gap = tangency['Sharpe_Ratio'] - rec_sharpe
    esg_gain = rec_esg - tangency['ESG_Score']
    cost_label = ax.annotate(
        f'ESG preference cost: {sharpe_gap:+.3f} SR\nESG improvement: {esg_gain:+.2f}',
        (recommended['ESG_Score'], tangency['Sharpe_Ratio']),
        textcoords='offset points',
        xytext=(18, 10),
        fontsize=9.6,
        color='#49606e',
        annotation_clip=False,
        zorder=11,
    )
    cost_label.set_clip_on(False)
    cost_label.set_path_effects([pe.withStroke(linewidth=3.2, foreground=APP_BG)])
    ax.set_xlabel('Portfolio ESG Score')
    ax.set_ylabel('Sharpe Ratio')
    _chart_title(ax, 'ESG-Efficient Frontier')
    style_axis(ax)
    fig.tight_layout()
    return fig


def make_price_history_figure(prices, tickers):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(APP_BG)
    palette = _asset_palette(len(tickers))
    cumulative_returns = prices[tickers].div(prices[tickers].iloc[0]).sub(1.0)
    end_values = [(ticker, float(cumulative_returns[ticker].iloc[-1])) for ticker in tickers]
    ranked = {ticker: rank for rank, (ticker, _) in enumerate(sorted(end_values, key=lambda item: item[1], reverse=True))}
    spacing_pattern = [-14, 14, -28, 28, -42, 42, -56, 56]
    for idx, ticker in enumerate(tickers):
        color = palette[idx]
        ax.plot(cumulative_returns.index, cumulative_returns[ticker], linewidth=2.6 if idx < 2 else 2.0, color=color, label=ticker)
        rank = ranked.get(ticker, idx)
        y_offset = spacing_pattern[rank] if rank < len(spacing_pattern) else ((rank + 1) * 12)
        add_end_label(ax, cumulative_returns.index[-1], cumulative_returns[ticker].iloc[-1], ticker, color, offset=(8, y_offset))
    _chart_title(ax, 'Historical Return Performance')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    style_axis(ax, y_percent=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def make_esg_radar_figure(asset_esg_data, w_e, w_s, w_g):
    categories = ['Environmental (E)', 'Social (S)', 'Governance (G)']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLORS['panel'])
    ax.set_facecolor('#f9fbfd')

    def _get_vals(esg_data):
        if esg_data is None:
            return [0.5, 0.5, 0.5]
        return [
            min(max(float(esg_data.get('E') or 50) / 100.0, 0), 1),
            min(max(float(esg_data.get('S') or 50) / 100.0, 0), 1),
            min(max(float(esg_data.get('G') or 50) / 100.0, 0), 1),
        ]

    inv_w = [w_e, w_s, w_g, w_e]
    palette = _asset_palette(len(asset_esg_data))
    for idx, (ticker, esg_data) in enumerate(asset_esg_data):
        vals = _get_vals(esg_data) + _get_vals(esg_data)[:1]
        ax.plot(angles, vals, color=palette[idx], linewidth=2.4 if idx < 2 else 1.8, label=ticker)
        ax.fill(angles, vals, color=palette[idx], alpha=0.08 if idx < 2 else 0.04)
    ax.plot(angles, inv_w, color=COLORS['accent'], linewidth=2.0, linestyle='--', label='Your ESG priorities')
    ax.fill(angles, inv_w, color=COLORS['accent'], alpha=0.07)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10.5, color='#56637a')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=9, color='#94a3b8')
    ax.grid(False)
    ax.spines['polar'].set_color(COLORS['line_soft'])
    ax.set_title('ESG Dimension Radar', size=16, color='#1f2f46', pad=20, fontweight='700')
    _style_legend(ax, loc='upper right', bbox_to_anchor=(1.38, 1.15))
    fig.tight_layout()
    return fig

