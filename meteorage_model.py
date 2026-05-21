"""
Pipeline Météorage - Weibull AFT pour prédiction de fin d'alerte orage.

Module reutilisable :
  - feature engineering (17 features de rythme/distance/intensite/saisonnalite)
  - fit du Weibull AFT general avec dummies aeroport
  - calibration empirique robuste (winsorize + monotonicite)
  - prediction de T_q(X) calibre sur de nouvelles donnees

Utilise par evaluate.py (batch) et app.py (Streamlit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler


FEATURES = [
    'cg_rank', 'time_since_start', 'prev_gap',
    'rolling_gap_3', 'rolling_gap_5', 'trend_gap',
    'dist_current', 'dist_cum_mean', 'dist_diff',
    'amp_abs', 'amp_cum_mean', 'is_negative', 'pct_neg_cum',
    'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
]

Q_LEVELS = [0.90, 0.95, 0.99]


def load_train(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    alerts = df.dropna(subset=['airport_alert_id']).copy()
    alerts['airport_alert_id'] = alerts['airport_alert_id'].astype(int).astype(str)
    alerts['airport_alert_id'] = alerts['airport'] + '_' + alerts['airport_alert_id']
    alerts['year'] = alerts['date'].dt.year
    mask_pise_2016 = (alerts['airport'] == 'Pise') & (alerts['year'] == 2016)
    alerts = alerts.loc[~mask_pise_2016].copy()
    cg = alerts.loc[~alerts['icloud']].copy()
    cg = cg.sort_values(['airport', 'airport_alert_id', 'date']).reset_index(drop=True)
    return cg


def load_eval(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.rename(columns={
        'alert_id': 'airport_alert_id',
        'is_last_lightning': 'is_last_lightning_cloud_ground',
    })
    alerts = df.dropna(subset=['airport_alert_id']).copy()
    alerts['year'] = alerts['date'].dt.year
    cg = alerts.loc[~alerts['icloud']].copy()
    cg = cg.sort_values(['airport', 'airport_alert_id', 'date']).reset_index(drop=True)
    return cg


def add_features(cg: pd.DataFrame) -> pd.DataFrame:
    cg = cg.copy()
    cg['month'] = cg['date'].dt.month
    cg['hour'] = cg['date'].dt.hour
    cg['month_sin'] = np.sin(2 * np.pi * cg['month'] / 12)
    cg['month_cos'] = np.cos(2 * np.pi * cg['month'] / 12)
    cg['hour_sin'] = np.sin(2 * np.pi * cg['hour'] / 24)
    cg['hour_cos'] = np.cos(2 * np.pi * cg['hour'] / 24)

    grp = cg.groupby(['airport', 'airport_alert_id'], sort=False)

    cg['time_since_start'] = (cg['date'] - grp['date'].transform('first')).dt.total_seconds() / 60
    cg['prev_gap'] = grp['date'].diff().dt.total_seconds().div(60).fillna(0)
    cg['cg_rank'] = grp.cumcount() + 1
    cg['n_cg_alert'] = grp['cg_rank'].transform('max')
    cg['rolling_gap_3'] = grp['prev_gap'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    cg['rolling_gap_5'] = grp['prev_gap'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    cg['trend_gap'] = (cg['prev_gap'] / cg['rolling_gap_5'].replace(0, np.nan)).fillna(1)

    cg['dist_current'] = cg['dist']
    cg['dist_cum_mean'] = grp['dist'].transform(lambda x: x.expanding().mean())
    cg['dist_diff'] = grp['dist'].diff().fillna(0)

    cg['amp_abs'] = cg['amplitude'].abs()
    cg['amp_cum_mean'] = grp['amp_abs'].transform(lambda x: x.expanding().mean())
    cg['is_negative'] = (cg['amplitude'] < 0).astype(int)
    cg['pct_neg_cum'] = grp['is_negative'].transform(lambda x: x.expanding().mean())
    return cg


def add_target(cg: pd.DataFrame, is_last_col: str) -> pd.DataFrame:
    """Calcule duration et event pour la survie. is_last_col distingue train/eval."""
    cg = cg.copy()
    grp = cg.groupby(['airport', 'airport_alert_id'], sort=False)
    cg['target'] = grp['time_since_start'].shift(-1) - cg['time_since_start']
    cg['is_last'] = cg['cg_rank'] == cg['n_cg_alert']
    cg['duration'] = cg['target'].copy()
    cg['event'] = (~cg['is_last']).astype(int)
    cg.loc[cg['is_last'], 'duration'] = 30.0
    return cg


def build_model_matrix(cg: pd.DataFrame, scaler: StandardScaler, apt_cols: list[str],
                       fit_scaler: bool = False):
    """Standardise les features et ajoute les dummies aeroport."""
    model_df = cg[cg['n_cg_alert'] >= 3].copy()
    model_df = model_df[model_df['duration'] > 0].copy()

    for col in FEATURES:
        model_df[col] = model_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    if fit_scaler:
        scaler.fit(model_df[FEATURES])
    model_df_sc = model_df.copy()
    model_df_sc[FEATURES] = scaler.transform(model_df[FEATURES])

    apt_dum = pd.get_dummies(model_df_sc['airport'], prefix='apt', drop_first=True).astype(float)
    if fit_scaler:
        apt_cols = list(apt_dum.columns)
    else:
        for col in apt_cols:
            if col not in apt_dum.columns:
                apt_dum[col] = 0.0
        apt_dum = apt_dum[apt_cols]

    fit_df = pd.concat([
        model_df_sc[FEATURES + ['duration', 'event']].reset_index(drop=True),
        apt_dum.reset_index(drop=True),
    ], axis=1)
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    fit_df = fit_df[fit_df['duration'] > 0]
    return model_df_sc.reset_index(drop=True), fit_df, apt_cols


def fit_weibull(train_fit: pd.DataFrame, penalizer: float = 0.05) -> WeibullAFTFitter:
    aft = WeibullAFTFitter(penalizer=penalizer, l1_ratio=0.0)
    aft.fit(train_fit, duration_col='duration', event_col='event')
    return aft


def compute_robust_calibration(aft: WeibullAFTFitter, fit_df: pd.DataFrame,
                                q_levels=Q_LEVELS) -> dict:
    """Winsorisation P95 + enforcement de monotonicite c_q."""
    evt = fit_df[fit_df['event'] == 1].copy()
    y_real = evt['duration'].values

    c_q_winsorized = {}
    for q in q_levels:
        Tq = aft.predict_percentile(evt, p=1.0 - q).values
        residuals = y_real / np.where(Tq > 1e-9, Tq, 1e-9)
        p95 = np.quantile(residuals, 0.95)
        residuals_wins = np.minimum(residuals, p95)
        c_q_winsorized[q] = float(np.quantile(residuals_wins, q))

    scaling = {}
    prev_c = 0.0
    for q in sorted(q_levels):
        c_q_mono = max(c_q_winsorized[q], prev_c)
        scaling[q] = c_q_mono
        prev_c = c_q_mono
    return scaling


def predict_T_q_calibrated(aft: WeibullAFTFitter, X: pd.DataFrame,
                            scaling: dict, q: float) -> np.ndarray:
    """Predict T_q calibre pour un lot d'observations."""
    Tq_raw = aft.predict_percentile(X, p=1.0 - q).values
    return scaling[q] * Tq_raw


def empirical_risk_table(cg: pd.DataFrame, T_grid=np.arange(1, 31)) -> pd.DataFrame:
    """Table de risque empirique non-parametrique : pour chaque T, P(G > T)."""
    max_gap = (
        cg.dropna(subset=['prev_gap'])
        .loc[cg['prev_gap'] > 0]
        .groupby(['airport', 'airport_alert_id'])['prev_gap']
        .max()
    )
    gaps = max_gap.values
    rows = []
    for T in T_grid:
        rows.append({'T': int(T), 'risk': (gaps > T).mean()})
    return pd.DataFrame(rows), gaps


def evaluate_calibration(aft: WeibullAFTFitter, fit_df: pd.DataFrame, scaling: dict,
                          q_levels=Q_LEVELS) -> pd.DataFrame:
    """Couverture empirique avant/apres calibration sur les obs avec event=1."""
    evt = fit_df[fit_df['event'] == 1].copy()
    y_real = evt['duration'].values
    rows = []
    for q in q_levels:
        Tq_raw = aft.predict_percentile(evt, p=1.0 - q).values
        Tq_cal = scaling[q] * Tq_raw
        rows.append({
            'q_nominal': q,
            'couverture_brute': float((y_real <= Tq_raw).mean()),
            'couverture_calibree': float((y_real <= Tq_cal).mean()),
            'cible': q,
        })
    return pd.DataFrame(rows)


def compute_cindex(aft: WeibullAFTFitter, fit_df: pd.DataFrame) -> float:
    return concordance_index(fit_df['duration'], aft.predict_median(fit_df), fit_df['event'])
