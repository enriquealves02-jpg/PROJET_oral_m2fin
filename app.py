"""
Demo Streamlit — Weibull AFT + simulation operationnelle.

PRINCIPE :
A chaque CG observe en zone 20 km, le modele Weibull AFT predit T_q(X) minutes.
On utilise un timer effectif = max(T_q, T_min) ou T_min est un plancher operationnel
(ne jamais lever avant T_min minutes apres un CG, peu importe ce que dit le modele).

Si timer expire sans nouveau CG  --> ON LEVE l'alerte.
Si nouveau CG arrive avant timer --> RESET au nouveau CG.

INCIDENT (= critere officiel) : un CG <3 km de l'aeroport arrive APRES notre levee predite.

L'app permet de tuner (q, T_min) et de voir le compromis gain/risque + breakdown par aeroport.

Lancer : streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

import meteorage_model as mm

ROOT = Path(__file__).parent
TRAIN_CSV = ROOT / 'data(1)' / 'data' / 'segment_alerts_all_airports_train.csv'
EVAL_CSV = ROOT / 'segment_alerts_all_airports_eval.csv'

st.set_page_config(page_title='Battle Météorage — Weibull AFT', layout='wide')


# ---------------------------------------------------------
# Chargement + fit modele (cache)
# ---------------------------------------------------------
@st.cache_resource
def fit_model():
    cg_train = mm.load_train(str(TRAIN_CSV))
    cg_train = mm.add_features(cg_train)
    cg_train = mm.add_target(cg_train, is_last_col='is_last_lightning_cloud_ground')
    scaler = StandardScaler()
    _, train_fit, apt_cols = mm.build_model_matrix(cg_train, scaler, [], fit_scaler=True)
    aft = mm.fit_weibull(train_fit, penalizer=0.05)
    scaling = mm.compute_robust_calibration(aft, train_fit)
    return aft, scaler, apt_cols, scaling


@st.cache_data
def load_eval():
    cg_eval = mm.load_eval(str(EVAL_CSV))
    cg_eval = mm.add_features(cg_eval)
    cg_eval = mm.add_target(cg_eval, is_last_col='is_last_lightning')
    cg_eval = cg_eval.sort_values(['airport', 'airport_alert_id', 'date']).reset_index(drop=True)
    cg_eval['date_utc'] = pd.to_datetime(cg_eval['date'], utc=True)

    df_raw = pd.read_csv(EVAL_CSV).rename(columns={'alert_id': 'airport_alert_id'})
    df_raw['date_utc'] = pd.to_datetime(df_raw['date'], utc=True)
    return cg_eval, df_raw


@st.cache_data
def predict_Tq_all_cg(q: float):
    """Calcule T_q calibre (cape 30 min) pour CHAQUE CG de l'eval (pas juste le dernier)."""
    aft, scaler, apt_cols, scaling = fit_model()
    cg_eval, _ = load_eval()
    cg = cg_eval.copy().reset_index(drop=True)

    feat = cg[mm.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_sc = pd.DataFrame(scaler.transform(feat), columns=mm.FEATURES)
    apt_dum = pd.get_dummies(cg['airport'], prefix='apt', drop_first=True).astype(float)
    for col in apt_cols:
        if col not in apt_dum.columns:
            apt_dum[col] = 0.0
    apt_dum = apt_dum[apt_cols].reset_index(drop=True)
    X = pd.concat([X_sc.reset_index(drop=True), apt_dum], axis=1)

    Tq_raw = aft.predict_percentile(X, p=1.0 - q).values
    qs = sorted(scaling.keys())
    c_q = float(np.interp(q, qs, [scaling[k] for k in qs]))
    cg['T_q'] = np.clip(c_q * Tq_raw, 0, 30)
    return cg


# ---------------------------------------------------------
# Simulation operationnelle d'une alerte
# ---------------------------------------------------------
def simulate_alert(cgs_alert: pd.DataFrame, lightnings_alert: pd.DataFrame,
                   T_min: float) -> dict:
    """Simule la mecanique du timer sur une alerte.

    A chaque CG, timer = max(T_q, T_min). Si timer expire avant prochain CG : LIFT.
    Sinon : reset. Au dernier CG, on leve a t_n + timer.
    """
    cgs = cgs_alert.reset_index(drop=True)
    n = len(cgs)

    lift_time = None
    lift_at_cg_rank = None
    timer_used = None

    for k in range(n):
        cg_k = cgs.iloc[k]
        timer_min = max(float(cg_k['T_q']), T_min)
        timer_end = cg_k['date_utc'] + pd.Timedelta(minutes=timer_min)

        if k < n - 1:
            next_cg_time = cgs.iloc[k + 1]['date_utc']
            if timer_end <= next_cg_time:
                lift_time = timer_end
                lift_at_cg_rank = k + 1
                timer_used = timer_min
                break
        else:
            lift_time = timer_end
            lift_at_cg_rank = k + 1
            timer_used = timer_min

    last_cg_date = cgs.iloc[-1]['date_utc']
    baseline_end = last_cg_date + pd.Timedelta(minutes=30)

    if len(lightnings_alert) > 0:
        missed_cg = int(((lightnings_alert['dist'] < 3)
                         & (~lightnings_alert['icloud'])
                         & (lightnings_alert['date_utc'] > lift_time)).sum())
    else:
        missed_cg = 0

    gain_min = (baseline_end - lift_time).total_seconds() / 60.0

    return {
        'lift_time': lift_time,
        'lift_at_cg_rank': lift_at_cg_rank,
        'timer_used': timer_used,
        'n_cgs_in_alert': n,
        'last_cg_date': last_cg_date,
        'baseline_end': baseline_end,
        'missed_cg': missed_cg,
        'gain_min': gain_min,
    }


@st.cache_data
def run_simulation(q: float, T_min: float):
    """Boucle sur toutes les alertes et agrege."""
    cg = predict_Tq_all_cg(q)
    _, df_raw = load_eval()
    lightnings_grp = df_raw.groupby(['airport', 'airport_alert_id'])

    rows = []
    for (apt, aid), cgs_alert in cg.groupby(['airport', 'airport_alert_id']):
        try:
            lightnings_alert = lightnings_grp.get_group((apt, aid))
        except KeyError:
            lightnings_alert = pd.DataFrame()
        res = simulate_alert(cgs_alert, lightnings_alert, T_min)
        res['airport'] = apt
        res['airport_alert_id'] = aid
        rows.append(res)

    return pd.DataFrame(rows)


@st.cache_data
def sweep_Tmin(q: float, T_mins: tuple):
    """Sweep sur T_min pour le q fixe, retourne table de synthese."""
    cg = predict_Tq_all_cg(q)
    _, df_raw = load_eval()
    lightnings_grp = df_raw.groupby(['airport', 'airport_alert_id'])
    total_cg_3km = int(((df_raw['dist'] < 3) & (~df_raw['icloud'])).sum())

    rows = []
    for T_min in T_mins:
        gain_total_min = 0
        missed_cg_total = 0
        timer_sum = 0
        n = 0
        for (apt, aid), cgs_alert in cg.groupby(['airport', 'airport_alert_id']):
            try:
                lightnings_alert = lightnings_grp.get_group((apt, aid))
            except KeyError:
                lightnings_alert = pd.DataFrame()
            res = simulate_alert(cgs_alert, lightnings_alert, T_min)
            gain_total_min += res['gain_min']
            missed_cg_total += res['missed_cg']
            timer_sum += res['timer_used']
            n += 1
        rows.append({
            'T_min (min)': int(T_min),
            'timer moyen (min)': round(timer_sum / n, 1),
            'gain total (h)': round(gain_total_min / 60, 0),
            'gain moyen / alerte (min)': round(gain_total_min / n, 1),
            'CG manqués / 385': f'{missed_cg_total}',
            'risque CG (%)': round(missed_cg_total / total_cg_3km * 100, 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("Battle Météorage — Weibull AFT, simulation opérationnelle")

with st.expander("Principe de la simulation opérationnelle", expanded=True):
    st.markdown(r"""
**Mécanique réelle (= ce qui se passe en production)** :

À chaque CG observé en zone 20 km autour d'un aéroport :

1. Le modèle Weibull AFT prédit un temps d'attente $T_q(X)$ minutes (adapté au contexte).
2. On utilise un **timer effectif** = $\max(T_q, T_{\min})$ où $T_{\min}$ est un **plancher
   opérationnel** (jamais lever avant $T_{\min}$ min, peu importe ce que dit le modèle).
3. Si un **nouveau CG arrive avant** que le timer expire → **RESET** au nouveau CG.
4. Si le **timer expire sans nouveau CG** → **LEVER** l'alerte.

**Définition d'un incident** : un CG <3 km de l'aéroport arrivant APRÈS notre levée prédite.
On compte le nombre total de CG <3 km manqués sur les 385 CG dangereux de l'eval (973 alertes).

**Deux leviers** :
- $q$ (quantile) : calibre l'agressivité de $T_q$. $q$ proche de 1 → $T_q$ long → plus prudent.
- $T_{\min}$ (plancher) : seuil de sécurité que l'opérateur impose, indépendant du modèle.
  - $T_{\min} = 0$ : on suit pleinement le modèle (risque max, gain max).
  - $T_{\min} = 30$ : on attend toujours au moins 30 min après chaque CG (≈ baseline Météorage).
""")

# Sidebar
st.sidebar.title('Réglages')
q = st.sidebar.select_slider('q (quantile T_q)',
                              options=[0.85, 0.90, 0.95, 0.97, 0.99], value=0.95)
T_min = st.sidebar.slider('T_min (plancher, min)', 0, 30, 25, step=5)
st.sidebar.markdown(f'**Configuration courante** : q = {q}, T_min = {T_min} min')
st.sidebar.caption(
    "**Recommandé** : q=0,95 et T_min=25 → ~17% de risque CG / ~250 h de gain.\n\n"
    "**Conservateur** : T_min=30 → ~8% / ~135 h.\n\n"
    "**Agressif** : T_min=10 → ~50% / ~670 h."
)

if 'launched' not in st.session_state:
    st.session_state.launched = False

st.markdown('---')
col_btn, col_status = st.columns([1, 3])
with col_btn:
    if st.button("Lancer la simulation", type='primary', width='stretch'):
        st.session_state.launched = True
with col_status:
    if st.session_state.launched:
        st.caption("Simulation active. Modifie q ou T_min à gauche pour explorer le compromis.")
    else:
        st.caption("Clique pour simuler le modèle sur les 973 alertes de l'eval.")

if st.session_state.launched:
    with st.spinner(f"Simulation à q={q}, T_min={T_min}..."):
        results = run_simulation(q, T_min)

    _, df_raw = load_eval()
    total_cg_3km = int(((df_raw['dist'] < 3) & (~df_raw['icloud'])).sum())
    n_alerts = len(results)
    missed_cg = int(results['missed_cg'].sum())
    gain_total_h = results['gain_min'].sum() / 60
    gain_avg_min = results['gain_min'].mean()
    timer_avg = results['timer_used'].mean()
    risque_cg_pct = missed_cg / total_cg_3km * 100

    # ==========================================================
    # KPI globaux
    # ==========================================================
    st.subheader(f"Résultats globaux à q = {q}, T_min = {T_min} min")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Alertes simulées', f"{n_alerts:,}",
              help="Toutes les alertes de l'eval 2023-25.")
    c2.metric('Timer effectif moyen', f"{timer_avg:.1f} min",
              help="Moyenne de max(T_q, T_min) sur les CG où on a levé.")
    c3.metric('Gain opérationnel cumulé', f"{gain_total_h:.0f} h",
              help=f"Somme sur {n_alerts} alertes de (baseline_30min − levée_prédite).")
    c4.metric('Gain moyen / alerte', f"{gain_avg_min:.1f} min",
              help="Économie moyenne par alerte vs règle Météorage 30 min.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('CG <3 km manqués', f"{missed_cg} / {total_cg_3km}",
              help="CG nuage-sol à <3 km de l'aéroport arrivant APRÈS notre levée prédite.")
    c2.metric('Risque CG <3 km', f"{risque_cg_pct:.2f} %",
              help=f"= {missed_cg} / {total_cg_3km}. Cible protocole : R < 2 %.")
    c3.metric('Cible protocole', "R < 2 %",
              help="Seuil acceptable défini par le protocole officiel Data Battle.")
    n_early = (results['lift_at_cg_rank'] < results['n_cgs_in_alert']).sum()
    c4.metric('Levées prématurées', f"{n_early} / {n_alerts}",
              help="Alertes où le timer a expiré avant le vrai dernier CG (= lift en milieu d'orage).")

    if risque_cg_pct <= 2.0:
        st.success(f"Risque {risque_cg_pct:.2f} % sous la cible 2 % — {gain_total_h:.0f} h gagnées.")
    elif risque_cg_pct <= 10.0:
        st.info(f"Risque modéré ({risque_cg_pct:.2f} %). Augmente T_min pour être plus prudent.")
    else:
        st.warning(f"Risque élevé ({risque_cg_pct:.2f} %). Augmente T_min ou q pour réduire.")

    # ==========================================================
    # Sweep T_min
    # ==========================================================
    st.markdown('---')
    st.subheader(f"Compromis T_min à q = {q}")
    st.caption("Le plancher T_min est le levier opérationnel : plus T_min est haut, "
               "moins on prend de risque mais moins on gagne de temps.")
    sweep_df = sweep_Tmin(q, (0, 5, 10, 15, 20, 25, 30))
    # Highlight la ligne courante
    def highlight_current(row):
        return ['background-color: #d4edda' if row['T_min (min)'] == T_min else ''] * len(row)
    st.dataframe(sweep_df.style.apply(highlight_current, axis=1),
                 hide_index=True, width='stretch')

    # ==========================================================
    # Synthese par aeroport
    # ==========================================================
    st.markdown('---')
    st.subheader(f"Détail par aéroport à q = {q}, T_min = {T_min}")

    cg_3km_per_apt = (df_raw[(df_raw['dist'] < 3) & (~df_raw['icloud'])]
                      .groupby('airport').size().rename('cg_3km_total'))
    by_apt = results.groupby('airport').agg(
        nb_alertes=('airport_alert_id', 'count'),
        timer_moyen_min=('timer_used', 'mean'),
        gain_moyen_par_alerte_min=('gain_min', 'mean'),
        gain_total_h=('gain_min', lambda s: s.sum() / 60),
        missed_cg=('missed_cg', 'sum'),
    ).round(2)
    by_apt = by_apt.join(cg_3km_per_apt)
    by_apt['risque_cg_pct'] = (by_apt['missed_cg'] / by_apt['cg_3km_total'] * 100).round(2)
    st.dataframe(by_apt, width='stretch')
    st.caption("Bastia et Pise ont les risques les plus élevés (orages longs et denses). "
               "Ajaccio et Nantes ont des profils plus calmes.")

    # ==========================================================
    # Visualisation d'une alerte avec incidents
    # ==========================================================
    incidents = results[results['missed_cg'] > 0].copy().sort_values('missed_cg', ascending=False)
    if len(incidents) > 0:
        st.markdown('---')
        st.subheader(f"Visualisation : {len(incidents)} alertes avec incidents")
        st.caption("Sélectionne une alerte pour voir sa timeline graphique.")

        labels = [
            f"{r['airport']} / {r['airport_alert_id']} — "
            f"{r['missed_cg']} CG manqués, timer {r['timer_used']:.1f} min"
            for _, r in incidents.head(50).iterrows()
        ]
        chosen = st.selectbox(f"Top 50 incidents (sur {len(incidents)})", labels)
        sel = incidents.iloc[labels.index(chosen)]

        cg = predict_Tq_all_cg(q)
        cgs_alert = cg[(cg['airport'] == sel['airport'])
                       & (cg['airport_alert_id'] == sel['airport_alert_id'])].sort_values('date_utc')
        sub = df_raw[(df_raw['airport'] == sel['airport'])
                     & (df_raw['airport_alert_id'] == sel['airport_alert_id'])].sort_values('date_utc')
        ref = cgs_alert['date_utc'].iloc[0]
        sub_plot = sub.copy()
        sub_plot['t_min'] = (sub_plot['date_utc'] - ref).dt.total_seconds() / 60
        lift_min = (sel['lift_time'] - ref).total_seconds() / 60
        baseline_min = (sel['baseline_end'] - ref).total_seconds() / 60

        fig, ax = plt.subplots(figsize=(13, 5.5))
        ax.axhspan(0, 3, color='red', alpha=0.06, zorder=0)
        ax.text(sub_plot['t_min'].max() * 0.99, 1.5, 'Zone dangereuse <3 km',
                ha='right', fontsize=10, color='#A83232', fontweight='bold')
        ax.axvspan(lift_min, baseline_min, color='#A83232', alpha=0.15, zorder=0,
                   label=f'Fenêtre "off-alerte" ({baseline_min - lift_min:.1f} min)')

        cg_dang = sub_plot[(~sub_plot['icloud']) & (sub_plot['dist'] < 3)]
        cg_norm = sub_plot[(~sub_plot['icloud']) & (sub_plot['dist'] >= 3)]
        ic_all = sub_plot[sub_plot['icloud']]

        ax.scatter(cg_norm['t_min'], cg_norm['dist'], color='#4F8BBF', s=80, marker='o',
                   edgecolors='black', linewidth=0.5, zorder=4, label='CG normal (3-20 km)')
        ax.scatter(ic_all['t_min'], ic_all['dist'], color='#BBBBBB', s=30, marker='x',
                   alpha=0.6, zorder=2, label='IC (non comptés)')
        ax.scatter(cg_dang['t_min'], cg_dang['dist'], color='#A83232', s=200, marker='*',
                   edgecolors='black', linewidth=1, zorder=5, label='CG dangereux (<3 km)')

        ax.axvline(lift_min, color='#2C8C5A', ls='-', lw=3,
                   label=f'Notre levée = {lift_min:.1f} min (T_q={sel["timer_used"]:.1f} min)')
        ax.axvline(baseline_min, color='black', ls=':', lw=2,
                   label=f'Baseline 30 min = {baseline_min:.1f} min')

        ax.set_xlabel('Temps depuis le 1er CG de l\'alerte (min)')
        ax.set_ylabel("Distance à l'aéroport (km)")
        ax.set_title(f"{sel['airport']} / {sel['airport_alert_id']} — "
                     f"{sel['missed_cg']} CG manqués (timer {sel['timer_used']:.1f} min)")
        ax.set_xlim(-2, max(35, sub_plot['t_min'].max() + 2))
        ax.set_ylim(0, max(22, sub_plot['dist'].max() + 2 if len(sub_plot) else 22))
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
