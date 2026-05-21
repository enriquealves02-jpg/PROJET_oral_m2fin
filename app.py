"""
Démo Streamlit — Weibull AFT évalué sur le critère opérationnel CG / 3 km.

Logique de l'app :

- Le modèle Weibull AFT prédit, après chaque CG observé en zone 20 km, un temps d'attente T_q minutes
  jusqu'à la levée d'alerte.
- Un INCIDENT = le prochain CG arrive APRÈS T_q ET il est à moins de 3 km de l'aéroport
  (CG dangereux pour les opérations sol). Les IC ne comptent pas.
- L'app permet de voir les incidents un par un (timeline visuelle) et de comparer au
  risque empirique non-paramétrique (T fixe sans modèle).

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
# Chargement et entraînement (cache)
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
    grp = cg_eval.groupby(['airport', 'airport_alert_id'], sort=False)
    cg_eval['gap_to_next_min'] = grp['date'].shift(-1).sub(cg_eval['date']).dt.total_seconds() / 60
    cg_eval['dist_next'] = grp['dist'].shift(-1)
    cg_eval['date_next'] = grp['date'].shift(-1)

    df_raw = pd.read_csv(EVAL_CSV).rename(columns={'alert_id': 'airport_alert_id'})
    df_raw['date'] = pd.to_datetime(df_raw['date'], utc=True)
    return cg_eval, df_raw


@st.cache_data
def predict_Tq(q: float):
    """Calcule T_q calibré (capé 30 min) pour chaque CG event de l'eval."""
    aft, scaler, apt_cols, scaling = fit_model()
    cg_eval, _ = load_eval()
    events = cg_eval[cg_eval['event'] == 1].copy().reset_index(drop=True)

    feat = events[mm.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_sc = pd.DataFrame(scaler.transform(feat), columns=mm.FEATURES)
    apt_dum = pd.get_dummies(events['airport'], prefix='apt', drop_first=True).astype(float)
    for col in apt_cols:
        if col not in apt_dum.columns:
            apt_dum[col] = 0.0
    apt_dum = apt_dum[apt_cols].reset_index(drop=True)
    X = pd.concat([X_sc.reset_index(drop=True), apt_dum], axis=1)

    Tq_raw = aft.predict_percentile(X, p=1.0 - q).values
    qs = sorted(scaling.keys())
    c_q = float(np.interp(q, qs, [scaling[k] for k in qs]))
    Tq = np.clip(c_q * Tq_raw, 0, 30)

    events['T_q'] = Tq
    events['date_utc'] = pd.to_datetime(events['date'], utc=True)
    events['date_next_utc'] = pd.to_datetime(events['date_next'], utc=True)
    return events


@st.cache_data
def evaluate(q: float):
    """Mesure le risque opérationnel + retourne la liste des incidents."""
    events = predict_Tq(q)
    _, df_raw = load_eval()
    total_cg_3km = int(((df_raw['dist'] < 3) & (~df_raw['icloud'])).sum())

    gap = events['gap_to_next_min'].values
    Tq = events['T_q'].values
    dist_next = events['dist_next'].values

    levee_avant_prochain = gap > Tq  # cas où on aurait levé avant le prochain CG
    incident_mask = levee_avant_prochain & (dist_next < 3)
    incidents = events[incident_mask].copy()
    incidents['ecart_min'] = incidents['gap_to_next_min'] - incidents['T_q']
    incidents = incidents.sort_values('ecart_min', ascending=False).reset_index(drop=True)

    return {
        'q': q,
        'events': events,
        'incidents': incidents,
        'n_incidents': len(incidents),
        'n_levees_avant_prochain': int(levee_avant_prochain.sum()),
        'total_cg_3km': total_cg_3km,
        'risque_reel': len(incidents) / total_cg_3km if total_cg_3km else 0,
        'risque_conditionnel': (len(incidents) / int(levee_avant_prochain.sum())
                                  if levee_avant_prochain.sum() else 0),
        'gain_total_min': float((30 - events['T_q']).clip(lower=0).sum()),
    }


@st.cache_data
def empirical_risk():
    """Risque empirique non-paramétrique sans modèle (T fixe).
    P(gap > T ET dist_next <3) / total CG <3km."""
    cg_eval, df_raw = load_eval()
    events = cg_eval[cg_eval['event'] == 1]
    total_cg_3km = int(((df_raw['dist'] < 3) & (~df_raw['icloud'])).sum())

    gaps = events['gap_to_next_min'].values
    dists = events['dist_next'].values
    danger = dists < 3

    T_grid = np.arange(1, 31)
    risk = np.array([((gaps > T) & danger).sum() / total_cg_3km for T in T_grid])
    return T_grid, risk, total_cg_3km


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("Battle Météorage — Weibull AFT, critère opérationnel CG / 3 km")

with st.expander("Comprendre le modèle et la mesure d'erreur", expanded=True):
    st.markdown("""
**Le modèle Weibull AFT en production :**

À chaque CG observé en zone 20 km autour d'un aéroport, le modèle prédit un temps d'attente
**T_q minutes** jusqu'à la levée d'alerte. Si aucun nouveau CG n'arrive pendant T_q,
on lève. Sinon, le compteur est remis à zéro au nouveau CG.

C'est la même mécanique que la règle Météorage 30 min, mais avec **T_q adapté au contexte**
de chaque éclair (rythme, distance, intensité, saison).

**Définition d'un incident :**

Un incident survient quand le prochain CG arrive **après T_q** (= on a levé trop tôt) **et**
qu'il est à **moins de 3 km** de l'aéroport (= dangereux pour les opérations sol).
Les IC (intra-nuage) ne comptent pas — ils ne ferment pas l'alerte côté Météorage
et ne sont pas dangereux au sol.

**Mesure du risque :**

- **Risque réel** = nombre d'incidents / nombre total de CG <3 km dans l'eval (385)
- **Risque conditionnel** = nombre d'incidents / nombre de fois où on aurait levé avant le prochain CG
""")

# Sidebar
st.sidebar.title('Réglage du modèle')
q = st.sidebar.select_slider(
    'Quantile q (niveau de confiance)',
    options=[0.85, 0.90, 0.93, 0.95, 0.97, 0.99],
    value=0.95,
)
st.sidebar.metric('Risque cible nominal (1 − q)', f'{(1-q)*100:.0f} %')
st.sidebar.caption(
    "$q$ proche de 1 = conservateur (gain modéré, peu d'incidents)\n\n"
    "$q$ proche de 0.5 = agressif (gain élevé, plus d'incidents)"
)

# Init session state for the launch button
if 'launched' not in st.session_state:
    st.session_state.launched = False

st.markdown('---')
col_btn, col_status = st.columns([1, 3])
with col_btn:
    if st.button("Lancer l'évaluation", type='primary', width='stretch'):
        st.session_state.launched = True
with col_status:
    if st.session_state.launched:
        st.caption("Évaluation active. Tu peux changer q (sidebar) ou les sélecteurs ci-dessous, "
                    "tout se met à jour automatiquement.")
    else:
        st.caption("Clique pour évaluer le modèle sur les 17 037 CG de l'eval.")

if st.session_state.launched:
    with st.spinner(f"Calcul à q = {q:.2f}..."):
        out = evaluate(q)

    # ==========================================================
    # KPI globaux
    # ==========================================================
    st.subheader(f"Résultats opérationnels à q = {q:.2f}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('CG évalués', f"{len(out['events']):,}")
    c2.metric('Levées avant prochain CG',
               f"{out['n_levees_avant_prochain']:,}",
               help="Nombre de fois où T_q prédit < gap réel jusqu'au prochain CG (= on aurait levé avant qu'il n'arrive).")
    c3.metric('Incidents (CG <3 km)',
               f"{out['n_incidents']} / {out['total_cg_3km']}",
               help="Parmi les levées prématurées, combien voyaient un prochain CG dangereux.")
    c4.metric('Risque réel',
               f"{out['risque_reel']*100:.2f} %",
               help=f"Risque conditionnel : {out['risque_conditionnel']*100:.2f} % (incidents / levées prématurées)")

    # Comparaison à la baseline 30 min
    cg_eval_full, df_raw_full = load_eval()
    events_full = cg_eval_full[cg_eval_full['event'] == 1]
    baseline_incidents = int(((events_full['gap_to_next_min'] > 30)
                                & (events_full['dist_next'] < 3)).sum())
    baseline_risk = baseline_incidents / out['total_cg_3km']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Baseline 30 min — risque', f"{baseline_risk*100:.2f} %",
               help=f"{baseline_incidents} CG <3 km dont le gap intra-alerte dépasse 30 min")
    c2.metric('Modèle T_q médian', f"{out['events']['T_q'].median():.1f} min",
               help='Médiane des T_q prédits sur les 17 037 CG')
    gain_h_per_event = out['gain_total_min'] / len(out['events'])
    c3.metric('Gain moyen par CG', f"{gain_h_per_event:.1f} min",
               help="Économie moyenne par CG par rapport aux 30 min de Météorage")
    c4.metric('Gain total cumulé', f"{out['gain_total_min']/60:.0f} h",
               help="Somme des (30 - T_q) sur tous les CG")

    if out['n_incidents'] == 0:
        st.success("Aucun incident détecté à ce niveau de q.")
    elif out['risque_reel'] <= baseline_risk:
        st.info(f"Risque réel {out['risque_reel']*100:.2f} % proche ou en dessous de la baseline "
                  f"{baseline_risk*100:.2f} % — bon compromis.")
    else:
        st.warning(f"Risque réel {out['risque_reel']*100:.2f} % supérieur à la baseline "
                    f"{baseline_risk*100:.2f} %. Augmente q pour être plus conservateur.")

    # ==========================================================
    # Liste des incidents avec timeline visuelle
    # ==========================================================
    if out['n_incidents'] > 0:
        st.markdown('---')
        st.subheader(f"Les {out['n_incidents']} incidents détectés à q = {q:.2f}")
        st.markdown(
            "Chaque ligne du tableau est un cas concret : *le modèle a dit T_q minutes après ce CG, "
            "mais le prochain CG est arrivé après T_q et il était dans la zone dangereuse 3 km.*"
        )

        disp = out['incidents'][['airport', 'airport_alert_id', 'date',
                                    'T_q', 'gap_to_next_min', 'ecart_min', 'dist_next']].copy()
        disp['date'] = pd.to_datetime(disp['date'], utc=True).dt.strftime('%Y-%m-%d %H:%M')
        disp['T_q'] = disp['T_q'].round(2)
        disp['gap_to_next_min'] = disp['gap_to_next_min'].round(2)
        disp['ecart_min'] = disp['ecart_min'].round(2)
        disp['dist_next'] = disp['dist_next'].round(2)
        disp.columns = ['Aéroport', 'Alerte', 'Date du CG (référence)',
                          'T_q prédit (min)', 'Gap réel jusqu\'au prochain CG (min)',
                          'Écart (min)', 'Distance prochain CG (km)']
        st.dataframe(disp, hide_index=True, width='stretch')

        # Selecteur pour timeline
        st.markdown('---')
        st.subheader("Voir un incident en détail (timeline)")
        labels = [
            f"{r['airport']} / {r['airport_alert_id']} — "
            f"CG du {pd.to_datetime(r['date'], utc=True).strftime('%Y-%m-%d %H:%M')} — "
            f"prédit {r['T_q']:.1f} min, réel {r['gap_to_next_min']:.1f} min, "
            f"prochain CG à {r['dist_next']:.2f} km"
            for _, r in out['incidents'].iterrows()
        ]
        chosen = st.selectbox(f"{len(labels)} incidents", labels)
        sel = out['incidents'].iloc[labels.index(chosen)]

        # Charger toute l'alerte
        _, df_raw = load_eval()
        sub = df_raw[(df_raw['airport'] == sel['airport'])
                       & (df_raw['airport_alert_id'] == sel['airport_alert_id'])].copy()
        sub = sub.sort_values('date').reset_index(drop=True)

        # Référence = date du CG où on a fait la prédiction
        ref = pd.to_datetime(sel['date'], utc=True)
        sub['t_min'] = (sub['date'] - ref).dt.total_seconds() / 60
        T_q = float(sel['T_q'])
        gap_real = float(sel['gap_to_next_min'])
        dist_next = float(sel['dist_next'])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Aéroport', sel['airport'])
        c2.metric('Alerte', sel['airport_alert_id'])
        c3.metric('Modèle a dit', f"{T_q:.1f} min")
        c4.metric('Vrai gap', f"{gap_real:.1f} min",
                   f"+{gap_real - T_q:.1f} min vs prédiction")
        c5.metric('Distance prochain CG', f"{dist_next:.2f} km")

        # Plot timeline
        fig, ax = plt.subplots(figsize=(13, 5.5))

        # Bande dangereuse <3 km
        ax.axhspan(0, 3, color='red', alpha=0.06, zorder=0)
        ax.text(sub['t_min'].max() * 0.99, 1.5, 'Zone dangereuse <3 km',
                  ha='right', fontsize=10, color='#A83232', fontweight='bold')

        # Bande "fenêtre d'erreur" : entre notre levée T_q et l'arrivée du prochain CG
        ax.axvspan(T_q, gap_real, color='#A83232', alpha=0.15, zorder=0,
                    label=f'Fenêtre "off-alerte" ({gap_real - T_q:.1f} min)')

        # Marquer les éclairs
        cg_dang = sub[(~sub['icloud']) & (sub['dist'] < 3)]
        cg_norm = sub[(~sub['icloud']) & (sub['dist'] >= 3)]
        ic_all = sub[sub['icloud']]

        ax.scatter(cg_norm['t_min'], cg_norm['dist'], color='#4F8BBF', s=80, marker='o',
                    edgecolors='black', linewidth=0.5, zorder=4, label='CG normal (3-20 km)')
        ax.scatter(ic_all['t_min'], ic_all['dist'], color='#BBBBBB', s=30, marker='x',
                    alpha=0.6, zorder=2, label='IC (non comptés)')
        ax.scatter(cg_dang['t_min'], cg_dang['dist'], color='#A83232', s=200, marker='*',
                    edgecolors='black', linewidth=1, zorder=5, label='CG dangereux (<3 km)')

        # Lignes verticales
        ax.axvline(0, color='gray', ls='-', alpha=0.4)
        ax.text(0, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 21,
                  ' CG référence (modèle prédit ici)', fontsize=9, color='gray', va='top')

        ax.axvline(T_q, color='#2C8C5A', ls='-', lw=3,
                    label=f'Notre levée prédite = {T_q:.1f} min')

        ax.axvline(gap_real, color='#A83232', ls='--', lw=2.5,
                    label=f'Prochain CG arrivé = {gap_real:.1f} min')

        # Annoter l'incident
        prochain = sub[(sub['date'] > ref) & (~sub['icloud'])
                          & (sub['dist'] < 3)].nsmallest(1, 't_min')
        if len(prochain):
            r0 = prochain.iloc[0]
            ax.annotate(
                f"INCIDENT\nCG <3 km à {r0['t_min']:.1f} min\n(distance {r0['dist']:.2f} km)\n"
                f"On aurait levé {r0['t_min'] - T_q:.1f} min trop tôt",
                xy=(r0['t_min'], r0['dist']),
                xytext=(r0['t_min'] + 2, max(r0['dist'] + 5, 8)),
                fontsize=11, fontweight='bold', color='#A83232',
                arrowprops=dict(arrowstyle='->', color='#A83232', lw=2),
            )

        ax.set_xlabel('Temps depuis le CG référence (min)')
        ax.set_ylabel("Distance de l'éclair à l'aéroport (km)")
        ax.set_title(
            f"Incident {sel['airport']} / {sel['airport_alert_id']} : "
            f"modèle a prédit {T_q:.1f} min, prochain CG arrivé à {gap_real:.1f} min "
            f"({dist_next:.2f} km de l'aéroport)"
        )
        ax.set_xlim(-2, max(35, sub['t_min'].max() + 2))
        ax.set_ylim(0, max(22, sub['dist'].max() + 2 if len(sub) else 22))
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.markdown(
            f"**Lecture pas à pas :**\n\n"
            f"1. À {ref.strftime('%H:%M:%S')} un CG est observé en zone d'alerte (= référence à t=0 sur le graphique).\n"
            f"2. Le modèle Weibull prédit qu'aucun autre CG n'arrivera dans les **{T_q:.1f} minutes** suivantes.\n"
            f"3. Donc, à t = {T_q:.1f} min, on lèverait l'alerte (ligne verte).\n"
            f"4. Mais en réalité, le prochain CG arrive à t = **{gap_real:.1f} min** (ligne rouge en pointillé),\n"
            f"   à **{dist_next:.2f} km** de l'aéroport — c'est un CG dangereux.\n"
            f"5. Pendant la fenêtre rouge [{T_q:.1f}, {gap_real:.1f}] min, "
            f"l'aéroport aurait été en opérations alors qu'un CG dangereux allait frapper. **Incident.**"
        )

    # ==========================================================
    # Risque empirique non-paramétrique
    # ==========================================================
    st.markdown('---')
    st.subheader("Risque empirique non-paramétrique (sans modèle)")

    st.markdown("""
Indépendamment du modèle Weibull, on peut mesurer pour chaque seuil $T$ **fixe** combien de
CG dangereux seraient ratés. C'est la même définition d'incident mais avec $T_q = T$ constant.

$$\\widehat{R}(T) = \\frac{|\\{i : \\Delta_i > T \\text{ et } \\text{dist}_{i+1} < 3 \\text{ km}\\}|}{N_{\\text{CG}<3\\text{km}}}$$

C'est la reproduction de l'analyse non-paramétrique du notebook `weibull_final` (cellule 72),
adaptée au critère 3 km.
""")

    T_grid, risk_curve, total_3km_emp = empirical_risk()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(T_grid, risk_curve * 100, marker='o', lw=2, color='#A83232')
    for alpha, col in [(0.01, '#2C8C5A'), (0.05, '#4F8BBF'), (0.10, '#D48A1E')]:
        ax.axhline(alpha * 100, color=col, ls='--', alpha=0.5, label=f'alpha = {alpha*100:.0f} %')
        ok = T_grid[risk_curve <= alpha]
        if len(ok):
            T_star = int(ok[0])
            ax.scatter([T_star], [alpha * 100], s=80, color=col, edgecolors='black', zorder=5)
            ax.annotate(f'T* = {T_star} min', xy=(T_star, alpha*100),
                         xytext=(T_star + 1, alpha*100 + 3), fontsize=10,
                         color=col, fontweight='bold')
    ax.axvline(30, color='black', ls=':', alpha=0.5, label='Baseline 30 min')
    # Marquer le T_q médian du modèle au q sélectionné
    ax.axvline(out['events']['T_q'].median(), color='#2C8C5A', ls='-', lw=2,
                label=f'T_q médian modèle (q={q}) = {out["events"]["T_q"].median():.1f} min')
    ax.set_xlabel('Seuil T fixe (min) après le dernier CG')
    ax.set_ylabel('Risque empirique = % CG <3 km ratés')
    ax.set_title('Risque empirique non-paramétrique sur eval 2023-25')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 31)
    st.pyplot(fig, clear_figure=True)

    # Tableau T* par seuil
    rows = []
    for alpha in [0.01, 0.05, 0.10]:
        ok = T_grid[risk_curve <= alpha]
        T_star = int(ok[0]) if len(ok) else None
        n_inc = int(((events_full['gap_to_next_min'] > T_star) & (events_full['dist_next'] < 3)).sum()) if T_star else None
        rows.append({
            'Niveau de risque alpha': f'{alpha*100:.0f} %',
            'T* (min)': f'{T_star}' if T_star else '> 30',
            'Gain vs 30 min': f'+{30 - T_star} min' if T_star else 'NA',
            'Incidents observés': f'{n_inc} / {total_3km_emp}' if T_star else 'NA',
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    # ==========================================================
    # Synthèse par aéroport
    # ==========================================================
    st.markdown('---')
    st.subheader("Synthèse par aéroport")
    by_apt = out['events'].groupby('airport').agg(
        n_cg=('cg_rank', 'count'),
        T_q_median=('T_q', lambda s: round(s.median(), 2)),
        gain_total_h=('T_q', lambda s: round((30 - s).clip(lower=0).sum() / 60, 1)),
    )
    inc_by_apt = out['incidents'].groupby('airport').size().rename('incidents')
    by_apt = by_apt.join(inc_by_apt).fillna(0)
    by_apt['incidents'] = by_apt['incidents'].astype(int)
    st.dataframe(by_apt, width='stretch')
