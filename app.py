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
def predict_Tq_last_cg(q: float):
    """T_q calibré (capé 30 min) pour le DERNIER CG de chaque alerte.
    C'est ce CG qui déclenche la vraie levée d'alerte en opérationnel."""
    aft, scaler, apt_cols, scaling = fit_model()
    cg_eval, _ = load_eval()
    last_cg = cg_eval[cg_eval['is_last']].copy().reset_index(drop=True)

    feat = last_cg[mm.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_sc = pd.DataFrame(scaler.transform(feat), columns=mm.FEATURES)
    apt_dum = pd.get_dummies(last_cg['airport'], prefix='apt', drop_first=True).astype(float)
    for col in apt_cols:
        if col not in apt_dum.columns:
            apt_dum[col] = 0.0
    apt_dum = apt_dum[apt_cols].reset_index(drop=True)
    X = pd.concat([X_sc.reset_index(drop=True), apt_dum], axis=1)

    Tq_raw = aft.predict_percentile(X, p=1.0 - q).values
    qs = sorted(scaling.keys())
    c_q = float(np.interp(q, qs, [scaling[k] for k in qs]))
    Tq = np.clip(c_q * Tq_raw, 0, 30)
    last_cg['T_q'] = Tq
    return last_cg


@st.cache_data
def build_predictions_csv(q: float):
    """Genere predictions.csv au format protocole officiel.

    Pour CHAQUE CG de l'eval (pas juste le dernier) :
      - predicted_date_end_alert = date(CG) + T_q(X) calibre cape a 30 min
      - confidence = S(30 | X) = proba prédite par le modele qu'aucun CG ne suive
        dans les 30 min (= ce CG est probablement le dernier de l'alerte)

    Cette confidence est monotone en position dans l'alerte : faible au debut,
    elevee a la fin.
    """
    aft, scaler, apt_cols, scaling = fit_model()
    cg_eval, _ = load_eval()
    all_cg = cg_eval.copy().reset_index(drop=True)

    feat = all_cg[mm.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_sc = pd.DataFrame(scaler.transform(feat), columns=mm.FEATURES)
    apt_dum = pd.get_dummies(all_cg['airport'], prefix='apt', drop_first=True).astype(float)
    for col in apt_cols:
        if col not in apt_dum.columns:
            apt_dum[col] = 0.0
    apt_dum = apt_dum[apt_cols].reset_index(drop=True)
    X = pd.concat([X_sc.reset_index(drop=True), apt_dum], axis=1)

    Tq_raw = aft.predict_percentile(X, p=1.0 - q).values
    qs = sorted(scaling.keys())
    c_q = float(np.interp(q, qs, [scaling[k] for k in qs]))
    Tq = np.clip(c_q * Tq_raw, 0, 30)

    survival = aft.predict_survival_function(X, times=[30.0])
    confidence = survival.iloc[0].values

    all_cg = all_cg.copy()
    all_cg['date_utc'] = pd.to_datetime(all_cg['date'], utc=True)

    pred_end_series = all_cg['date_utc'] + pd.to_timedelta(Tq, unit='m')
    predictions = pd.DataFrame({
        'airport': all_cg['airport'].values,
        'airport_alert_id': all_cg['airport_alert_id'].values,
        'prediction_date': all_cg['date_utc'].reset_index(drop=True),
        'predicted_date_end_alert': pred_end_series.reset_index(drop=True),
        'confidence': confidence,
        'T_q_min': Tq,
    })
    predictions['prediction_date'] = pd.to_datetime(predictions['prediction_date'], utc=True)
    predictions['predicted_date_end_alert'] = pd.to_datetime(
        predictions['predicted_date_end_alert'], utc=True)
    return predictions, all_cg


@st.cache_data
def evaluate(q: float):
    """Mesure protocole officielle Evaluation_databattle_meteorage.ipynb.

    Pour theta = q (= seuil sur la confidence) :
      1. On garde les predictions avec confidence >= theta.
      2. Pour chaque alerte, on prend la prediction LA PLUS PRECOCE (= early lift).
      3. On compte les CG <3 km arrivant APRES cette levee predite.
      4. Le gain par alerte = baseline_end (= last_lightning + 30 min) - levee_predite.

    Restriction CG-only au <3 km (les IC ne sont pas dangereux au sol).
    """
    predictions, _ = build_predictions_csv(q)
    events = predict_Tq(q)  # pour visu pedagogique en bas
    last_cg = predict_Tq_last_cg(q)
    _, df_raw = load_eval()
    df_raw = df_raw.copy()
    df_raw['date_utc'] = pd.to_datetime(df_raw['date'], utc=True)

    total_cg_3km = int(((df_raw['dist'] < 3) & (~df_raw['icloud'])).sum())
    total_eclairs_3km = int((df_raw['dist'] < 3).sum())

    # --- Protocole officiel : theta = q ---
    theta = q
    pred_over_theta = predictions.loc[predictions['confidence'] >= theta].copy()
    pred_over_theta['predicted_date_end_alert'] = pd.to_datetime(
        pred_over_theta['predicted_date_end_alert'], utc=True)
    pred_min = (pred_over_theta
                .groupby(['airport', 'airport_alert_id'])['predicted_date_end_alert']
                .min())
    pred_min = pd.to_datetime(pred_min, utc=True)

    alerts_grp = df_raw.groupby(['airport', 'airport_alert_id'])

    missed_cg = 0
    missed_all = 0
    gain_total_min = 0.0
    per_alert = []

    for (apt, aid), pred_end in pred_min.items():
        try:
            sub = alerts_grp.get_group((apt, aid))
        except KeyError:
            continue
        baseline_end = sub['date_utc'].max() + pd.Timedelta(minutes=30)
        gain_min = (baseline_end - pred_end).total_seconds() / 60.0

        # Restriction CG-only au <3 km (les IC ne ferment pas l'alerte et ne sont pas dangereux)
        mask_cg = (sub['dist'] < 3) & (~sub['icloud']) & (sub['date_utc'] > pred_end)
        mask_all = (sub['dist'] < 3) & (sub['date_utc'] > pred_end)
        miss_cg = int(mask_cg.sum())
        miss_all = int(mask_all.sum())

        missed_cg += miss_cg
        missed_all += miss_all
        gain_total_min += gain_min
        per_alert.append({
            'airport': apt,
            'airport_alert_id': aid,
            'pred_end': pred_end,
            'baseline_end': baseline_end,
            'gain_min': gain_min,
            'missed_cg_3km': miss_cg,
        })

    per_alert_df = pd.DataFrame(per_alert)
    n_alerts_evaluated = len(per_alert_df)
    n_alerts_total = alerts_grp.ngroups

    # === Mesure dynamique conservée pour la visualisation ===
    gap = events['gap_to_next_min'].values
    Tq_dyn = events['T_q'].values
    dist_next = events['dist_next'].values
    levee_avant_prochain = gap > Tq_dyn
    incident_mask = levee_avant_prochain & (dist_next < 3)
    incidents = events[incident_mask].copy()
    incidents['ecart_min'] = incidents['gap_to_next_min'] - incidents['T_q']
    incidents = incidents.sort_values('ecart_min', ascending=False).reset_index(drop=True)

    return {
        'q': q,
        'theta': theta,
        # --- MESURES PROTOCOLE OFFICIEL (headline) ---
        'n_alerts_total': n_alerts_total,
        'n_alerts_evaluated': n_alerts_evaluated,
        'predictions': predictions,
        'per_alert_df': per_alert_df,
        'last_cg': last_cg,
        'missed_cg_3km': missed_cg,
        'missed_eclairs_3km': missed_all,
        'total_cg_3km': total_cg_3km,
        'total_eclairs_3km': total_eclairs_3km,
        'risk_cg_pct': (missed_cg / total_cg_3km * 100) if total_cg_3km else 0,
        'risk_eclairs_pct': (missed_all / total_eclairs_3km * 100) if total_eclairs_3km else 0,
        'gain_total_h': gain_total_min / 60.0,
        'gain_avg_min_per_alert': (gain_total_min / n_alerts_evaluated) if n_alerts_evaluated else 0,
        # --- MESURE DYNAMIQUE (pedagogie / visualisation) ---
        'events': events,
        'incidents': incidents,
        'n_incidents': len(incidents),
        'n_levees_avant_prochain': int(levee_avant_prochain.sum()),
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

**Mesure opérationnelle (= protocole officiel Data Battle, headline) :**

Une fois l'alerte fermée naturellement (= 30 min sans nouveau CG), on a observé une décision
de levée par alerte au dernier CG. On compte :
- **Incident** : un éclair (CG ou IC) à <3 km de l'aéroport arrivant **après notre levée prédite**.
- **Risque protocole** : nombre d'incidents / total des éclairs <3 km dans l'eval (1 995).
- **Gain par alerte** : baseline_end (= last_éclair + 30 min) − levée_prédite (= last_CG + T_q).

C'est exactement ce que mesure la Section 8 du notebook *Weibull_final3km.ipynb* et la
procédure de l'officiel *Evaluation_databattle_meteorage.ipynb*.

**Mesure dynamique (pédagogique, en bas de page) :**

À chaque CG, on vérifie si T_q prédit < gap réel jusqu'au prochain CG. Sert à visualiser
des cas concrets où le modèle se trompe en cours d'alerte, mais ne reflète pas le risque
opérationnel final.
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
    # KPI globaux — PROTOCOLE OFFICIEL DATA BATTLE
    # ==========================================================
    st.subheader(f"Résultats à q = {q:.2f}  —  protocole officiel Data Battle, critère CG <3 km")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Alertes évaluées', f"{out['n_alerts_evaluated']:,} / {out['n_alerts_total']:,}",
               help="Alertes ayant au moins une prédiction avec confidence ≥ θ (= q). "
                    "Les autres conservent la baseline 30 min.")
    c2.metric('CG <3 km manqués', f"{out['missed_cg_3km']} / {out['total_cg_3km']}",
               help="CG (nuage-sol) à <3 km arrivant APRÈS la levée prédite la plus précoce de chaque alerte. "
                    "Seuls les CG comptent (les IC ne sont pas dangereux au sol).")
    c3.metric('Risque CG <3 km', f"{out['risk_cg_pct']:.2f} %",
               help=f"= {out['missed_cg_3km']} / {out['total_cg_3km']} CG <3 km. "
                    "C'est le critère opérationnel : seuls les éclairs nuage-sol sont dangereux.")
    c4.metric('Gain opérationnel cumulé', f"{out['gain_total_h']:.0f} h",
               help=f"Somme sur {out['n_alerts_evaluated']:,} alertes de (baseline_end − levée_prédite). "
                    "baseline_end = date du dernier éclair + 30 min (règle Météorage).")

    # Baseline 30 min Météorage — incidents avec la règle actuelle
    cg_eval_full, df_raw_full = load_eval()
    df_raw_full2 = df_raw_full.copy()
    df_raw_full2['date_utc'] = pd.to_datetime(df_raw_full2['date'], utc=True)
    last_cg_full = cg_eval_full[cg_eval_full['is_last']].copy()
    last_cg_full['date_utc'] = pd.to_datetime(last_cg_full['date'], utc=True)
    last_cg_full['pred_end_baseline'] = last_cg_full['date_utc'] + pd.Timedelta(minutes=30)
    baseline_grp = df_raw_full2.groupby(['airport', 'airport_alert_id'])
    baseline_missed_cg = 0
    for _, r in last_cg_full.iterrows():
        try:
            sub = baseline_grp.get_group((r['airport'], r['airport_alert_id']))
        except KeyError:
            continue
        baseline_missed_cg += int(((sub['dist'] < 3) & (~sub['icloud'])
                                    & (sub['date_utc'] > r['pred_end_baseline'])).sum())
    baseline_risk = baseline_missed_cg / out['total_cg_3km'] if out['total_cg_3km'] else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Baseline 30 min — risque', f"{baseline_risk*100:.2f} %",
               help=f"{baseline_missed_cg} CG <3 km arrivés après last_CG + 30 min "
                    "(= règle Météorage actuelle).")
    c2.metric('T_q médian (predictions)', f"{out['predictions']['T_q_min'].median():.1f} min",
               help="Médiane des T_q prédits sur l'ensemble des prédictions générées.")
    c3.metric('Gain moyen par alerte', f"{out['gain_avg_min_per_alert']:.1f} min",
               help=f"Moyenne sur les {out['n_alerts_evaluated']:,} alertes filtrées par θ.")
    c4.metric('Cible protocole', f"R < 2 %",
               help="Le protocole officiel fixe une cible R < 2 % comme limite acceptable.")

    if out['risk_cg_pct'] <= 2.0 and out['risk_cg_pct'] <= baseline_risk * 100:
        st.success(f"Risque CG {out['risk_cg_pct']:.2f} % "
                    f"≤ cible 2 % et ≤ baseline {baseline_risk*100:.2f} %  →  "
                    f"modèle plus sûr que la règle Météorage avec "
                    f"{out['gain_total_h']:.0f} h gagnées.")
    elif out['risk_cg_pct'] <= 2.0:
        st.info(f"Risque CG {out['risk_cg_pct']:.2f} % sous la cible 2 %. "
                  f"Gain : {out['gain_total_h']:.0f} h.")
    else:
        st.warning(f"Risque CG {out['risk_cg_pct']:.2f} % > cible 2 %. "
                    "Augmente q pour être plus conservateur.")

    with st.expander("Détail méthodologique — calcul identique à Evaluation_databattle_meteorage.ipynb"):
        st.markdown(f"""
**Étape 1 — Génération de `predictions.csv`** : pour CHAQUE CG de l'eval (pas juste le
dernier), on émet une prédiction :

```python
predicted_date_end_alert = date(CG) + T_q calibré
confidence = S(30 | X)  # proba que ce CG soit le dernier de l'alerte
```

**Étape 2 — Filtrage et sélection** (code repris du notebook officiel) :

```python
pred_over_theta = predictions.loc[predictions['confidence'] >= theta]
pred_min = pred_over_theta.groupby(['airport', 'airport_alert_id'])\\
                          ['predicted_date_end_alert'].min()
for (apt, aid), pred_end in pred_min.items():
    sub = alerts.get_group((apt, aid))
    baseline_end = sub['date'].max() + 30min
    gain += (baseline_end - pred_end).total_seconds()
    missed_cg += sum((sub['dist'] < 3) & (~sub['icloud']) & (sub['date'] > pred_end))
```

**Seule différence avec le notebook officiel** : le notebook officiel compte les éclairs <3 km
(CG + IC). Nous restreignons aux **CG seulement** (les IC sont des éclairs intra-nuage qui ne
touchent pas le sol — ils ne sont pas dangereux pour les opérations sol et ne ferment pas
l'alerte côté Météorage). C'est une lecture plus stricte du critère opérationnel.

**À q = {q:.2f}** :
- {out['n_alerts_evaluated']:,} alertes ont au moins une prédiction passant θ = {q:.2f}
- {out['missed_cg_3km']} CG <3 km arrivent après la levée prédite sur les {out['total_cg_3km']} CG dangereux totaux
- Risque CG <3 km = **{out['risk_cg_pct']:.2f} %**
- Gain cumulé = **{out['gain_total_h']:.0f} heures** sur les {out['n_alerts_evaluated']:,} alertes
""")

    # ==========================================================
    # Visualisation pédagogique des erreurs intermédiaires (mesure dynamique)
    # ==========================================================
    if out['n_incidents'] > 0:
        st.markdown('---')
        st.subheader(f"Cas pédagogiques — {out['n_incidents']} sous-estimations intermédiaires (mesure dynamique)")
        st.caption(
            f"⚠️ Ces {out['n_incidents']} cas ne sont **pas** les incidents du protocole officiel "
            f"(qui en compte {out['missed_cg_3km']} sur les CG <3 km). Ce sont des CG intermédiaires où T_q < gap "
            f"réel jusqu'au prochain CG dangereux. Ils servent à visualiser les erreurs de prédiction du modèle "
            f"en cours d'alerte, mais en opérationnel un autre CG arrive après donc on ne lève pas "
            f"de toute façon."
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
    events_for_emp = out['events']
    rows = []
    for alpha in [0.01, 0.05, 0.10]:
        ok = T_grid[risk_curve <= alpha]
        T_star = int(ok[0]) if len(ok) else None
        n_inc = (int(((events_for_emp['gap_to_next_min'] > T_star)
                       & (events_for_emp['dist_next'] < 3)).sum())
                 if T_star else None)
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
    # Gain opérationnel agrégé par aéroport via last_cg
    by_apt = out['last_cg'].groupby('airport').agg(
        n_alertes=('T_q', 'count'),
        T_q_median_last_min=('T_q', lambda s: round(s.median(), 2)),
        gain_operationnel_h=('T_q', lambda s: round((30 - s).clip(lower=0).sum() / 60, 1)),
    )
    inc_by_apt = out['incidents'].groupby('airport').size().rename('incidents_pedago')
    by_apt = by_apt.join(inc_by_apt).fillna(0)
    by_apt['incidents_pedago'] = by_apt['incidents_pedago'].astype(int)
    st.dataframe(by_apt, width='stretch')
