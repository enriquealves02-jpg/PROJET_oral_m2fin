# Battle Météorage — dossier oral

Tous les documents nécessaires pour la présentation orale.

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| **`script_oral.tex`** | **Script complet à dire à l'oral** (compilable avec pdflatex/Overleaf). Contient toutes les définitions, formules, exemples, et le narratif slide par slide. |
| **`presentation_oral.pptx`** | **PowerPoint** 22 slides avec formules, encadrés "DÉFINITION", exemples concrets. |
| **`rapport_final.tex`** | Rapport long (couvre tout : train + calibration + eval 3 km). |
| **`Weibull_final3km.ipynb`** | Notebook d'évaluation 3 km (exécuté avec sorties visibles). |
| **`weibull_final.ipynb`** | Notebook de base (théorie + calibration sur train). |
| **`app.py`** | Démo Streamlit interactive (`streamlit run app.py`). |
| **`meteorage_model.py`** | Module Python réutilisable (load, features, fit). |
| **`segment_alerts_all_airports_eval.csv`** | Données d'évaluation. |

## Ordre conseillé pour la présentation

1. **Slides 1-3** (PowerPoint) — Introduction du problème et de l'approche
2. **Slides 4-12** — Données, modèle Weibull AFT, calibration
3. **Slides 13-18** — Évaluation sur eval avec critère 3 km, exemples concrets
4. **Slide 19** — Démo Streamlit live (`streamlit run app.py`)
5. **Slides 20-22** — Limites, conclusion, merci

Le `script_oral.tex` contient le texte exact à dire pour chaque section.

## Lancement de la démo Streamlit

Depuis ce dossier ou depuis le dossier parent `PROJET/` :

```powershell
streamlit run app.py
```

(Attention : `app.py` doit pouvoir trouver `data(1)/data/segment_alerts_all_airports_train.csv`.
Vérifier les chemins relatifs si tu lances depuis ce dossier oral_projet/.)

## Résultats clés à mentionner

**Sur eval 2023-25 (973 alertes inédites) à q = 0,95 :**

| Mesure | Risque | Gain |
|---|---|---|
| Protocole officiel Data Battle (1 pred par alerte au last CG) | **0,60 %** (12/1995 éclairs <3km) | **~330 h** |
| Mesure dynamique (à chaque CG) | 9,61 % (37 incidents) | --- |
| Baseline 30 min Météorage | 1,04 % | référence |

**Phrase d'or pour le jury** : *"Sur eval 2023-25, à q=0,95, le protocole officiel donne 0,60 % de risque CG/3 km avec un gain de 330 heures. La démo Streamlit montre les 37 cas concrets où le modèle, mesuré dynamiquement à chaque CG, aurait sous-estimé un écart vers un CG dangereux."*
