GV2-EDGE – Architecture complète de détection des runners et estimation du potentiel

Ce document décrit l’architecture complète du système GV2-EDGE pour :

- détecter les low-float runners
- analyser les patterns historiques du marché
- estimer le potentiel de mouvement (+50%, +100%, +200%)
- exploiter les données temps réel pour générer des alertes rapides

Le système combine :

- données de marché
- mémoire historique (Market Memory)
- estimation du float actif
- modèles probabilistes

Objectif : identifier les actions susceptibles d’exploser avant les scanners publics.

---

1. Architecture globale

Pipeline du système :

Polygon Market Data
        ↓
Market Scanner
        ↓
Feature Extraction
        ↓
Float Estimation
        ↓
Market Memory Database
        ↓
Runner Probability Model
        ↓
Expected Move Estimation
        ↓
Trading Alerts

---

2. Sources de données

Le système utilise principalement les données provenant de Polygon :

Trades

price
size
timestamp
exchange

Quotes

bid
ask
bid_size
ask_size

Candles

open
high
low
close
volume
vwap

Reference data

shares_outstanding
sector
industry
market_cap

---

3. Market Memory

La Market Memory est une base qui stocke les patterns historiques des runners.

Table recommandée :

historical_runners

Structure :

date
ticker
open
high
close
volume
float
max_gain
sector

Exemple :

2024-05-12
ABCD
open 1.20
high 3.40
volume 45M
float 12M
max_gain 183%

---

4. Calcul du gain maximal

Pour chaque runner historique :

max_gain = (high_price - open_price) / open_price

Exemple :

open = 1.00
high = 3.20

max_gain = +220%

Ces données servent à construire la distribution des gains.

---

5. Features importantes

Les variables utilisées pour détecter les runners.

Relative Volume

RVOL = volume_1m / average_volume_1m

Exemple :

RVOL = 8

---

Price Velocity

price_vel_1m = (price_now - price_1m_ago) / price_1m_ago
price_vel_3m = (price_now - price_3m_ago) / price_3m_ago

---

Distance au VWAP

vwap_dist = (price - vwap) / vwap

Breakout au-dessus du VWAP :

signal momentum

---

Spread

spread = ask - bid

Spread serré :

liquidité forte

---

Liquidity Ratio

liq_ratio = order_book_depth / volume_1m

Un ratio faible indique :

faible liquidité

---

6. Estimation du float actif

Le float officiel ne reflète pas toujours le float réellement échangé.

Estimation :

float_actif ≈ volume_peak_intraday / turnover_typique

Avec :

turnover_typique ≈ 4 à 6

Exemple :

volume_peak = 24M
turnover_typique = 4

float_actif ≈ 6M

---

7. Turnover du float

turnover = volume_intraday / float

Exemple :

volume = 30M
float = 10M

turnover = 3

Un turnover élevé indique :

fort momentum

---

8. Détection des low-float runners

Conditions typiques :

RVOL ≥ 5
price_velocity ≥ 3%
vwap_dist > 0
spread faible
liq_ratio faible

Bonus :

turnover ≥ 1

---

9. Score de momentum

Exemple de scoring :

volume_spike        +2
price_acceleration  +2
vwap_breakout       +1
spread_serré        +1
liquidité faible    +1
turnover élevé      +1

Interprétation :

score ≥ 6 → runner potentiel
score ≥ 8 → runner explosif

---

10. Dataset Machine Learning

Structure :

ticker
date
relative_volume
price_velocity
vwap_distance
spread
float_estimate
label_runner
max_gain

Labels possibles :

gain ≥ 50%
gain ≥ 100%
gain ≥ 200%

---

11. Modèles Machine Learning

Modèles adaptés :

XGBoost
LightGBM
Random Forest

Pipeline :

features
↓
training
↓
probability prediction

Sortie :

runner_probability

---

12. Estimation du potentiel de mouvement

Le système prédit :

P(+50%)
P(+100%)
P(+200%)

Exemple :

P(+50%) = 0.89
P(+100%) = 0.61
P(+200%) = 0.22

---

13. Distribution des gains historiques

Exemple :

0-20%    █████████████████
20-50%   ███████████
50-100%  ███████
100-200% ████
200%+    ██

La plupart des moves restent sous :

+100%

---

14. Estimation du maximum move probable

Méthode simple :

expected_move = moyenne des runners similaires

Exemple :

120%
95%
160%
110%

Résultat :

expected_move ≈ 120%

---

15. Méthode par quantiles

quantile_50 = +60%
quantile_75 = +110%
quantile_90 = +180%

Interprétation :

90% des runners similaires ne dépassent pas +180%

---

16. Algorithme de détection (pseudo code)

for ticker in market_stream:

    rvol = volume_1m / avg_volume_1m
    price_vel = (price - price_1m_ago) / price_1m_ago
    vwap_dist = (price - vwap) / vwap

    spread = ask - bid
    depth = bid_size + ask_size
    liq_ratio = depth / volume_1m

    score = 0
    if rvol >= 5: score += 2
    if volume_1m >= 10 * avg_volume_1m: score += 2
    if price_vel >= 0.03: score += 2
    if vwap_dist > 0: score += 1
    if spread < threshold: score += 1
    if liq_ratio < threshold: score += 1

    if score >= 6:
        alert_runner(ticker)

---

17. Dashboard

Affichage possible :

TICKER : ABCD

RVOL : 9.4
Price Velocity : +4.8%
VWAP Distance : +1.3%
Spread : 0.18%

Estimated Active Float : ~6M
Momentum Score : 8

Signal : LOW FLOAT RUNNER

---

18. Fonctionnement temps réel

Le système tourne toutes les 1 à 5 secondes :

nouveaux trades
↓
update features
↓
score momentum
↓
runner detection
↓
watchlist dynamique
↓
alert

---

19. Objectif final

GV2-EDGE doit être capable de :

détecter les low float runners
estimer leur potentiel
reconnaître les patterns historiques
générer des alertes rapides

Pipeline final :

Market Data
        ↓
Feature Engine
        ↓
Float Estimation
        ↓
Market Memory
        ↓
Machine Learning
        ↓
Runner Probability
        ↓
Expected Move
        ↓
Alerts

Le système transforme ainsi les données historiques et temps réel en avantage prédictif pour détecter les futurs top gainers.
