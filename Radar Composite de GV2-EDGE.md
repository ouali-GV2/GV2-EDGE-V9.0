Spécification complète — Radar Composite de GV2-EDGE

Objectif

Le Radar Composite est le moteur principal de détection d’opportunités de GV2-EDGE.

Son objectif est de détecter les actions susceptibles de devenir des top gainers avant que le mouvement soit visible pour la majorité du marché.

Le radar doit identifier les déséquilibres du marché dès leur apparition, en analysant simultanément plusieurs signaux :

- augmentation anormale du volume
- accélération des transactions
- pression acheteur
- rotation du float
- compression du prix avant breakout

Le radar fonctionne en temps réel sur les données marché.

Les données temps réel peuvent provenir de :

- Polygon.io

---

Architecture générale

Le système fonctionne selon le pipeline suivant :

Market Data Stream
↓
Universe Filter
↓
Feature Engine
↓
Composite Radar Engine
↓
Momentum Score
↓
Ranking des actions
↓
Alertes

Le radar doit recalculer les scores toutes les 100 à 500 millisecondes.

---

1. Universe Filter

Avant toute analyse avancée, le système doit réduire l’univers d’actions afin de supprimer les titres non pertinents.

L’univers initial peut contenir environ :

8000 actions US

Le filtre doit réduire cet univers à environ :

200 à 400 actions

Critères recommandés :

- prix > 0.20$
- prix < 20$
- float < 50M
- volume moyen > 500k
- action listée sur NASDAQ ou NYSE
- exclure les actions OTC

Objectif :

Les plus gros runners du marché proviennent généralement des small caps à faible float.

---

2. Feature Engine

Le Feature Engine maintient en mémoire plusieurs métriques temps réel pour chaque action.

Ces métriques sont mises à jour à chaque nouveau trade ou quote.

Variables principales :

- last_price
- volume_1min
- volume_today
- average_volume
- bid_price
- ask_price
- bid_size
- ask_size
- trade_count
- spread
- timestamp

Variables dérivées :

- relative_volume
- price_change_30s
- price_change_60s
- price_change_5min
- trade_speed
- bid_ask_ratio
- float_rotation
- order_flow_ratio
- volatility

---

3. Modules du Radar Composite

Le radar est composé de plusieurs modules de détection.

Chaque module produit un signal et contribue au Momentum Score.

---

3.1 Volume Anomaly

Ce module détecte une activité anormale du volume.

Calcul :

relative_volume = volume_1min / average_volume_1min

Signaux :

- relative_volume > 3 → activité inhabituelle
- relative_volume > 5 → forte activité
- relative_volume > 10 → breakout potentiel

Contribution au score :

+2

---

3.2 Price Momentum

Ce module détecte une accélération du prix.

Calculs :

- price_change_30s
- price_change_60s
- price_change_5min

Signaux :

- price_change_60s > 3%
- price_change_5min > 5%

Contribution au score :

+2

---

3.3 Liquidity Imbalance

Ce module analyse la pression acheteur / vendeur.

Calcul :

bid_ask_ratio = bid_size / ask_size

Signal :

bid_ask_ratio > 3

Cela signifie que les acheteurs dominent le carnet d’ordres.

Contribution au score :

+2

---

3.4 Trade Speed

Ce module détecte l’augmentation rapide du nombre de transactions.

Calcul :

trade_speed = trades_last_5s / average_trades_5s

Signal :

trade_speed > 4

Contribution au score :

+1

---

3.5 Spread Compression

Ce module détecte une réduction du spread.

Calcul :

spread = ask_price − bid_price

Signal :

spread < spread_average

Cela indique une augmentation de la liquidité.

Contribution au score :

+1

---

3.6 Float Rotation

Ce module mesure la rotation du float.

Calcul :

float_rotation = volume_today / float

Signaux :

- rotation > 1
- rotation > 2

Cela signifie que le float commence à tourner.

Contribution au score :

+1

---

3.7 Order Flow Aggression

Ce module détecte les acheteurs agressifs.

Méthode :

- transaction proche du ask → achat agressif
- transaction proche du bid → vente agressive

Calcul :

order_flow_ratio = aggressive_buy_volume / aggressive_sell_volume

Signaux :

- ratio > 3 → domination des acheteurs
- ratio > 5 → forte accumulation

Contribution au score :

+2

---

3.8 Pre-Breakout Compression

Avant certains mouvements violents, le prix entre dans une phase de compression.

Pattern :

impulsion
↓
consolidation
↓
compression
↓
breakout

Conditions :

- range très serré
- volume élevé
- pression acheteur dominante

Calcul :

range = high_30s − low_30s

Signal :

range < 1% du prix

Contribution au score :

+2

---

3.9 Volatility Spike

Ce module détecte une augmentation soudaine de volatilité.

Calcul :

volatility = ATR court terme

Signal :

volatility > average_volatility

Contribution au score :

+1

---

4. Catalysts (non bloquant)

Les catalyseurs ne doivent jamais bloquer le radar.

Ils servent uniquement à augmenter la priorité d’une action.

Exemples :

- earnings
- annonces FDA
- partenariats
- insider buying
- filings SEC

Sources possibles :

- Finnhub
- U.S. Securities and Exchange Commission

Contribution au score :

+3

---

5. Calcul du Momentum Score

Le score final est la somme des contributions de chaque module.

Exemple :

volume anomaly = +2
momentum = +2
liquidity imbalance = +2
trade speed = +1
spread compression = +1
float rotation = +1
order flow aggression = +2
compression = +2
catalyst = +3

Momentum Score = somme des points

---

6. Seuils d’alerte

score ≥ 3 → action intéressante
score ≥ 5 → momentum fort
score ≥ 7 → runner potentiel
score ≥ 9 → breakout très probable

---

7. Classement des actions

Le radar doit produire une liste d’actions classées par score.

Structure de sortie :

- symbol
- momentum_score
- relative_volume
- price_change
- float_rotation
- timestamp

Exemple :

1. ABC score 9
2. XYZ score 7
3. DEF score 6

---

8. Boucle temps réel

Le radar fonctionne en continu.

Cycle :

trade reçu
↓
mise à jour des features
↓
recalcul du score
↓
mise à jour du classement
↓
déclenchement des alertes

Fréquence recommandée :

100 ms à 500 ms

---

9. Sessions de marché

Le radar doit adapter ses paramètres selon la session :

- Premarket
- Regular Market
- After Hours
- Overnight

Les seuils peuvent varier selon la liquidité.

---

10. Watchlist manuelle

Le système doit permettre d’ajouter des actions manuellement.

Exemple :

DGNX → catalyst FDA
TSLA → earnings

Les actions de la watchlist doivent être surveillées avec une fréquence plus élevée.

---

Objectif final

Le Radar Composite doit détecter le moment précis où l’accumulation commence sur une action.

Signaux clés :

- volume qui augmente
- pression acheteur
- accélération du trading
- compression du prix

Ce moment précède souvent :

- breakout
- top gainer
- short squeeze

GV2-EDGE doit détecter cette phase avant les scanners traditionnels.
