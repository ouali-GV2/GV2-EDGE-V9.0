# GV2-EDGE — System Reference & Development Guide

> **Version** : V8.0 (Architecture V7.0 + Acceleration Engine V8)
> **Derniere mise a jour** : 2026-02-17
> **Deploiement** : Hetzner CX33 (headless Linux) + IBKR Gateway (optionnel)
> **Langage** : Python 3.11+ (asyncio + threading)

---

## 1. VISION

GV2-EDGE est un systeme de **detection anticipative** des top gainers small-cap US intraday.

**Objectif principal** : Identifier les actions susceptibles de gagner +30% a +300% en une journee, **AVANT** que le mouvement ne commence, en analysant les catalysts, le volume, les options, le sentiment social et les patterns techniques.

**Philosophie fondamentale** :
- La **detection ne s'arrete JAMAIS** — seule l'execution est limitee
- Le trader voit **TOUS** les signaux, meme ceux bloques par les limites
- Le systeme apprend de ses erreurs (Market Memory) et s'ameliore
- **Anticipation > Reaction** : detecter les signaux AVANT le spike, pas apres

**Cible de marche** :
- Small caps US (market cap < $2B)
- Prix $0.50 - $50
- Catalysts : FDA, M&A, Earnings, Contracts, Short Squeeze
- Sessions : Pre-market (04:00-09:30 ET), RTH (09:30-16:00 ET), After-hours (16:00-20:00 ET)

---

## 2. ARCHITECTURE

### 2.1 Pipeline Principal V7.0 (3 couches)

```
COUCHE 1 — SIGNAL PRODUCER (detection illimitee)
    Produit des signaux 24/7, jamais bloque
    Input:  MonsterScore + Catalyst + AccelerationState + PreSpike
    Output: UnifiedSignal (BUY_STRONG / BUY / WATCH / EARLY_SIGNAL / NO_SIGNAL)

COUCHE 2 — ORDER COMPUTER (calcul systematique)
    Calcule un ordre pour CHAQUE signal BUY/BUY_STRONG
    Input:  UnifiedSignal + prix + ATR + capital
    Output: ProposedOrder (shares, entry, stop, targets, R/R)

COUCHE 3 — EXECUTION GATE (limites et autorisations)
    SEULE couche avec des limites d'execution
    Input:  ProposedOrder + RiskFlags + AccountState
    Output: ExecutionDecision (ALLOWED / REDUCED / BLOCKED / DELAYED)
    Le signal original est TOUJOURS preserve (signal_preserved=True)
```

### 2.2 Cycle Principal (`main.py`)

```
run_edge() — Boucle infinie
│
├── After-Hours (16:00-20:00 ET) — MODE ANTICIPATION
│   ├── News Flow Screener (SEC + Finnhub)
│   ├── Extended Hours Gap Scan
│   ├── Options Flow Scan
│   ├── Anticipation Engine (mode "afterhours")
│   └── After-Hours Scanner
│   Sleep: 600s (10 min)
│
├── Pre-Market (04:00-09:30 ET) — MODE CONFIRMATION
│   ├── Anticipation Engine (mode "premarket")
│   ├── Signal Upgrades (WATCH_EARLY → BUY)
│   └── edge_cycle_v7() [pipeline complet]
│   Sleep: 300s (5 min)
│
├── RTH (09:30-16:00 ET) — MODE EXECUTION
│   ├── Clear expired signals
│   └── edge_cycle_v7() [pipeline complet]
│   Sleep: 180s (3 min)
│
└── Market Closed — MODE IDLE
    Sleep: 900s (15 min)
```

### 2.3 Pipeline V7 par ticker (`process_ticker_v7`)

```
Pour chaque ticker dans l'univers:
  1. compute_monster_score(ticker)        → score 0.0-1.0
  2. compute_features(ticker)             → features techniques
  3. get_ibkr().get_quote(ticker)         → prix temps reel
  4. producer.detect(DetectionInput)       → SignalType (Layer 1)
  5. pre_halt.assess(ticker)              → PreHaltState
  6. enrich_signal_with_context()         → MRP/EP (Market Memory)
  7. computer.compute_order()             → ProposedOrder (Layer 2)
  8. guard.assess(ticker)                 → RiskFlags
  9. gate.evaluate()                      → ExecutionDecision (Layer 3)
  10. handle_signal_result()              → Log + Telegram + Market Memory
```

---

## 3. MODULES

### 3.1 Carte des modules

```
GV2-EDGE-V7.0/
├── main.py                              # Point d'entree, boucle principale
├── config.py                            # Configuration globale (464 lignes)
├── daily_audit.py                       # Audit quotidien (hit rate, lead time)
├── weekly_deep_audit.py                 # Audit hebdomadaire avance
├── performance_attribution.py           # Attribution des trades backtest
│
├── src/
│   ├── signal_engine.py                 # [LEGACY] Moteur signal (delegue a V7)
│   ├── anticipation_engine.py           # Detection precoce (IBKR radar + V6.1 ingestors)
│   ├── afterhours_scanner.py            # Scan post-marche (16h-20h ET)
│   ├── catalyst_score_v3.py             # Score catalyst ponderes (5 tiers, 18 types)
│   ├── dark_pool_alternatives.py        # Detection dark pool indirecte
│   ├── ensemble_engine.py               # Confluence douce (boost, jamais bloque)
│   ├── extended_hours_quotes.py         # Quotes pre/post marche + gap boost
│   ├── fda_calendar.py                  # Calendrier FDA (PDUFA, phases, conferences)
│   ├── feature_engine.py               # Calcul features techniques (Finnhub candles)
│   ├── historical_beat_rate.py          # Taux de beat earnings historique
│   ├── ibkr_connector.py               # Connexion IBKR Level 1 (reconnexion auto)
│   ├── ibkr_news_trigger.py            # Alertes rapides news IBKR (trigger, pas score)
│   ├── news_flow_screener.py           # Screener news global (SEC + Finnhub → tickers)
│   ├── nlp_enrichi.py                  # NLP avance (sentiment, entites, urgence)
│   ├── options_flow.py                 # Options flow monitor
│   ├── options_flow_ibkr.py            # Detection options inhabituelles (4 signaux)
│   ├── pattern_analyzer.py             # Patterns techniques (flag, squeeze, HLs)
│   ├── pm_scanner.py                   # Scanner pre-marche V8 (gap zones)
│   ├── pm_transition.py                # Analyse transition PM → RTH
│   ├── portfolio_engine.py             # Sizing positions (ATR-based)
│   ├── pre_halt_engine.py              # Evaluation risque halt (LOW/MEDIUM/HIGH)
│   ├── pre_spike_radar.py              # Signaux precurseurs de spike
│   ├── repeat_gainer_memory.py         # Memoire des repeat runners (SQLite)
│   ├── signal_logger.py                # Log persistant signaux (SQLite)
│   ├── social_buzz.py                  # Tracker buzz social (Reddit, StockTwits)
│   ├── universe_loader.py              # Chargement univers V3 (1 API call, ~3000 tickers)
│   └── watch_list.py                   # Watchlist calendrier (J-7 → J-Day)
│
│   ├── engines/                         # Moteurs V7/V8 (coeur du systeme)
│   │   ├── signal_producer.py           # LAYER 1: Detection illimitee V8
│   │   ├── order_computer.py            # LAYER 2: Calcul ordres systematique
│   │   ├── execution_gate.py            # LAYER 3: Gate execution (9 checks)
│   │   ├── acceleration_engine.py       # V8: Derivees + z-scores (anticipation)
│   │   ├── smallcap_radar.py            # V8: Radar small-cap (ACCUMULATING → BREAKOUT)
│   │   └── ticker_state_buffer.py       # V8: Ring buffer (120 snapshots/ticker)
│   │
│   ├── scoring/                         # Systeme de scoring
│   │   ├── monster_score.py             # Monster Score V4 (9 composants)
│   │   └── score_optimizer.py           # Auto-tuning poids
│   │
│   ├── models/                          # Types de donnees
│   │   ├── signal_types.py              # SignalType, PreSpikeState, ProposedOrder, etc.
│   │   └── __init__.py                  # Exports modeles
│   │
│   ├── event_engine/                    # Gestion evenements/catalysts
│   │   ├── event_hub.py                 # Hub central catalysts (cache 15 min)
│   │   └── nlp_event_parser.py          # Extraction NLP via Grok (18 types, 5 tiers)
│   │
│   ├── ingestors/                       # Ingestion donnees V6.1 (100% real APIs)
│   │   ├── sec_filings_ingestor.py      # SEC EDGAR 8-K + Form 4 (GRATUIT)
│   │   ├── global_news_ingestor.py      # News globale Finnhub + SEC
│   │   ├── company_news_scanner.py      # News par ticker (Finnhub company)
│   │   └── social_buzz_engine.py        # Reddit + StockTwits buzz
│   │
│   ├── processors/                      # Traitement donnees V6.1
│   │   ├── keyword_filter.py            # Pre-filtre regex rapide (1000 items/s)
│   │   ├── ticker_extractor.py          # Extraction + validation tickers
│   │   └── nlp_classifier.py            # Classification EVENT_TYPE via Grok
│   │
│   ├── risk_guard/                      # Gestion risques V8
│   │   ├── dilution_detector.py         # Detection dilution (4 tiers V8)
│   │   ├── compliance_checker.py        # Conformite exchange + delisting
│   │   ├── halt_monitor.py              # Monitoring halts (LULD, news pending)
│   │   └── unified_guard.py             # Orchestrateur risque (MIN-based V8)
│   │
│   ├── market_memory/                   # Apprentissage V7
│   │   ├── missed_tracker.py            # Track opportunites ratees
│   │   ├── pattern_learner.py           # Apprentissage patterns historiques
│   │   ├── context_scorer.py            # Scores MRP/EP contextuels
│   │   └── memory_store.py              # Persistence (JSON/SQLite)
│   │
│   ├── schedulers/                      # Orchestration scans
│   │   ├── hot_ticker_queue.py          # File prioritaire (HOT/WARM/NORMAL + TTL)
│   │   ├── scan_scheduler.py            # Scheduler dynamique (REALTIME/BATCH)
│   │   └── batch_scheduler.py           # Traitement batch hors-marche
│   │
│   ├── monitors/                        # Monitoring pipeline
│   │   └── pipeline_monitor.py          # Health checks API + alertes
│   │
│   ├── boosters/                        # Boosters additifs
│   │   ├── insider_boost.py             # Boost SEC Form 4 (+5-15%)
│   │   └── squeeze_boost.py             # Boost short interest (+5-20%)
│   │
│   ├── social_engine/                   # Moteur social
│   │   ├── grok_sentiment.py            # Sentiment via Grok API
│   │   └── news_buzz.py                 # Buzz news par ticker
│   │
│   ├── weekend_mode/                    # Mode week-end
│   │   ├── weekend_scanner.py           # Scan batch complet
│   │   ├── monday_prep.py              # Preparation watchlist lundi
│   │   ├── batch_processor.py           # Calcul lourd (backfill, training)
│   │   └── weekend_scheduler.py         # Orchestration week-end
│   │
│   └── api_pool/                        # Pool multi-cles API
│       ├── key_registry.py              # Registre cles + tiers + quotas
│       ├── pool_manager.py              # Rotation + health + cooldown
│       ├── request_router.py            # Routage par priorite (CRITICAL → BATCH)
│       └── usage_tracker.py             # Tracking usage + health score
│
├── alerts/
│   └── telegram_alerts.py              # Alertes Telegram V7 (emojis, badges, blocked)
│
├── backtests/
│   └── backtest_engine_edge.py          # Moteur backtest (V7 + legacy)
│
├── dashboards/
│   └── streamlit_dashboard.py           # Dashboard Streamlit + Plotly
│
├── monitoring/
│   └── system_guardian.py               # Guardian systeme (CPU, RAM, API health)
│
├── validation/                          # Suite de validation
│   ├── walk_forward.py                  # Walk-forward (30j fenetre, 15j step)
│   ├── regime_test.py                   # Detection regimes (bull/bear/vol)
│   ├── stress_test.py                   # Stress test resultats
│   ├── monte_carlo.py                   # Analyse Monte Carlo
│   ├── report_generator.py              # Rapport consolide
│   └── validation_engine.py             # Orchestrateur validation
│
├── tests/
│   ├── mock_data.py                     # Donnees mock (pas d'API)
│   ├── test_pipeline.py                 # Test pipeline complet
│   └── test_advanced_patterns.py        # Tests patterns + PM transition
│
└── utils/
    ├── api_guard.py                     # Safe API calls + retry + backoff
    ├── cache.py                         # TTL Cache in-memory
    ├── data_validator.py                # Validation donnees
    ├── logger.py                        # Logging rotatif
    ├── market_calendar.py               # Calendrier NYSE (2024-2027)
    └── time_utils.py                    # Sessions marche (ET timezone)
```

### 3.2 Dependances entre modules

```
Universe Loader ──→ Main Loop ──→ Process Ticker V7
                                    │
                    ┌───────────────┤
                    ▼               ▼
            Feature Engine    Monster Score V4
            (Finnhub candles) (9 composants ponderes)
                    │               │
                    └───────┬───────┘
                            ▼
                    Signal Producer V8 (Layer 1)
                    + AccelerationEngine
                    + SmallCapRadar
                            │
                            ▼
                    Pre-Halt Engine
                    Market Memory (MRP/EP)
                            │
                            ▼
                    Order Computer (Layer 2)
                            │
                            ▼
                    Risk Guard V8
                    (Dilution + Compliance + Halt)
                            │
                            ▼
                    Execution Gate (Layer 3)
                            │
                    ┌───────┴───────┐
                    ▼               ▼
            Signal Logger    Telegram Alert
            (SQLite)         (async)
```

---

## 4. MONSTER SCORE V4

### 4.1 Formule

```
base_score = (
    0.25 x event_score          +   # Catalysts (FDA, M&A, Earnings)
    0.17 x volume_spike         +   # Volume z-score normalise
    0.17 x pattern_score        +   # Patterns techniques
    0.13 x pm_transition_score  +   # Qualite transition PM→RTH
    0.10 x options_flow_score   +   # Activite options inhabituelle
    0.07 x acceleration_score   +   # [V8] Derivees + z-scores
    0.04 x momentum             +   # [Reduit V8] Velocite prix
    0.04 x squeeze              +   # Compression Bollinger
    0.03 x social_buzz_score        # [Reduit V8] Buzz social
)

final_score = clamp(base_score + beat_rate_boost + extended_hours_boost + acceleration_boost, 0, 1)
```

### 4.2 Seuils signaux

| Signal | Score | Condition supplementaire |
|--------|-------|--------------------------|
| BUY_STRONG | >= 0.80 | — |
| BUY | >= 0.65 | — |
| WATCH | >= 0.50 | — |
| EARLY_SIGNAL | >= 0.40 | + catalyst OU pre_spike != DORMANT |
| NO_SIGNAL | < 0.40 | — |

### 4.3 Boosts additifs

| Boost | Source | Max |
|-------|--------|-----|
| Beat rate | Historique earnings | +0.15 |
| Extended hours | Gap + volume AH/PM | +0.22 |
| Acceleration V8 | ACCUMULATING/LAUNCHING | +0.15 |
| Insider | SEC Form 4 | +15% base_score |
| Squeeze | Short interest | +20% base_score |

---

## 5. SIGNAL TYPES & ENUMS

### 5.1 Detection (Layer 1)

```python
SignalType:
  BUY_STRONG    # Score >= 0.80, haute confiance
  BUY           # Score >= 0.65, signal standard
  WATCH         # Score >= 0.50, a surveiller
  EARLY_SIGNAL  # Score >= 0.40, pre-spike (V8: ACCUMULATING)
  NO_SIGNAL     # Score < 0.40

PreSpikeState:
  DORMANT       # Pas d'activite
  CHARGING      # Accumulation momentum
  READY         # Pret a bouger
  LAUNCHING     # Mouvement demarre
  EXHAUSTED     # Mouvement termine

PreHaltState:
  LOW           # Pas de restriction
  MEDIUM        # Taille reduite / confirmation requise
  HIGH          # Execution bloquee/retardee
```

### 5.2 Execution (Layer 3)

```python
ExecutionStatus:
  EXECUTE_ALLOWED   # Execution complete
  EXECUTE_REDUCED   # Taille reduite (size_multiplier < 1.0)
  EXECUTE_BLOCKED   # Bloque (signal visible, execution non)
  EXECUTE_DELAYED   # En attente (marche ferme, etc.)
  ALERT_ONLY        # Notification uniquement

BlockReason:
  DAILY_TRADE_LIMIT, CAPITAL_INSUFFICIENT, POSITION_LIMIT
  DILUTION_HIGH, COMPLIANCE_HIGH, DELISTING_HIGH, PRE_HALT_HIGH
  MARKET_CLOSED, CIRCUIT_BREAKER, BROKER_DISCONNECTED
  PENNY_STOCK_RISK, MANUAL_BLOCK
```

### 5.3 Acceleration V8

```python
AccelerationState (strings, pas d'enum):
  "DORMANT"       # Pas d'activite
  "ACCUMULATING"  # Volume monte, prix stable → SIGNAL LE PLUS PRECOCE
  "LAUNCHING"     # Prix commence a bouger avec volume
  "BREAKOUT"      # Breakout confirme (prix + volume forts)
  "EXHAUSTED"     # Momentum decelerant

SmallCap Radar Phases:
  ACCUMULATING    # 5-15 min avant le mouvement
  PRE_LAUNCH      # 1-5 min avant
  LAUNCHING       # Au debut du mouvement
  BREAKOUT        # Confirme (apres +5%)
  DORMANT         # Rien de special
```

---

## 6. SOURCES DE DONNEES

### 6.1 Donnees de marche

| Source | Type | Rate Limit | Cout |
|--------|------|-----------|------|
| IBKR Level 1 | TCP socket (ib-insync) | Illimite | Abonnement |
| Finnhub REST | Polling HTTP | 60 req/min (free) | Gratuit |
| Finnhub WebSocket | Streaming | A implementer | Gratuit |

### 6.2 News & Catalysts

| Source | Contenu | Rate Limit | Cout |
|--------|---------|-----------|------|
| SEC EDGAR | 8-K filings, Form 4 | 10 req/s | Gratuit |
| Finnhub News | General + company news | 60 req/min | Gratuit |
| FDA Calendar | PDUFA, trials, conferences | Manuel | Gratuit |

### 6.3 Sentiment & NLP

| Source | Contenu | Rate Limit | Cout |
|--------|---------|-----------|------|
| Grok/xAI | Classification EVENT_TYPE | 10 req/min (free) | Gratuit |
| Reddit PRAW | Mentions WSB + subreddits | 60 req/min | Gratuit |
| StockTwits | Messages + sentiment labels | 200 req/min | Gratuit |

### 6.4 Pool API (`src/api_pool/`)

Rotation multi-cles avec priorites :
- **CRITICAL** (30% reserve) : Pre-halt, execution-blocking
- **HIGH** : Hot tickers, breaking news
- **STANDARD** : Scans normaux
- **LOW/BATCH** : Traitement hors-marche

---

## 7. RISK GUARD V8

### 7.1 Composants

| Module | Detection | Fichier |
|--------|-----------|---------|
| Dilution Detector | S-3, 424B, ATM, PIPE, toxic | `risk_guard/dilution_detector.py` |
| Compliance Checker | Deficiency, delisting, $1 rule | `risk_guard/compliance_checker.py` |
| Halt Monitor | LULD, vol spike, news pending | `risk_guard/halt_monitor.py` |
| Unified Guard | Orchestrateur central | `risk_guard/unified_guard.py` |

### 7.2 Corrections V8

| Probleme V7 | Fix V8 |
|-------------|--------|
| S-3 = CRITICAL = bloque 90j | 4 tiers : ACTIVE → SHELF_RECENT → SHELF_DORMANT → CAPACITY_ONLY |
| Multiplicatif 0.5 x 0.5 x 0.25 = 0.06 | MIN-based : min(0.5, 0.5, 0.25) = 0.25 |
| Pas de plancher → position 0.0 | Planchers : penny 0.10, standard 0.25 |
| Top gainers bloques par dilution | Momentum override : catalyst + vol + prix = -50% penalty |

---

## 8. SCHEDULING & SCAN INTERVALS

### 8.1 Intervalles

| Scan | Intervalle | Contexte |
|------|-----------|----------|
| Global news | 180s (3 min) | scan_scheduler.py |
| Hot tickers | 90s (1.5 min) | Tickers prioritaires |
| Warm tickers | 300s (5 min) | Tickers secondaires |
| Universe rotation | 600s (10 min) | 3 tickers aleatoires/cycle |
| Social buzz | 600s (10 min) | Reddit + StockTwits |
| SmallCap Radar | 5s | Buffer local (pas d'API) |

### 8.2 Hot Ticker Queue

| Priorite | TTL | Intervalle scan |
|----------|-----|-----------------|
| HOT | 3600s (1h) | 90s |
| WARM | 1800s (30 min) | 300s |
| NORMAL | 900s (15 min) | 600s |

Triggers pour devenir HOT :
- Pre-Spike Radar >= 2 signaux
- Catalyst detecte dans scan global
- Repeat Gainer score > 0.7
- Mouvement PM/AH > 5%
- Buzz social acceleration > 3x

---

## 9. CONFIGURATION CLES (`config.py`)

### 9.1 Feature Flags

| Flag | Default | Ligne | Description |
|------|---------|-------|-------------|
| `USE_V7_ARCHITECTURE` | True | 357 | Pipeline V7 unifie |
| `USE_IBKR_DATA` | False | 17 | Donnees IBKR (requiert Gateway) |
| `ENABLE_ACCELERATION_ENGINE` | True | 441 | V8 derivees + z-scores |
| `ENABLE_SMALLCAP_RADAR` | True | 457 | V8 radar anticipatif |
| `ENABLE_PRE_HALT_ENGINE` | True | 372 | Evaluation risque halt |
| `ENABLE_RISK_GUARD` | True | 414 | Guard risque unifie |
| `ENABLE_MARKET_MEMORY` | True | 429 | Apprentissage MRP/EP |
| `ENABLE_OPTIONS_FLOW` | True | 236 | Analyse options |
| `ENABLE_SOCIAL_BUZZ` | True | 244 | Tracking buzz social |
| `ENABLE_NLP_ENRICHI` | True | 328 | NLP avance (Grok) |
| `ENABLE_CATALYST_V3` | True | 282 | Score catalyst V3 |
| `ENABLE_PRE_SPIKE_RADAR` | True | 304 | Radar pre-spike |

### 9.2 Limites Execution

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| `DAILY_TRADE_LIMIT` | 5 | Trades max par jour |
| `MAX_POSITION_PCT` | 10% | Max capital par position |
| `MAX_TOTAL_EXPOSURE` | 80% | Exposition totale max |
| `MIN_ORDER_USD` | $100 | Ordre minimum |
| `RISK_BUY` | 2% | Risque par trade standard |
| `RISK_BUY_STRONG` | 2.5% | Risque par trade fort |

### 9.3 Stops

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| `ATR_MULTIPLIER_STOP` | 2.0 | Stop loss en ATR |
| `ATR_MULTIPLIER_TRAIL` | 1.5 | Trailing stop en ATR |

---

## 10. ETAT ACTUEL & PROBLEMES CONNUS

### 10.1 Problemes identifies (PLAN_CORRECTION_COVERAGE.md)

| ID | Probleme | Severite | Statut |
|----|----------|----------|--------|
| P1 | Finnhub 60 req/min → 6% couverture/cycle | CRITIQUE | A corriger |
| P2 | Feature Engine exige 20 bars minimum | HAUT | A corriger |
| P3 | NO_SIGNAL si score < 0.40 sans catalyst | HAUT | A corriger |
| P4 | SmallCap Radar filtre volume 500K redondant | MOYEN | A corriger |
| P5 | Batch limits durs dans anticipation_engine | MOYEN | A corriger |
| P6 | Aucun flux streaming (tout est polling) | CRITIQUE | A corriger |
| P7 | Acceleration Engine : 3+ samples minimum | MOYEN | A corriger |
| P8 | Alert cooldown fixe 2 min | MOYEN | A corriger |
| P9 | Hot ticker TTL 1h expire | MOYEN | A corriger |
| P10 | Aucune source externe top gainers | MOYEN | A corriger |

### 10.2 Corrections prevues (PLAN_CORRECTION_COVERAGE.md)

| ID | Correction | Phase | Statut |
|----|-----------|-------|--------|
| C1 | WebSocket Screener (Finnhub WS) | Phase 3 | Planifie |
| C2 | Supprimer filtres SmallCap Radar | Phase 1 | Planifie |
| C3 | Score plancher adaptatif (vol z > 2.5) | Phase 2 | Planifie |
| C4 | Warm-up accelere (2 samples) | Phase 4 | Planifie |
| C5 | Alert cooldown adaptatif (15-120s) | Phase 1 | Planifie |
| C6 | Hot Ticker TTL 4h + auto-renewal | Phase 1 | Planifie |
| C7 | Relever batch limits anticipation | Phase 2 | Planifie |
| C8 | Source externe top gainers (IBKR+Yahoo) | Phase 4 | Planifie |
| C9 | Feature Engine 5 bars minimum | Phase 1 | Planifie |

### 10.3 Fichiers de reference

| Document | Contenu |
|----------|---------|
| `PLAN_TRANSFORMATION_V8.md` | Plan transformation V7 → V8 |
| `PLAN_CORRECTION_COVERAGE.md` | Corrections couverture 100% top gainers |
| `CLAUDE.md` | Ce fichier — reference systeme |

---

## 11. CONVENTIONS DE DEVELOPPEMENT

### 11.1 Architecture

- **Detection JAMAIS bloquee** — seule l'execution a des limites
- **Additif, pas multiplicatif** — les boosts s'ajoutent, les penalites utilisent MIN
- **Fallback systematique** — IBKR → Finnhub → Cache → Default
- **Singleton pattern** — `get_ibkr()`, `get_v7_state()`, `get_signal_producer()`
- **Async + Thread-safe** — asyncio pour I/O, threading pour background

### 11.2 Sources de donnees

- **100% APIs reelles** — jamais de simulation LLM pour les donnees
- **Grok = classification UNIQUEMENT** — pas de sourcing de donnees
- **SEC EDGAR = gratuit et illimite** — privilegier pour les filings
- **Finnhub free tier** — gerer les 60 req/min via pool_manager

### 11.3 Scoring

- **Monster Score** : composant ponderes normalises 0-1 + boosts additifs
- **Catalyst Score** : 5 tiers x 18 types + recency decay + quality + confluence
- **Signal Producer** : score ajuste + boosts V8 (acceleration, pre-spike, repeat gainer)

### 11.4 Risk

- **V8 MIN-based** — `min(risk1, risk2, risk3)` pas `risk1 * risk2 * risk3`
- **Momentum override** — catalyst + volume + prix = penalty reduite 50%
- **Planchers** — penny stock 0.10, standard 0.25 (jamais 0.0 sauf toxique)
- **Block UNIQUEMENT** sur : delisting, halt actif, toxic financing, block manuel

### 11.5 Tests & Validation

- `tests/test_pipeline.py` — test complet avec mock data
- `validation/validation_engine.py` — walk-forward, regime, stress, Monte Carlo
- `daily_audit.py` — hit rate, lead time, miss rate quotidiens
- `weekly_deep_audit.py` — analyse profonde hebdomadaire

### 11.6 Deploiement

- **Serveur** : Hetzner CX33 (headless Linux 4.4.0)
- **IBKR** : Optionnel (USE_IBKR_DATA=False par defaut)
- **Fallback** : Finnhub REST si pas d'IBKR
- **Alertes** : Telegram (token + chat_id via .env)
- **Logs** : `data/logs/` avec rotation
- **Persistance** : SQLite (`data/*.db`) + JSON + CSV

---

## 12. GLOSSAIRE

| Terme | Definition |
|-------|-----------|
| **Monster Score** | Score composite 0-1 combinant 9 composants ponderes |
| **Signal Producer** | Layer 1 — detection illimitee de signaux |
| **Order Computer** | Layer 2 — calcul systematique d'ordres |
| **Execution Gate** | Layer 3 — seule couche avec limites |
| **Acceleration Engine** | V8 — detection par derivees (velocite + acceleration) |
| **SmallCap Radar** | V8 — radar anticipatif (ACCUMULATING → BREAKOUT) |
| **Ticker State Buffer** | V8 — ring buffer 120 snapshots par ticker |
| **Hot Ticker Queue** | File prioritaire avec TTL et promotion/demotion |
| **Anticipation Engine** | Scan multi-source (IBKR + SEC + Finnhub + NLP) |
| **Market Memory** | Apprentissage : MRP (recovery potential), EP (edge probability) |
| **Risk Guard** | Evaluation risque : dilution + compliance + halt |
| **Unified Guard** | Orchestrateur risque V8 (MIN-based, momentum override) |
| **Pre-Halt Engine** | Detection risque halt (LULD, vol spike, news) |
| **Catalyst Score V3** | Score evenement : 5 tiers, 18 types, temporal decay |
| **Beat Rate** | Probabilite de beat earnings basee sur historique |
| **PM Transition** | Qualite de la transition pre-market → RTH |
| **Gap Zone** | Classification gap : NEGLIGIBLE / EXPLOITABLE / EXTENDED / OVEREXTENDED |
| **RTH** | Regular Trading Hours (09:30-16:00 ET) |
| **PDUFA** | Prescription Drug User Fee Act — date limite decision FDA |
