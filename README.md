# Glyphh Marketing — Campaign & Content Intelligence

Encodes campaigns, posts, and audiences as searchable HDC glyphs with continuous performance vectors. Supports semantic search ("find campaigns similar to our product launch"), cross-type matching, and real-time performance tracking via Pipedream webhooks through the listener endpoint.

Built on [**Glyphh Ada 1.1**](https://www.glyphh.ai/products/runtime) · **[Docs →](https://glyphh.ai/docs)** · **[Glyphh Hub →](https://glyphh.ai/hub)**

---

## Getting Started

### 1. Install the Glyphh CLI

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install with runtime dependencies (includes FastAPI, SQLAlchemy, pgvector)
pip install 'glyphh[runtime]'
```

### 2. Clone and start the model

This model requires PostgreSQL + pgvector for similarity search.

```bash
git clone https://github.com/glyphh-ai/model-marketing.git
cd model-marketing

# Start the Glyphh shell (prompts login on first run)
glyphh

# Inside the shell:
# glyphh> docker init            # generates docker-compose.yml + init.sql
# glyphh> exit

# Start PostgreSQL + pgvector and the Glyphh runtime
docker compose up -d --wait
```

This starts:
- **PostgreSQL 16 + pgvector** on port 5432 (with HNSW indexing)
- **Glyphh Runtime** on port 8002 (auto-deploys the model via volume mount)

### 3. Deploy the model

```bash
glyphh
# glyphh> model package                              # build .glyphh package
# glyphh> model deploy model-marketing.glyphh        # deploy to runtime
```

### 4. Query the model

```bash
# Text queries (intent extraction → HDC encode → cosine search)
# glyphh> chat "find campaigns similar to our product launch"
# glyphh> chat "what content works for technical audiences"
# glyphh> chat "compare email vs social campaign performance"
# glyphh> chat "show high-converting webinar campaigns"
```

## How It Works

Marketing follows the same bifurcated architecture as all Glyphh models: **HDC handles deterministic parsing, LLM handles generation**.

```
QUERY INPUT (NL or data record)
    ↓
Intent Extraction (intent.py)
    ↓ action, target, domain, keywords
Encoder (4-layer HDC binding)
    ↓ identity + strategy + content + performance
Bundle → 2,000-dim bipolar vector
    ↓
Searchable, comparable, updatable
```

**Three glyph types, one encoder config:**
- **Campaigns** — goal, audience, platforms, themes, aggregate performance
- **Posts** — content type, tone, keywords, per-post engagement/conversion
- **Audiences** — segment interests, preferred platforms, response patterns

**Two matching paths:**
- **NL queries** (text, no metrics) → match on identity + strategy + content layers
- **Data records** (metrics + text) → match on all layers including performance

## Encoded Layers

| Layer | Weight | Roles | Encoding |
|-------|--------|-------|----------|
| **identity** | 0.10 | entity_type, entity_id | categorical, key_part |
| **strategy** | 0.25 | goal, audience_segment, platform, content_type | categorical, bag_of_words, bag_of_words, categorical |
| **content** | 0.30 | themes, tone, keywords | bag_of_words, categorical, bag_of_words |
| **performance** | 0.35 | engagement_metrics, conversion_metrics, reach_metrics | continuous (5-dim), continuous (5-dim), continuous (4-dim) |

**Three encoding types:**
- **Categorical** — Exact symbol match. Same value = identical vector.
- **Bag of words** — Split into words, encode each, bundle. Shared words = shared signal.
- **Continuous** — `ContinuousProjector` (random projection + sign quantization). Similar float vectors → similar bipolar vectors.

## Performance Metrics (Continuous Vectors)

Performance is encoded as three continuous vectors, updated in real-time via the listener endpoint:

| Role | Dimensions | Fields |
|------|-----------|--------|
| engagement_metrics | 5 | likes, shares, comments, saves, reactions |
| conversion_metrics | 5 | clicks, signups, purchases, trials, demos |
| reach_metrics | 4 | impressions, unique_views, frequency, growth_rate |

ContinuousProjector projects these float vectors into bipolar HDC space via deterministic random projection + sign quantization. Similar performance profiles → similar bipolar vectors → high cosine similarity.

**Note:** ContinuousProjector is scale-invariant (sign quantization). Vectors that differ only in magnitude project identically — similarity is based on the *pattern* of metrics, not absolute values.

## Real-Time Updates via Pipedream

Performance metrics are updated in real-time through Pipedream webhooks → runtime listener endpoint:

```
Pipedream Webhook (click/conversion event)
    ↓
POST /{org_id}/marketing/listener
    Body: {
        "records": [{
            "name": "product_launch_q1",
            "attributes": {
                "entity_type": "campaign",
                "entity_id": "product_launch_q1",
                "goal": "conversion",
                "audience_segment": "technical enterprise",
                "platform": "twitter linkedin",
                "themes": "product launch api integration",
                "tone": "professional",
                "keywords": "product launch enterprise api integration",
                "engagement_metrics": [1250, 340, 89, 45, 156],
                "conversion_metrics": [2100, 180, 42, 95, 12],
                "reach_metrics": [45000, 12000, 3.2, 0.15]
            }
        }]
    }
    ↓
Runtime encodes → stores glyph with temporal timestamp
```

Each update creates a new glyph version (temporal tracking via `entity_id` key_part). Temporal deltas capture what changed between updates.

## Model Structure

```
marketing/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # 4-layer encoder config, thresholds
├── encoder.py             # ENCODER_CONFIG + encode_query + entry_to_record
├── intent.py              # marketing NL intent extraction (~40 verbs, ~30 targets, 6 domains)
├── data/
│   └── exemplars.jsonl    # seed campaigns, posts, audiences (22 exemplars)
├── tests/
│   ├── conftest.py        # shared fixtures (encoder, exemplar glyphs)
│   ├── test_encoding.py   # config validation, encoding pipeline, determinism
│   ├── test_similarity.py # cross-type matching, performance proximity, query-to-exemplar
│   └── test_intent.py     # action/target/domain/keyword extraction
├── requirements.txt       # no special deps (pure HDC)
└── README.md
```

## Testing

```bash
# Run from the marketing/ directory
PYTHONPATH="../../glyphh-runtime:.:" python -m pytest tests/ -v

# Or specific test files
PYTHONPATH="../../glyphh-runtime:.:" python -m pytest tests/test_encoding.py -v
PYTHONPATH="../../glyphh-runtime:.:" python -m pytest tests/test_similarity.py -v
PYTHONPATH="../../glyphh-runtime:.:" python -m pytest tests/test_intent.py -v
```

The test suite runs entirely on synthetic data — no external dependencies needed. 65 tests covering:
- **test_encoding.py** — config validation, encode_query, entry_to_record, determinism
- **test_similarity.py** — cross-type matching, performance patterns, layer-level, query-to-exemplar
- **test_intent.py** — action/target/domain/keyword/modifier extraction

## Data Format

Exemplars in `data/exemplars.jsonl` are pre-built marketing entities:

```json
{
    "entity_type": "campaign",
    "entity_id": "product_launch_q1",
    "goal": "conversion",
    "audience_segment": "technical enterprise",
    "platform": "twitter linkedin blog",
    "content_type": "none",
    "themes": "product launch api integration developer platform",
    "tone": "professional",
    "keywords": "product launch enterprise api integration developer platform new release",
    "engagement_metrics": [1250, 340, 89, 45, 156],
    "conversion_metrics": [2100, 180, 42, 95, 12],
    "reach_metrics": [45000, 12000, 3.2, 0.15]
}
```

## Use Cases

### Find Similar Campaigns
```bash
# glyphh> chat "find campaigns similar to our product launch"
```

### Audience-Content Matching
```bash
# glyphh> chat "what content works for technical audiences"
```

### Platform Performance
```bash
# glyphh> chat "show me email campaign performance"
```

### Cross-Type Search
```bash
# glyphh> chat "compare email vs social performance"
```
