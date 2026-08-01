# Traffic Sign Recognition Engine

A **dataset-driven image retrieval engine** for traffic signs. It is *not* a
fixed-class classifier — there is no 300-way softmax and **no retraining** is
ever required. Recognition accuracy improves simply by adding more sample
images to the dataset.

```
image_path
   -> YOLO detection        (where are the signs? single "traffic_sign" class)
   -> crop                   (one crop per detection)
   -> DINOv2 embedding       (L2-normalized feature vector)
   -> FAISS memory search    (top-K nearest neighbours)
   -> voting                 (majority + average + weighted similarity)
   -> shape / color / blur    validation
   -> rule engine            (fuse all signals -> 0..100 score)
   -> Prediction
```

The **Traffic Sign Memory** (FAISS index + metadata) is the single source of
truth. YOLO is trusted only for *location*; the sign *identity* always comes
from the memory.

## Project layout

```
ai/
  config.py                 EngineConfig + RuleWeights (all tunables)
  factory.py                Composition root (dependency injection)
  __main__.py               CLI: build / append / recognize / benchmark
  detector/                 YOLO detector -> single "traffic_sign" class
  cropper/                  Detection -> crop image (+ saved path)
  embedding/                DINOv2 encoder + L2 normalizer + engine
  memory/                   FAISS index + 2-layer online-learning memory
  matcher/                  Similarity search + voting engine
  validator/                Shape / color / blur validators
  rule_engine/              Weighted 0..100 validation score
  pipeline/                 RecognitionPipeline (orchestration)
  models/                   BoundingBox / Embedding / Prediction value objects
  utils/                    Logging (Loguru) / device / benchmark / image IO
```

Every stage sits behind an interface and is wired via `ai/factory.py`, so any
component (encoder, index, detector, validator) can be replaced independently.

## Installation

The engine's heavy dependencies live in the `ai` optional-dependency group:

```bash
# from backend/
uv sync --extra ai --extra dev
# or: pip install -e ".[ai,dev]"
```

On an Apple Silicon MacBook the compute backend is auto-detected in this
order: **CUDA → MPS → CPU**. Set `EngineConfig.device` to force one.

## How the memory works

- `dataset/<sign_id>/*.png` — each **folder name is the official sign id**.
  Nothing is hardcoded; the recognizable set is whatever exists on disk.
- `build()` scans the dataset, embeds every image with DINOv2, and stores:
  - the L2-normalized vector in a FAISS `IndexFlatIP` (exact cosine search),
  - a metadata entry `{vector_id, sign_id, image_path, embedding_model, created_time}`.
- Persisted artifacts:
  - `memory/vectors.faiss` — the FAISS index,
  - `memory/metadata.json` — the aligned metadata (entry *i* ↔ vector *i*).
- On startup the pipeline **loads** the persisted memory if present and only
  **builds** it when missing — the index is not rebuilt every run.

## How FAISS works here

Vectors are L2-normalized, so inner product equals cosine similarity. The
engine uses `IndexFlatIP` for exact top-K search. A query crop is embedded,
searched against the index, and the returned positions map back to metadata
entries to recover `(sign_id, image_path, similarity)`.

`MemoryManager` public API: `build()`, `load()`, `save()`, `append()`,
`search()`, `rebuild()`.

## How to add new traffic signs

1. Drop images into a new (or existing) folder: `dataset/<sign_id>/*.png`.
2. Append them to the memory **without a full rebuild**:

```bash
python -m ai append dataset/145 145
```

or programmatically:

```python
from ai.factory import build_memory_manager
memory = build_memory_manager()
memory.load()
memory.append("dataset/145/1.png", "145")   # appends one vector + metadata, then saves
```

Only new vectors are appended — existing vectors are untouched and no model
is retrained.

## How to update / rebuild memory

```bash
python -m ai build          # full rebuild from dataset/ (clears existing memory)
```

Use `build`/`rebuild` only when you want to regenerate everything (e.g. after
changing the embedding model). Day-to-day additions should use `append`.

## Recognize an image

```bash
python -m ai recognize /path/to/image.jpg
```

Programmatic use (the backend passes a local image path):

```python
from ai import build_pipeline
pipeline = build_pipeline()                 # loads/builds memory, wires stages
result = pipeline.run("/path/to/image.jpg")
for prediction in result.predictions:
    print(prediction.traffic_sign_id, prediction.validation_score)
```

Each `Prediction` contains `traffic_sign_id`, `similarity`,
`validation_score` (0..100), `bbox`, `crop_path`, `shape`, `colors`,
`blur_score`, `yolo_confidence`, `voting_confidence` and `top_k_matches`.

## How to benchmark

Per-stage wall-clock timings (milliseconds) are captured on every run and
returned in `RecognitionResult.timings_ms` (`yolo`, `embedding`, `faiss`,
`validation`, `total`):

```bash
python -m ai benchmark /path/to/image.jpg
```

## Tests

```bash
pytest tests_ai
```

The engine tests use numpy-only fakes (a brute-force index and a
deterministic encoder), so they run without torch / faiss / ultralytics /
OpenCV installed and are skipped automatically if numpy is unavailable.

## Online Memory Learning

The memory is **two layers**:

- **Permanent Memory** — verified vectors; the only layer recognition
  searches (`vectors.faiss` + `metadata.json`).
- **Candidate Memory** — unverified embeddings discovered at inference time
  (`candidates.json`).

### Discovery (candidate append)

After each prediction the pipeline feeds the result into the learning hook.
An embedding is appended to **Candidate Memory only** when it passes the
gate:

```
validation_score >= 90  AND  similarity >= 0.95
```

Nothing is ever written straight to Permanent Memory. Every candidate
observation stores `traffic_sign_id`, `embedding`, `gps`, `timestamp`,
`device_id`, `confidence`, `validation_score`, `blur_score`, `image_hash`
and `image_path`.

Supply capture context per run:

```python
from ai import build_pipeline
from ai.memory.observation import ObservationContext

pipeline = build_pipeline()
pipeline.run("/path/to/image.jpg", context=ObservationContext(device_id="cam-01", gps=(10.77, 106.70)))
```

### Duplicate detection

Before storing, Candidate Memory searches itself. Cosine similarity
`> 0.995` is treated as the same frame and **ignored**. Similar-but-distinct
sightings (`>= 0.95`) merge into the existing candidate cluster — this is
what advances the promotion counters.

### Promotion (candidate → permanent)

`PromotionEngine` promotes a candidate when **any** rule holds:

1. observed on `>= 5` different days, **or**
2. observed by `>= 3` different devices, **or**
3. human-verified (`OnlineMemoryManager.verify(candidate_id)`).

Promotion copies the candidate's representative (mean) embedding into
Permanent Memory.

### Pruning (Memory Optimizer)

`prune()` clusters same-sign permanent vectors by cosine similarity
(`>= 0.98` = redundant) and keeps one representative per cluster.

### Versioning / rollback / export / import

Mutating operations (`promote`, `prune`, `import_`) snapshot the memory
first, so `rollback()` undoes the last change (or `rollback(version=n)`
restores a specific snapshot). `export(path.zip)` bundles the memory;
`import_(path.zip)` restores it.

### `OnlineMemoryManager` API

`append()`, `search()`, `promote()`, `rollback()`, `prune()`, `export()`,
`import_()`, `statistics()` (plus `verify()` and `observe()`).

```python
from ai.factory import build_online_memory

memory = build_online_memory()
memory.ensure_ready()
memory.promote()                 # candidates -> permanent
memory.prune()                   # drop redundant permanent vectors
print(memory.statistics().model_dump())
memory.export("backup.zip")
```

`statistics()` reports number of vectors, candidates, promoted count, memory
size (bytes), duplicate ratio and a recognition-accuracy estimate.

### CLI

```bash
python -m ai promote            # promote eligible candidates
python -m ai prune              # remove redundant permanent vectors
python -m ai stats              # print memory statistics
python -m ai export backup.zip  # export the memory bundle
python -m ai import backup.zip  # import a memory bundle
```
