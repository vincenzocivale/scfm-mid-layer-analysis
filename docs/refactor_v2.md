# Refactor v2: experiment architecture

## Goal

The repository should answer one scientific question cleanly:

> For a given single-cell foundation model, when does an intermediate
> representation preserve or expose more useful biological information than
> the model's final representation?

The implementation therefore treats the atomic experimental unit as:

`dataset × model variant × layer × benchmark × metric`

and never infers those fields back from filenames.

## Problems in the legacy pipeline

1. Chunk files are full AnnData objects and duplicate expression/metadata.
2. The grid driver later merges the chunks into a monolithic h5ad, partially
   defeating the reason for sharding.
3. The matrix embedded in each output h5ad is the matrix after model-specific
   preprocessing. A downstream biological reference can therefore become
   coupled to the foundation model being evaluated.
4. Result identity is encoded in filenames, and the aggregator reverse-parses
   dataset/model/task from those filenames.
5. Benchmark outputs have heterogeneous schemas. Baselines are represented by
   sentinel layer values, making comparison logic brittle.
6. "Does an intermediate layer beat the final layer?" is not a first-class
   analysis; it has to be reconstructed later.

## v2 contracts

### 1. Canonical biological dataset

The original h5ad is the single source of truth for:

- obs annotations;
- raw/count expression;
- cell identity and ordering;
- task-specific biological references.

No model-specific preprocessing is persisted back into this dataset.

### 2. Embedding store

Each `dataset × model variant` produces:

```text
artifacts/embeddings/<dataset>/<model-variant>/
  manifest.json
  shards/
    000000000000_000000020000.h5
    000000020000_000000040000.h5
    ...
```

Each shard stores only:

- `cell_id`;
- `layers/<layer_idx>` arrays.

There is no merge stage. Benchmarks load one layer across shards on demand.

The manifest is authoritative for:

- dataset/model/model size;
- total layer count;
- hidden dimension;
- pooling/input semantics;
- completed ranges;
- layer coverage;
- completion state.

Extraction is resumable and shard writes are atomic.

### 3. Metric schema

Every benchmark writes one canonical long-format `metrics.csv`:

```text
dataset
model
model_size
task
representation
layer
n_layers_total
relative_depth
metric
value
higher_is_better
n_obs
split
notes
```

`representation=model_layer` has an integer layer.

Future baselines should use named representations such as:

- `pca_hvg`
- `raw_expression`
- `random_projection`

and leave `layer` empty. Do not encode baselines as `layer=-1`.

### 4. Final-layer comparison

`scripts/summarize_results_v2.py` adds:

- `final_value`
- `raw_delta_vs_final`
- `signed_gain_vs_final`
- `relative_gain_vs_final`
- `beats_final`

`higher_is_better` controls the sign, so positive `signed_gain_vs_final`
always means "the intermediate layer is better than the final layer".

This makes the central paper question queryable with a single filter.

## Migration

The legacy pipeline can remain temporarily for reproducibility, but new
experiments should use v2.

Recommended migration order:

1. Validate v2 on one existing brain/scFoundation run.
2. Compare v2 classification numbers against legacy output.
3. Port pseudotime evaluator to return `MetricRecord` objects.
4. Port perturbation evaluator to return `MetricRecord` objects.
5. Add shared baseline evaluators.
6. Add a study/grid launcher that expands `config/study_v2.yaml`.
7. Once parity is established, move legacy scripts under `legacy/` and stop
   versioning generated CSV outputs.

## Scientific extensions enabled by this layout

The same result table can support:

- best layer per task and model;
- gain over final layer;
- optimal relative depth across architectures;
- cross-dataset stability of the optimal layer;
- Pareto analysis across biological tasks;
- layer-wise rank aggregation;
- uncertainty intervals from resampling without changing the artifact format;
- new FM backends without touching benchmark storage.
