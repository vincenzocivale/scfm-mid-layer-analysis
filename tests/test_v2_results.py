import pandas as pd

from scfm_eval.v2.compare import add_final_layer_comparison


def test_final_layer_comparison_respects_metric_direction():
    df = pd.DataFrame([
        dict(dataset="d", model="m", model_size="", task="t", representation="model_layer",
             layer=0, n_layers_total=2, relative_depth=0.0, metric="acc", value=0.8,
             higher_is_better=True, n_obs=10, split="eval", notes=""),
        dict(dataset="d", model="m", model_size="", task="t", representation="model_layer",
             layer=1, n_layers_total=2, relative_depth=1.0, metric="acc", value=0.7,
             higher_is_better=True, n_obs=10, split="eval", notes=""),
        dict(dataset="d", model="m", model_size="", task="t", representation="model_layer",
             layer=0, n_layers_total=2, relative_depth=0.0, metric="error", value=0.2,
             higher_is_better=False, n_obs=10, split="eval", notes=""),
        dict(dataset="d", model="m", model_size="", task="t", representation="model_layer",
             layer=1, n_layers_total=2, relative_depth=1.0, metric="error", value=0.3,
             higher_is_better=False, n_obs=10, split="eval", notes=""),
    ])
    out = add_final_layer_comparison(df)
    first_acc = out[(out.metric == "acc") & (out.layer == 0)].iloc[0]
    first_err = out[(out.metric == "error") & (out.layer == 0)].iloc[0]
    assert first_acc.signed_gain_vs_final > 0
    assert first_err.signed_gain_vs_final > 0
