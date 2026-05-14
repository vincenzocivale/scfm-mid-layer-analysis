"""Helpers to introspect foundation model metadata without running extraction."""
import argparse
import sys

from scfm_eval.embedders import get_embedder, list_embedders


def get_model_num_layers(model_name: str, model_size=None) -> int:
    """Number of transformer blocks for a given (model, size)."""
    kwargs = {}
    if model_size is not None:
        kwargs['model_size'] = model_size
    embedder = get_embedder(model_name, **kwargs)
    return len(embedder.get_all_layer_indices())


def main():
    parser = argparse.ArgumentParser(description="Print n_layers_total for a foundation model.")
    parser.add_argument('--model', required=True, choices=list_embedders())
    parser.add_argument('--model-size', '--tahoe_size', dest='model_size', default=None)
    args = parser.parse_args()

    try:
        print(get_model_num_layers(args.model, model_size=args.model_size))
    except Exception as e:
        print(f"Error getting number of layers for model {args.model}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
