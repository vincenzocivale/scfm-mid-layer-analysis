import sys
import os
import argparse

# Add parent directory to path to allow importing models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add models directory to path for embedder imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))


def get_model_num_layers(model_name: str, tahoe_size=None) -> int:
    """
    Returns the number of layers for a given model.
    """
    if model_name == 'tahoe':
        from tahoe_embedder import TahoeEmbedder
        if tahoe_size is None:
            embedder = TahoeEmbedder(model_size="70m")
        else:
            embedder = TahoeEmbedder(model_size=tahoe_size)
    elif model_name == 'scfoundation':
        from scfoundation_embedder import scFoundationEmbedder
        embedder = scFoundationEmbedder()
    elif model_name == 'scgpt':
        from scgpt_embedder import scGPTEmbedder
        embedder = scGPTEmbedder()
    else:
        raise ValueError(f"Model '{model_name}' not supported or not found.")
    
    return len(embedder.get_all_layer_indices())

def main():
    parser = argparse.ArgumentParser(description="Get the number of layers for a given model.")
    parser.add_argument('--model', required=True, choices=['tahoe', 'scfoundation', 'scgpt'], 
                        help="Name of the model to query (e.g., 'tahoe', 'scfoundation', 'scgpt').")
    parser.add_argument('--tahoe_size', required=False, choices=['1b', '70m', '3b'])
    args = parser.parse_args()

    try:
        if args.tahoe_size:
            num_layers = get_model_num_layers(args.model, tahoe_size=args.tahoe_size)
        else:
            num_layers = get_model_num_layers(args.model)
        print(num_layers)
    except Exception as e:
        print(f"Error getting number of layers for model {args.model}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

