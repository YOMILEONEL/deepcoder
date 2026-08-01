import argparse
import json
import os
import random

import numpy as np
import tensorflow as tf

from deepcoder.nn.model import get_model, get_XY

def set_seed(seed):
    """Makes weight init and training deterministic across runs.

    Without this, model.fit() picks different random weight
    initializations every run, so accuracy after training varies
    (observed 66-73% for the same gas=1000 config across runs).
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--val_split', type=float)
    parser.add_argument('-E', type=int,
        default=2, help='embedding dimension')
    parser.add_argument('--nb_inputs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42,
        help='random seed for reproducible weight init/training')
    args = parser.parse_args()

    set_seed(args.seed)

    problems = json.loads(open(args.infile).read())
    X, y = get_XY(problems, args.nb_inputs)
    model = get_model(args.nb_inputs, args.E)
    model.fit(X, y, epochs=args.epochs, validation_split=args.val_split)
    print('saving model to ', args.outfile)
    model.save(args.outfile)

if __name__ == '__main__':
    main()