import sys

sys.path.extend([".", ".."])
import argparse
import csv
from sys import argv

import numpy as np
import pandas as pd
from monty.serialization import loadfn
from pymatgen.core import Composition
from tqdm import tqdm

from skipatom import (
    atom_vectors_from_csv,
    max_pool,
    mean_pool,
    species_vectors_from_csv,
    sum_pool,
)

try:
    import cPickle as pickle
except ImportError:
    import pickle

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skipatom")
POOLINGS = ["sum", "mean", "max"]
SUM_POOLING = POOLINGS[0]
MEAN_POOLING = POOLINGS[1]
MAX_POOLING = POOLINGS[2]


def featurise(comp, dictionary, embeddings, pool):
    comps_featurised = []
    formulas = []
    for c in tqdm(comp, desc="Featurising the compositions"):
        if any([str(e) not in dictionary for e in c.elements]):
            print([str(e) not in dictionary for e in c.elements])
            raise Exception(f"{c.reduced_formula} contains unsupported atoms/species")
        comps_featurised.append(pool(c, dictionary, embeddings))
        formulas.append(c.reduced_formula)
    return np.array(comps_featurised), formulas


if __name__ == "__main__":
    supported_tasks = [
        "formation_energy_per_atom",
        "band_gap",
        "is_metal",
        "is_magnetic",
    ]
    parser = argparse.ArgumentParser(
        description="Create an Materials Project dataset for training and evaluation with ElemNet-like models."
    )
    parser.add_argument(
        "--data",
        nargs="?",
        required=True,
        type=str,
        help="path to MP .json.gz file; this file must contain at least the following "
        "columns: 'composition', 'oxi_composition'",
    )
    parser.add_argument(
        "--task",
        nargs="?",
        required=True,
        type=str,
        help="The name of the task to generate the dataset for. Must be one of the following: "
        "'formation_energy_per_atom', 'band_gap', 'is_metal', 'is_magnetic' ",
    )

    parser.add_argument(
        "--out",
        nargs="?",
        required=True,
        type=str,
        help="path to the output file; a .pkl extension should be used (the file will not be gzipped)",
    )

    parser.add_argument(
        "--atom_vectors",
        required=False,
        type=str,
        help="path to the file containing the atom vectors,",
    )
    parser.add_argument(
        "--species_vectors",
        required=False,
        type=str,
        help="path to the file containing the species vectors,",
    )
    parser.add_argument(
        "--pooling",
        required=True,
        choices=POOLINGS,
        help="the type of pooling operation to use",
    )
    args = parser.parse_args()

    pool = None
    if args.pooling == SUM_POOLING:
        pool = sum_pool
    elif args.pooling == MEAN_POOLING:
        pool = mean_pool
    elif args.pooling == MAX_POOLING:
        pool = max_pool
    else:
        raise Exception(f"unsupported pooling: {args.pooling}")

    print(f"Loading data from {args.data}...")
    df = loadfn(args.data)

    # Convert to pymatgen Composition
    if args.atom_vectors:
        composition = df["composition"].to_list()
        compositions = [Composition.from_dict(c) for c in composition]
        dictionary, embeddings = atom_vectors_from_csv(args.atom_vectors)
    elif args.species_vectors:
        oxi_compositon = df["oxi_composition"].to_list()
        compositions = [Composition.from_dict(c) for c in oxi_compositon]
        dictionary, embeddings = species_vectors_from_csv(args.species_vectors)
    else:
        raise Exception(f"Supply either '--atom_vectors' or '--species_vectors'...")

    # Featurise the compositions
    print(f"Featurising the compositions")

    comps, formulas = featurise(compositions, dictionary, embeddings, pool)
    # Get the task data from the dataframe
    targets = df[args.task].to_list()

    dataset = []

    for comp, target in tqdm(
        zip(comps, targets), desc="Joining the targets and featurised compositions"
    ):
        dataset.append([comp, target])

    print(f"The dataset has {len(dataset)} rows")
    print(f"writing dataset to {args.out}")
    with open(args.out, "wb") as f:
        pickle.dump((formulas, dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    print("done.")
