import sys

sys.path.extend([".", ".."])
import argparse
import logging
import os
import shutil
from sys import argv
from time import time

import numpy as np
import pandas as pd
from keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint
from pymatgen.core import Composition
from sklearn.model_selection import KFold, train_test_split

from skipatom import ElemNet, ElemNetClassifier, max_pool, mean_pool, sum_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skipatom")

POOLINGS = ["sum", "mean", "max"]
SUM_POOLING = POOLINGS[0]
MEAN_POOLING = POOLINGS[1]
MAX_POOLING = POOLINGS[2]


class LogMetrics(Callback):
    def __init__(self, current_fold):
        super().__init__()
        self.current_fold = current_fold

    def on_epoch_end(self, epoch, logs=None):
        logs["fold"] = self.current_fold


def atom_vectors_from_csv(embedding_csv):
    logger.info(f"reading atom vectors from {embedding_csv}")
    df = pd.read_csv(embedding_csv)
    elements = list(df["element"])
    df.drop(["element"], axis=1, inplace=True)
    embeddings = df.to_numpy()
    dictionary = {e: i for i, e in enumerate(elements)}
    return dictionary, embeddings


def species_vectors_from_csv(embedding_csv):
    logger.info(f"reading species vectors from {embedding_csv}")
    df = pd.read_csv(embedding_csv)
    elements = list(df["species"])
    df.drop(["species"], axis=1, inplace=True)
    embeddings = df.to_numpy()
    dictionary = {e: i for i, e in enumerate(elements)}
    return dictionary, embeddings


def get_composition(val, input_type):
    if input_type == "composition":
        return Composition(val)
    elif input_type == "structure":
        return val.composition
    else:
        raise Exception(f"unrecognized input type: {input_type}")


def featurize(X, input_type, atom_dictionary, atom_embeddings, pool):
    X_featurized = []
    for val in X.values:
        composition = get_composition(val, input_type)

        if any([e.name not in atom_dictionary for e in composition.elements]):
            raise Exception(f"{composition.reduced_formula} contains unsupported atoms")

        X_featurized.append(pool(composition, atom_dictionary, atom_embeddings))
    return np.array(X_featurized)


def featurize_species(
    X,
    input_type,
    species_dictionary,
    species_embeddings,
    atom_dictionary,
    atom_embeddings,
    pool,
):
    X_featurized = []
    for val in X.values:
        composition = get_composition_species(val, input_type)
        # print(composition.reduced_formula)

        if any(
            [
                e.to_pretty_string() not in species_dictionary
                for e in composition.elements
            ]
        ):
            try:
                composition = get_composition(val, input_type)
                if any([e.name not in atom_dictionary for e in composition.elements]):
                    raise Exception(
                        f"{composition.reduced_formula} contains unsupported atoms"
                    )
                X_featurized.append(pool(composition, atom_dictionary, atom_embeddings))
            except:
                raise Exception(
                    f"{composition.reduced_formula} contains unsupported species and atoms"
                )
        else:
            X_featurized.append(
                pool(
                    composition,
                    species_dictionary,
                    species_embeddings,
                    species_mode=True,
                )
            )
    return np.array(X_featurized)


if __name__ == "__main__":
    supported_tasks = [
        "is_metal",
        "formation_energy_per_atom",
        "band_gap",
        "is_magnetic",
    ]

    parser = argparse.ArgumentParser(
        description="Run the SkipSpecies benchmark on the Materials Project dataset."
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        required=True,
        type=str,
        help="path to the dataset .json.gz file",
    )
    parser.add_argument(
        "--results",
        nargs="?",
        required=True,
        type=str,
        help="path to the directory where the results .csv file will be written "
        "(this directory must already exist)",
    )
    parser.add_argument(
        "--models",
        nargs="?",
        required=True,
        type=str,
        help="path to the directory where models will be persisted "
        "(NOTE: this directory will be deleted if it exists)",
    )

    parser.add_argument(
        "--vectors", required=True, type=str, help="path to the atom vectors .csv file"
    )
    parser.add_argument(
        "--species-vectors",
        required=False,
        type=str,
        help="path to the species vectors .csv file",
    )
    parser.add_argument(
        "--pooling",
        required=True,
        choices=POOLINGS,
        help="the type of pooling operation to use",
    )
    parser.add_argument()

    parser.add_argument(
        "--val_size",
        required=False,
        type=float,
        default=0.10,
        help="a number between 0 and 1 representing the fraction of the training set to "
        "withold as a validation set during training",
    )
    parser.add_argument(
        "--test_size",
        required=False,
        type=float,
        default=0.20,
        help="a number between 0 and 1 representing the fraction of the training set to "
        "withold as a test set during training",
    )

    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=42,
        help="the random state to use for the validation set split",
    )
    parser.add_argument(
        "--epochs",
        required=False,
        type=int,
        default=100,
        help="the maximum number of epochs",
    )
    parser.add_argument(
        "--batch", required=False, type=int, default=32, help="the batch size"
    )
    parser.add_argument(
        "--lr", required=False, type=float, default=0.0001, help="the learning rate"
    )
    parser.add_argument(
        "--activation",
        required=False,
        type=str,
        default="relu",
        help="the type of activation to use",
    )
    parser.add_argument(
        "--l2",
        required=False,
        type=float,
        default=0.00001,
        help="the L2 lambda value to use",
    )

    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        default=False,
        help="whether to use early stopping",
    )
    parser.add_argument(
        "--patience",
        required=("--early-stopping" in argv),
        type=int,
        help="the patience to use if early stopping was specified",
    )
    parser.add_argument(
        "--species",
        action="store_true",
        help="whether to use species vectors",
    )

    args = parser.parse_args()

    task = mb.tasks_map[args.task]
    task.load()

    experiment = f"experiment_{int(time())}"
    model_dir = os.path.join(args.models, experiment)
    if os.path.exists(model_dir):
        # delete the model dir, if it exists
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    logger.info(f"saving models in {model_dir}")

    if not os.path.exists(args.results):
        os.makedirs(args.results)
    logger.info(f"saving results in: {args.results}")

    is_classification = task.metadata["task_type"] == "classification"
    architecture = None
    if is_classification:
        architecture = ElemNetClassifier
    else:
        architecture = ElemNet

    pool = None
    if args.pooling == SUM_POOLING:
        pool = sum_pool
    elif args.pooling == MEAN_POOLING:
        pool = mean_pool
    elif args.pooling == MAX_POOLING:
        pool = max_pool
    else:
        raise Exception(f"unsupported pooling: {args.pooling}")

    if args.species:
        atom_dictionary, atom_embeddings = atom_vectors_from_csv(args.vectors)
        species_dictionary, species_embeddings = species_vectors_from_csv(
            args.species_vectors
        )
    else:
        atom_dictionary, atom_embeddings = atom_vectors_from_csv(args.vectors)

    input_type = task.metadata["input_type"]

    logger.info(f"architecture: {architecture}")
    logger.info(f"val_size: {args.val_size}")
    logger.info(f"seed: {args.seed}")
    logger.info(f"max epochs: {args.epochs}")
    logger.info(f"batch size: {args.batch}")
    logger.info(f"learning rate: {args.lr}")
    logger.info(f"activation: {args.activation}")
    logger.info(f"l2: {args.l2}")
    logger.info(f"early stopping: {args.early_stopping}")
    if args.early_stopping:
        logger.info(f"patience: {args.patience}")

    for fold in task.folds:
        logger.info(f"FOLD {fold+1}")

        log_metrics = LogMetrics(current_fold=fold + 1)
        csv_log_filename = os.path.join(args.results, f"{experiment}-fold-results.csv")
        csv_logger = CSVLogger(csv_log_filename, separator=",", append=True)

        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(model_dir, f"best_model_fold_{fold+1}"),
            save_best_only=True,
            monitor="val_auc" if is_classification else "val_mae",
            mode="max" if is_classification else "min",
            verbose=1,
        )

        callbacks = [log_metrics, csv_logger, model_checkpoint]
        if args.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_auc" if is_classification else "val_mae",
                    mode="max" if is_classification else "min",
                    patience=args.patience,
                )
            )

        model = architecture(
            input_dim=X_train[0].shape[0], activation=args.activation, l2_lambda=args.l2
        )

        model.train(
            X_train,
            y_train,
            X_val,
            y_val,
            num_epochs=args.epochs,
            batch_size=args.batch,
            step_size=args.lr,
            callbacks=callbacks,
        )

        checkpoints = [os.path.join(model_dir, name) for name in os.listdir(model_dir)]
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        logger.info(f"restoring from: {latest_checkpoint}")
        best_model = model.load(latest_checkpoint)

        logger.info("evaluating on test set...")
        # Get testing data
        test_inputs = task.get_test_data(fold, include_target=False)
        if args.species:
            X_test = featurize_species(
                test_inputs,
                input_type,
                species_dictionary,
                species_embeddings,
                atom_dictionary,
                atom_embeddings,
                pool,
            )
        else:
            X_test = featurize(
                test_inputs, input_type, atom_dictionary, atom_embeddings, pool
            )

        # Predict on the testing data;
        #  output should be a pandas series, numpy array, or python iterable
        #  where the array elements are floats or bools
        predictions = best_model.predict(X_test)
        task.record(fold, predictions.flatten())

    # Save results
    # TODO according to docs, `to_file` should apparently save to a .json.gz file,
    #   but it doesn't, in matbench v0.6
    mb.to_file(os.path.join(args.results, f"{experiment}-benchmark.json"))
