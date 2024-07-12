import argparse
import os
from datetime import datetime

import pandas as pd
from mp_api.client import MPRester


def is_integer(num: float) -> bool:
    """Determines if a float is an integer"""
    return int(num) == num


def all_integers(listNums: list[float]) -> bool:
    return all([is_integer(num) for num in listNums])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a structure dataset from the Materials Project"
    )
    parser.add_argument(
        "--apikey", required=False, type=str, help="API Key for the Materials Project"
    )
    parser.add_argument(
        "--datapath",
        nargs="?",
        required=True,
        type=str,
        help="path to the directory where the structure dataset will be downloaded",
    )
    parser.add_argument(
        "--oxi_mp",
        action="store_true",
        help="Whether to get an oxidation state decorated structure dataset from the Materials Project",
    )

    args = parser.parse_args()

    if args.apikey == None:
        print("API Key not supplied, checking environment for MP_API_KEY")
        APIKEY = os.environ["MP_API_KEY"]
    else:
        APIKEY = args.apikey
    if APIKEY is None:
        raise Exception("No API KEY found. Please supply API KEY")
    if args.oxi_mp:
        with MPRester(APIKEY, use_document_model=False) as mpr:
            docs = mpr.oxidation_states.search(
                fields=[
                    "material_id",
                    "structure",
                    "deprecated",
                    "method",
                    "possible_valences",
                    "possible_species",
                ]
            )

            db_version = mpr.get_database_version()

        # Format db_version
        date_object = datetime.strptime(db_version, "%Y.%m.%d").date()
        db_version_fmt = str(date_object).replace("-", "_")

        print(
            f"Querying the oxidation states route of the Materials Project v{db_version}, returned {len(docs)} entries."
        )

        # Create a dataframe from the documents
        df = pd.DataFrame(docs)

        print("Filtering out deprecated structures")
        df = df[df["deprecated"] == False]

        print(f"Filtering out the deprecated structures resulted in {len(df)} entries")

        print("Filtering out structures with no oxidation states")
        df = df.dropna(subset="method")

        print(
            f"Filtering out structures with no oxidation states resulted in {len(df)} entries"
        )

        print("Filtering out structures with non-integer oxidation states")
        df["int_states"] = df["possible_valences"].apply(all_integers)
        df = df[df["int_states"] == True].reset_index(drop=True)

        print(
            f"Filtering out structures with non-integer oxidation states resulted in {len(df)} entries"
        )

        print("Saving the dataset to disk")

        df[["structure", "material_id"]].to_pickle(
            os.path.join(args.datapath, f"mp_{db_version_fmt}_oxi.pkl.gz")
        )

        print(
            f"Dataset saved to {os.path.join(args.datapath,f'mp_{db_version_fmt}_oxi.pkl.gz')}"
        )
    else:
        with MPRester(APIKEY, use_document_model=False) as mpr:
            docs = mpr.materials.summary.search(
                fields=["material_id", "structure", "deprecated"]
            )

            db_version = mpr.get_database_version()

        # Format db_version
        date_object = datetime.strptime(db_version, "%Y.%m.%d").date()
        db_version_fmt = str(date_object).replace("-", "_")

        print(
            f"Querying the materials summary route of the Materials Project v{db_version}, returned {len(docs)} entries."
        )

        # Create a dataframe from the documents
        df = pd.DataFrame(docs)

        print("Saving the dataset to disk")

        df[["structure", "material_id"]].to_pickle(
            os.path.join(args.datapath, f"mp_{db_version_fmt}.pkl.gz")
        )

        print(
            f"Dataset saved to {os.path.join(args.datapath,f'mp_{db_version_fmt}.pkl.gz')}"
        )
