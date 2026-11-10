"""Étape « prepare » : load_data -> clean -> build_features -> sauvegarde.

Usage (depuis la racine du projet) :
    python -m src.prepare
"""
import logging

import yaml

from src.clean import clean
from src.features import build_features, save_processed
from src.load_data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")


def main(params_path: str = "params.yaml") -> None:
    with open(params_path) as f:
        params = yaml.safe_load(f)

    df = load_data(params["data"]["raw_path"])
    df = clean(df, max_dimension_mm=params["clean"]["max_dimension_mm"])
    df = build_features(df)
    save_processed(df, params["data"]["processed_path"])


if __name__ == "__main__":
    main()