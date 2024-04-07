#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from asyncio import run
from math import ceil

import pandas as pd


def collect_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tf description based genre classification"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input file path (CSV)",
    )
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        default=0.2,
        help="Input file path (CSV)",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=32,
    )
    return parser.parse_args()

def train_test_split(data_path: str, _size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets
    """
    master = pd.read_csv(data_path)
    n_samples, n_features = master.shape
    n_train = ceil(n_samples * (1 - _size))
    # n_test = n_samples - n_train
    train_df = master.iloc[:n_train]
    test_df = master.iloc[n_train:]
    return train_df, test_df


def create_model(epochs: int = 10, batch_size: int = 32):
    # TODO: Implement model creation
    return None


def train_model(model, df_train: pd.DataFrame):
    # TODO: Implement model creation
    return None


async def main():
    args = collect_args()
    train, test = train_test_split(args.input, args.split)
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    # model = create_model(args.epochs, args.batch_size)
    # train_model(model, train)
    # evaluate_model(model, test)



if __name__ == "__main__":
    run(main())
