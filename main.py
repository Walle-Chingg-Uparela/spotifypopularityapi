import argparse

from train import train_model
from predict import predict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)

    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "predict":
        predict()


if __name__ == "__main__":
    main()