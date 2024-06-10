from datasets import load_dataset
import argparse
import os

def download(dataset_name: str):
    dataset = load_dataset(dataset_name, "queries")["queries"]
    # save the dataset out as a tsv file
    os.makedirs("artifacts", exist_ok=True)
    dataset.to_csv(f"artifacts/{dataset_name.replace('/', '--')}.tsv", sep="\t", index=False, header=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, help="The name of the dataset to download")
    args = parser.parse_args()
    download(args.dataset_name)