from datasets import load_dataset
import argparse
import os
import mteb

def download(dataset_name: str, split: str, subsplit: str = None):
    tasks = mteb.get_tasks(tasks=[dataset_name])
    assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
    tasks[0].load_data()
    queries = tasks[0].queries[split]
    if subsplit is not None:
        queries = queries[subsplit]
    # save it as a csv to output path
    output_path = f"artifacts/{dataset_name.replace('/', '--')}-{split}.tsv"
    print(f"Dataset {dataset_name} has {len(queries)} queries")

    # save the dataset out as a tsv file
    os.makedirs("artifacts", exist_ok=True)
    with open(output_path, "w") as f:
        for query_id, query in queries.items():
            f.write(f"{query_id}\t{query}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, help="The name of the dataset to download")
    parser.add_argument("-s", "--split", type=str, default="test", help="The split to download")
    parser.add_argument("-ss", "--subsplit", type=str, default=None, help="The subsplit to download")
    args = parser.parse_args()
    download(args.dataset_name, args.split)