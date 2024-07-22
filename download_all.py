from datasets import load_dataset
import argparse
import os
import mteb
import json
import tqdm


def download(dataset_name: str, split: str, output_folder: str, subsplit=None):
    tasks = mteb.get_tasks(tasks=[dataset_name])
    assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
    print(f"Downloading {dataset_name} {split} split")
    tasks[0].load_data()

    queries = tasks[0].queries[split]
    if args.subsplit is not None:
        queries = queries[args.subsplit]

    # save it as a csv to output path
    output_path = os.path.join(output_folder, f"{dataset_name}--{split}", "queries.tsv")
    # make directory if it doesn't exist
    os.makedirs(os.path.join(output_folder, f"{dataset_name}--{split}"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, f"{dataset_name}--{split}", "corpus"), exist_ok=True)
    print(f"Dataset {dataset_name} has {len(queries)} queries")

    # save the dataset out as a tsv file
    with open(output_path, "w") as f:
        for query_id, query in tqdm.tqdm(queries.items()):
            f.write(f"{query_id}\t{query}\n")

    # now do the corpus
    corpus = tasks[0].corpus[split]
    if args.subsplit is not None:
        corpus = corpus[args.subsplit]

    output_path = os.path.join(output_folder, f"{dataset_name}--{split}", "corpus", "corpus.jsonl")
    print(f"Dataset {dataset_name} has {len(corpus)} documents")

    # save the dataset out as a jsonl file
    with open(output_path, "w") as f:
        for doc_id, doc in tqdm.tqdm(corpus.items()):
            text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            f.write(json.dumps({"id": doc_id, "contents": text}) + "\n")

    # now do the qrels
    qrels = tasks[0].relevant_docs[split]
    if args.subsplit is not None:
        qrels = qrels[args.subsplit]
    output_path = os.path.join(output_folder, f"{dataset_name}--{split}", "qrels.tsv")
    print(f"Dataset {dataset_name} has {len(qrels)} qrels")

    # save the dataset out as a tsv file
    with open(output_path, "w") as f:
        for query_id, relevant_docs in tqdm.tqdm(qrels.items()):
            for doc_id, score in relevant_docs.items():
                f.write(f"{query_id}\t{doc_id}\t{score}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, help="The name of the dataset to download")
    parser.add_argument("-s", "--split", type=str, default="test", help="The split to download")
    parser.add_argument("-ss", "--subsplit", type=str, default=None, help="The subsplit to download")
    parser.add_argument("-o", "--output_path", type=str, default="datasets", help="The output path to save the dataset")
    args = parser.parse_args()
    download(args.dataset_name, args.split, args.output_path, args.subsplit)