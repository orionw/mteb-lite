import os
import json
import datasets
import argparse
import pandas as pd
import tqdm
import copy

NUM_DOCS = [2, 5, 10, 50, 100, 500]
flatten = lambda l: [item for sublist in l for item in sublist]


def create_subselections(args, qrels, corpus, queries, run_files, qids):
    # for each qid, grab the top args.docs_per_query documents, pooling from the run files
    qids = set(qids)
    subselections = {}
    for qid in tqdm.tqdm(qids):
        if args.add_correct:
            # get all relevant documents
            pd_qrels = qrels.to_pandas()
            annotations = pd_qrels[pd_qrels["query-id"] == qid]
            relevant_docs = annotations[annotations["score"] != 0]["corpus-id"].tolist()
            subselections[qid] = relevant_docs
        else:
            subselections[qid] = []
        top_runs_per_qid = [run_file[run_file["qid"] == qid]["doc_id"].tolist() for run_file in run_files]
        is_empty = False
        while len(subselections[qid]) < args.docs_per_query and not is_empty:
            # print(f"QID: {qid}, Docs: {len(subselections[qid])}")
            is_empty = True
            for run_file_docs in top_runs_per_qid:
                # print(f"Length of run_file_docs: {len(run_file_docs)}")
                if len(run_file_docs) == 0:
                    continue

                is_empty = False
            
                doc_id = run_file_docs.pop(0)
                if doc_id in subselections[qid]:
                    continue

                subselections[qid].append(doc_id)

    # write out the subselections by making a huggingface dataset, similar to the one already specified
    # but instead make the corpus smaller by only including those that are relevant
    # also make the qrels only the ones in the corpus
    all_docs = set(flatten([subselections[qid] for qid in subselections]))
    corpus = corpus.filter(lambda x: x["_id"] in all_docs)
    qrels = qrels.filter(lambda x: x["corpus-id"] in all_docs)
    
    # get all doc ids in corpus
    corpus_ids = set([doc["_id"] for doc in corpus])
    # assert that all_docs is equivalent to corpus_ids
    assert all_docs == corpus_ids, f"all_docs diff from corpus_ids: {all_docs - corpus_ids}"

    # push to the hub and update with the new stats
    correct_name = "" if not args.add_correct else "_w_correct"
    dataset_name_underscores = args.dataset_name.replace("-", "_")
    queries.push_to_hub(dataset_name_underscores + f"_top_{args.docs_per_query}_only{correct_name}", "queries")
    qrels.push_to_hub(dataset_name_underscores + f"_top_{args.docs_per_query}_only{correct_name}")
    corpus.push_to_hub(dataset_name_underscores + f"_top_{args.docs_per_query}_only{correct_name}", "corpus")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_files", type=str, nargs="+", help="The run files to subselect from")
    parser.add_argument("--docs_per_query", type=int, help="The number of documents to subselect per query")
    parser.add_argument("--dataset_name", type=str, help="The name of the dataset to subselect from")
    parser.add_argument("--add_correct", action="store_true", help="Add the correct documents to the subselections")
    args = parser.parse_args()

    if args.docs_per_query is None:
        docs_to_use = NUM_DOCS
    else:
        docs_to_use = [args.docs_per_query]

    # read in the run files
    run_files = []
    qids = []
    print(f"Reading in run files: {args.run_files}")
    # NOTE: Assumes they are sorted by highest to lowest score
    for file_name in args.run_files:
        run_files.append(pd.read_csv(file_name, sep="\s+", header=None, names=["qid", "Q0", "doc_id", "rank", "score", "model"]))
        qids.extend(run_files[-1]["qid"].unique().tolist())

    # read in the corpus
    print(f"Reading in corpus: {args.dataset_name}")
    qrels = datasets.load_dataset(args.dataset_name)["test"]
    corpus = datasets.load_dataset(args.dataset_name, "corpus")["corpus"]
    queries = datasets.load_dataset(args.dataset_name, "queries")["queries"]

    for with_correct in [True, False]:
        args.add_correct = with_correct
        print(f"Creating subselections with correct: {with_correct}")
        for num_docs in docs_to_use:
            print(f"Creating subselections for {num_docs} documents")
            args.docs_per_query = num_docs
            create_subselections(args, copy.deepcopy(qrels), copy.deepcopy(corpus), queries, run_files, qids)

    # example usage
    #   python subselect_and_create_datasets.py --run_files run_mteb--nq.tsv --dataset_name "mteb/nq"

