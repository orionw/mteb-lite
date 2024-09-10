import os
import faiss
import mteb
import csv
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import argparse
from tqdm import tqdm

from pyserini.output_writer import get_output_writer, OutputFormat

def load_docids(docid_path: str) -> List[str]:
    id_f = open(docid_path, 'r')
    docids = [line.rstrip() for line in id_f.readlines()]
    id_f.close()
    return docids

def load_index(index_dir: str):
    index_path = os.path.join(index_dir, 'index')
    docid_path = os.path.join(index_dir, 'docid')
    index = faiss.read_index(index_path)
    docids = load_docids(docid_path)
    return index, docids

def batch_search(query_encoder, index, queries: List[str], q_ids: List[str], batch_size: int = 32, k: int = 10, threads: int = 1) \
        -> Dict[str, Tuple[Union[str, List[str]], List[float]]]:
    faiss.omp_set_num_threads(threads)

    results = {}
    num_batches = len(queries) // batch_size + (1 if len(queries) % batch_size != 0 else 0)
    for i in tqdm(range(0, len(queries), batch_size), total=num_batches, desc="Processing Batches"):
        q_batch = queries[i:i + batch_size]
        q_batch_ids = query_ids[i:i + batch_size]
        q_embs = np.array(query_encoder.encode(q_batch))
        n, m = q_embs.shape
        assert m == index.d 
        D, I = index.search(q_embs, k)
        for key, distances, indexes in zip(q_batch_ids, D, I):
            # Filter out invalid indices (-1)
            filtered_results = [(int(idx), float(score)) for score, idx in zip(distances, indexes) if idx != -1]
            results[key] = filtered_results

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search a Faiss index.')
    parser.add_argument('--query-file', type=str, metavar='topic_name', required=True,
                        help="Name of topics. Available: msmarco-passage-dev-subset.")
    parser.add_argument('--index', type=str, metavar='path to index or index name', required=True,
                        help="Path to Faiss index or name of prebuilt index.")
    parser.add_argument('--encoder', type=str, metavar='path to query encoder checkpoint or encoder name',
                        required=False,
                        help="Path to query encoder pytorch checkpoint or hgf encoder model name")
    parser.add_argument('--hits', type=int, metavar='num', required=False, default=1000, help="Number of hits.")
    parser.add_argument('--batch-size', type=int, metavar='num', required=False, default=1,
                        help="search batch of queries in parallel")
    parser.add_argument('--output', type=str, metavar='path', required=True, help="Path to output file.")
    parser.add_argument('--threads', type=int, metavar='num', required=False, default=1,
                        help="maximum threads to use during search")
    parser.add_argument('--max-passage', action='store_true',
                        default=False, help="Select only max passage from document.")
    parser.add_argument('--max-passage-hits', type=int, metavar='num', required=False, default=100,
                        help="Final number of hits when selecting only max passage.")
    parser.add_argument('--max-passage-delimiter', type=str, metavar='str', required=False, default='#',
                        help="Delimiter between docid and passage id.")
    args = parser.parse_args()

    index, docids = load_index(args.index)
    print(index.ntotal)

    # load queries
    query_ids = []
    queries = []
    with open(args.query_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            query_ids.append(row[0])
            queries.append(row[1])
            # if len(queries) == 100:
            #     break

    model = mteb.get_model(args.encoder)
    model.model = model.model.cuda()

    results = batch_search(model, index, queries, query_ids, batch_size=args.batch_size, k=args.hits, threads=args.threads)

    tag = 'Faiss'
    with open(args.output, 'w') as f:
        for query_id, hits in results.items():
            for rank, (idx, score) in enumerate(hits):
                f.write(f'{query_id} Q0 {docids[idx]} {rank + 1} {score:.6f} {tag}\n')
