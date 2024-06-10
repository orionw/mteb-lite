### Downsampling MTEB Retrieval Datasets



## To Reproduce
0. Install a python (3.10+) enviroment with requirements.txt
1. Start by embedding a corpus so we can search for the top results: `python embed_corpus.py --dataset DATASET --model MODEL`
2. Convert the embeddings to faiss so we can search with pyserini `bash convert_to_faiss.sh PATH_TO_EMBEDDING.json DIM_SIZE`
3. Download the queries file with `python download_queries.py --dataset DATASET_NAME`
4. Mine the run file with the hard negatives per corpus embedding using `bash mine_hard_negatives.sh INDEX_FOLDER/full QUERIES_FILE NORMALIZE_TRUE_OR_FALSE BATCH_SIZE NUM_HITS MODEL_NAME`
5. Subsample the dataset and push them to the hub with `python subsample_dataset.py --run_files RUN_FILE_LIST --dataset_name DATASET_NAME"
6. Add all of those new datasets to MTEB using `python automatically_add_dataset_to_mteb.py --original_repo_name DATASET_NAME --path_to_existing PATH_TO_ORIGINAL_FILE_IN_MTEB_REPO`
7. Evaluate all models on those new datasets with `python run_mteb_on_datasets.py --dataset_name DATASET_NAME`

For example, use "mteb/nq" for the dataset name and "intfloat/e5-large-v2" as the model name.