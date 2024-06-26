# Downsampling MTEB Retrieval Datasets


## Enviroment Setup
0. Install a python (3.10+) enviroment with requirements.txt (`conda create -n mteb-lite python=3.10 -y && conda activate mteb-lite && pip install -r requirements.txt`)
1. Install Java `conda install -c conda-forge openjdk -y` 

## Tasks to Downsample
The tasks to downsample are located in [tasks_to_downsample.txt](https://github.com/orionw/mteb-lite/blob/master/tasks_to_downsample.txt) and include both the name and the split. 


## To Reproduce (two steps)
Run `bash run_all.sh NFCorpus intfloat/e5-small-v2 384 test` switching out the datasets and models you prefer. It needs the dimension (`384`) and the split of the dataset (default is `test` if none is passed).  If the model's embeddings **should not** be normalized, pass an additional `false` parameter after `test`.

Then we need to push the shared run files to [mteb/mteb-lite-run-files](https://huggingface.co/datasets/mteb/mteb-lite-run-files). They will be located locally in `artifacts/run_{MODEL_NAME}_{DATASET_NAME}-{DATASET_SPLIT}.tsv`.


## To Reproduce (step by step)
0. Start by embedding a corpus so we can search for the top results: `python embed_corpus.py --dataset DATASET --model MODEL --split SPLIT_NAME`
1. Convert the embeddings to faiss so we can search with pyserini `bash convert_to_faiss.sh PATH_TO_EMBEDDING.json DIM_SIZE`
2. Download the queries file with `python download_queries.py --dataset DATASET_NAME --split SPLIT_NAME`
3. Mine the run file with the hard negatives per corpus embedding using `bash mine_hard_negatives.sh INDEX_FOLDER/full QUERIES_FILE NORMALIZE_TRUE_OR_FALSE BATCH_SIZE NUM_HITS MODEL_NAME`
4. Subsample the dataset and push them to the hub with `python subsample_dataset.py --run_files RUN_FILE_LIST --dataset_name DATASET_NAME"
5. Add all of those new datasets to MTEB using `python automatically_add_dataset_to_mteb.py --original_repo_name DATASET_NAME --path_to_existing PATH_TO_ORIGINAL_FILE_IN_MTEB_REPO`
6. Evaluate all models on those new datasets with `python run_mteb_on_datasets.py --dataset_name DATASET_NAME`

For example, use "NQ" for the dataset name and "intfloat/e5-large-v2" as the model name.
