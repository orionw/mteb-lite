import os
import json
import pandas as pd
import glob
import tqdm

def gather_results(results_dir: str = "results"):
    # gather all the results from the results directory
    results = []
    print(f"Searching in {results_dir}/*/*.json")
    for file in tqdm.tqdm(glob.glob(f"{results_dir}/**/*.json", recursive=True)):
        if "model_meta" in file:
            continue
        with open(file, "r") as f:
            cur_results = json.load(f)

        model_name = file.split("/")[-4]
        dataset_name = cur_results["task_name"].split("_top")[0].replace("_", "-")
        assert len(cur_results["scores"]["test"]) == 1, f"Expected only one test score, got {len(cur_results['scores']['test'])}"
        results_dict = cur_results["scores"]["test"][0]
        results.append({
            "eval_time": cur_results["evaluation_time"],
            "task_name": cur_results["task_name"],
            "task_size": cur_results["task_name"].split("_top")[-1].split("_")[0],
            "with_correct": "with_correct" in cur_results["task_name"],
            "dataset_name": dataset_name,
            "model_name": model_name,
            **results_dict
        })

    df = pd.DataFrame(results)
    print(df)

    table_per_dataset = []
    for dataset_name, group in df.groupby("dataset_name"):
        # want it to look like models on the rows, tasks on the columns, scores in the middle (ndcg_at_10 only)
        # keep only three columns
        group_cur = group[["model_name", "task_name", "ndcg_at_10"]].sort_values("ndcg_at_10", ascending=False)
        # check if there are duplicates
        if group_cur.duplicated(subset=["model_name", "task_name"]).any():
            print(f"Found duplicates in {dataset_name}")
            breakpoint()


        table = group_cur.pivot(index="model_name", columns="task_name", values="ndcg_at_10")
        # drop any row with NAs
        table = table.dropna()
        table_per_dataset.append((dataset_name, table))

        # now convert the ndcg_at_10 into ranks for which model did the best for each task (e.g. rank per column)
        new_rank_table = table.rank(ascending=False, method='dense').reset_index()
        table_per_dataset.append((dataset_name + "_rank", new_rank_table))

    # save out each table and the overall df
    os.makedirs("tables", exist_ok=True)
    df.to_csv("tables/overall_results.csv", index=False)
    for dataset_name, table in table_per_dataset:
        # sort the columns by task_size
        # table = table.reindex(sorted(table.columns, key=lambda x: int(x.split("_top")[-1].split("_")[0])))
        table.to_csv(f"tables/{dataset_name}_results.csv")



if __name__ == "__main__":
    gather_results()