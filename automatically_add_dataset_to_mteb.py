import os
import glob
import json
import argparse
# import the huggingface hub
from huggingface_hub import list_repo_refs

def add_dataset_to_mteb(original_repo_name: str, new_repo_name: str, path_to_existing: str):
    new_name = new_repo_name.split("/")[-1].replace("-", "_")
    existing_dataset_name = original_repo_name.split("/")[-1]

    # get latest revision (e.g. git hash) of the new_repo_name
    repo_hash = list_repo_refs(new_repo_name.replace("-", "_"), repo_type="dataset").branches[0].target_commit
    
    # copy the template file with the new name over
    
    # replace the last name with the new name
    new_name_filled = os.path.join(os.path.dirname(path_to_existing), new_name + ".py")
    os.system(f"cp -r {path_to_existing} {new_name_filled}")

    # replace key fields so that
    """
    class NQ(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ",
        dataset={
            "path": "mteb/nq",
    """

    # becomes
    """
    class {new_name}(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name={new_name},
        dataset={
            "path": {new_repo_path},
    """
    with open(new_name_filled, "r") as f:
        lines = f.readlines()
    # first start by finding the class name
    class_name = None
    for line in lines:
        if "class" in line:
            class_name = line.split(" ")[1].split("(")[0]
            break

    # replace that anywhere in the file with the new name
    for i, line in enumerate(lines):
        if class_name in line:
            lines[i] = line.replace(class_name, new_name)

    # now replace the "path" with the new path
    for i, line in enumerate(lines):
        if 'path":' in line:
            lines[i] = f'\t\t\t"path": "{new_repo_name.replace('-', '_')}",\n'
        elif 'revision":' in line:
            lines[i] = f'\t\t\t"revision": "{repo_hash}",\n'

    with open(new_name_filled, "w") as f:
        f.writelines(lines)
        
    print(f"Saved new file to {new_name_filled}")

    # add to the __init__.py by opening `mteb/mteb/tasks/Retrieval/__init__.py` and inserting it anywhere
    init_file = "mteb_repo/mteb/tasks/Retrieval/__init__.py"
    with open(init_file, "r") as f:
        lines = f.readlines()
    
    # insert the new line (`from .eng.NFCorpusRetrieval import * `) after the original
    original_lang_folder = path_to_existing.split("/")[-2]
    original_file_name = path_to_existing.split("/")[-1].split(".")[0]
    new_line = f"from .{original_lang_folder}.{new_name} import *\n"
    for i, line in enumerate(lines):
        if original_file_name in line:
            lines.insert(i + 1, new_line)
            break

    with open(init_file, "w") as f:
        f.writelines(lines)

    print(f"Added new line to {init_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new dataset to MTEB")
    parser.add_argument("--original_repo_name", type=str, help="The original repo name")
    parser.add_argument("--path_to_existing", type=str, help="The path to the existing file")
    args = parser.parse_args()

    num_docs = [2, 5, 10, 50, 100, 500]
    for w_correct in [True, False]:
        w_correct_str = "_w_correct" if w_correct else ""
        for num_doc in num_docs:
            add_dataset_to_mteb(args.original_repo_name, args.original_repo_name + f"_top_{num_doc}_only{w_correct_str}", args.path_to_existing)


    # example usage
    #   python automatically_add_dataset_to_mteb.py --original_repo_name "mteb/nq" --path_to_existing "/home/hltcoe/oweller/my_exps/mteb-lite/mteb/mteb/tasks/Retrieval/eng/NQRetrieval.py"


