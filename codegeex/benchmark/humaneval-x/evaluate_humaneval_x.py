import os
import sys
import fire
import json
import gzip
import regex
import numpy as np

from typing import *
from tqdm.auto import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../..")))

from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
from codegeex.benchmark.metric import estimate_pass_at_k
from codegeex.benchmark.execution import check_correctness

LANGUAGE_NAME = {
    "cpp"   : "CPP",
    "go"    : "Go",
    "java"  : "Java",
    "js"    : "JavaScript",
    "python": "Python",
}


def create_complete_test_program(problem_generation, dataset_problems, example_test=False):
    task_id = problem_generation["task_id"]
    language = task_id.split("/")[0].lower()

    prompt = problem_generation["prompt"]
    if example_test and "example_test" in dataset_problems[task_id] and dataset_problems[task_id]["example_test"] != "":
        test = dataset_problems[task_id]["example_test"]
    else:
        test = dataset_problems[task_id]["test"]
    generated_code = problem_generation["generation"]

    # Pre-process for different languages
    if language == "python":
        code_lines = []
        # Iterate over each line in the generated code
        for line in generated_code.split("\n"):
            # Check if the line is not empty and does not start with a space or tab
            if (len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t'):
                break
            code_lines.append(line)
        # Join the lines back together into a single string
        generated_code = "\n".join(code_lines)
        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
        full_test_program = test_setup + prompt + generated_code + "\n" + test + "\n"
    elif language == "cpp":
        test_set_up = ""
        for s in IMPORT_HELPER["cpp"]:
            if s not in prompt:
                test_set_up += s + "\n"
        full_test_program = test_set_up + "\n" + prompt + generated_code + "\n" + test
    elif language == "java":
        full_test_program = prompt + generated_code + "\n" + test
    elif language == "js" or language == "javascript":
        full_test_program = prompt + generated_code + "\n" + test
    elif language == "go":
        import_string = dataset_problems[task_id]["import"]
        prompt = prompt.replace(import_string, "")
        if example_test and "example_test" in dataset_problems[task_id]:
            test = dataset_problems[task_id]["example_test"]
        else:
            test = dataset_problems[task_id]["test"]
        test_setup = dataset_problems[task_id]["test_setup"]
        other_pkgs = []
        for pkg in IMPORT_HELPER["go"]:
            if pkg not in test_setup:
                p = pkg.split("/")[-1]
                if p + "." in generated_code:
                    other_pkgs.append(f"\"{pkg}\"")
        if other_pkgs:
            import_other_pkgs = "import (\n" + "    ".join([p + "\n" for p in other_pkgs]) + ")"
            full_test_program = test_setup + "\n" + import_other_pkgs + "\n" + prompt + generated_code + "\n" + test
        else:
            full_test_program = test_setup + "\n" + prompt + generated_code + "\n" + test
    elif language == "rust":
        main = "\nfn main(){ \n } \n"
        declaration = dataset_problems[task_id]["declaration"]
        full_test_program = main + declaration + prompt + generated_code + test

    return full_test_program


def stream_jsonl_all(filename: str) -> Iterable[Dict]:
    results = []
    if filename.endswith(".gz"):
        fp = gzip.open(open(filename, "rb"), "rt")
    else:
        fp = open(filename, "r")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()

    return results


def evaluate_functional_correctness(
        load_generations_path: str = None,
        tmp_dir: str = "./",
        n_workers: int = int(os.cpu_count()/2),
        timeout: float = 500.0,
        dataset_path: str = "../data/humaneval_python.jsonl.gz",
        out_dir: str = None,
        k: List[int] = [1, 10, 100],
        test_groundtruth: bool = False,
        example_test: bool = False,
):
    if example_test:
        print("Example test...")

    dataset_problems = read_dataset(dataset_path, dataset_type="humaneval")
    dataset_generations = stream_jsonl_all(load_generations_path)

    if example_test:
        suffix = "_example_test.jsonl"
    else:
        suffix = "_results.jsonl"
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, load_generations_path.split('/')[-1].replace(".jsonl", suffix))
    else:
        out_file = os.path.join(load_generations_path.replace(".jsonl", suffix))

    if "/codegeex/benchmark/humaneval-x/" in load_generations_path:
        test_groundtruth = True

    if "-to-" in load_generations_path:
        translation_mode = True
    else:
        translation_mode = False

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        print(f'executing check_correctness by {n_workers} n_workers')
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        if test_groundtruth:
            print("Testing ground truth...")
            for problem_generation in tqdm(dataset_problems.values()):
                task_id = problem_generation["task_id"]
                lang = task_id.split("/")[0].lower()
                if lang == "javascript":
                    lang = "js"
                tmp_eval_dir = os.path.join(tmp_dir, lang, "evaluation")
                print(f'tmp_eval_dir={tmp_eval_dir}')

                problem_generation["generation"] = problem_generation["canonical_solution"]
                problem_generation["test_code"] = create_complete_test_program(problem_generation, dataset_problems, example_test)
                if problem_generation["test_code"] is None:
                    continue
                args = (task_id, problem_generation, lang, timeout, tmp_eval_dir, completion_id[task_id])
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1
        else:
            print("Reading samples...")
            for problem_generation in tqdm(dataset_generations):
                task_id = problem_generation["task_id"]
                lang = task_id.split("/")[0].lower()
                if translation_mode:
                    task_id = problem_generation["task_id"].split("/")[-1]
                    lang = regex.findall("-to-.*-", load_generations_path)[0].split("-to-")[-1].rstrip("-")
                    for l in LANGUAGE_NAME:
                        if l in lang:
                            lang = l
                            break
                    task_id = f"{LANGUAGE_NAME[lang]}/{task_id}"
                if lang == "javascript":
                    lang = "js"
                tmp_eval_dir = os.path.join(tmp_dir, lang, "evaluation")
                print(f'tmp_eval_dir={tmp_eval_dir}')
                
                problem_generation["task_id"] = task_id
                problem_generation["test_code"] = create_complete_test_program(problem_generation, dataset_problems, example_test)
                if problem_generation["test_code"] is None:
                    continue
                if "completion_id" in problem_generation:
                    completion_id_ = problem_generation["completion_id"]
                else:
                    completion_id_ = completion_id[task_id]
                args = (task_id, problem_generation, lang, timeout, tmp_eval_dir, completion_id_)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        print(completion_id)
        if len(completion_id) == len(dataset_problems):
            evaluate_pass_at_k = True
        else:
            evaluate_pass_at_k = False

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)
    if evaluate_pass_at_k:
        ks = k
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                     for k in ks if (total >= k).all()}
        print(pass_at_k)
    else:
        print("Total:", np.sum(total))
        print("Correct:", np.sum(correct))

    print("Writing to: ", out_file)
    if out_file.endswith(".gz"):
        fp = gzip.GzipFile(fileobj=open(out_file, "wb"), mode="wb")
        for res in results.values():
            for r in res:
                fp.write((json.dumps(r[1]) + "\n").encode("utf-8"))
    else:
        fp = open(out_file, 'w')
        for res in results.values():
            for r in res:
                fp.write(json.dumps(r[1]) + "\n")
    fp.close()

    print("Evaluation finished.")


def main():
    fire.Fire(evaluate_functional_correctness)


if __name__ == "__main__":
    sys.exit(main())
