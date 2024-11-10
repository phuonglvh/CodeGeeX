import argparse
import os
from pathlib import Path
from codegeex.benchmark.humanevalx.evaluate_humaneval_x import evaluate_functional_correctness
#GLOBALS
GENERATIONS_PATH: str  
LANGUAGE: str  
N_WORKERS: int  
TIMEOUT: int 


parser = argparse.ArgumentParser("Debugging evaluate humaneval_x")
# Path to the .jsonl file that contains the generated codes.
parser.add_argument("-s","--load_generations_path", type=str)

# Target programming language, currently support one of ["python", "java", "cpp", "js", "go"]
parser.add_argument("-l","--language", default="python", type=str)

# Number of parallel workers.
parser.add_argument("-w","--workers", default=8, type=int)

# Timeout in seconds.
parser.add_argument("-t","--timeout", default=5, type=int)

args = parser.parse_args()

GENERATIONS_PATH = args.load_generations_path
print(f'GENERATIONS_PATH={GENERATIONS_PATH}')
LANGUAGE = args.language  
N_WORKERS = args.workers  
TIMEOUT= args.timeout


SCRIPT_PATH: str = Path(os.path.abspath(__file__))
print(f'SCRIPT_PATH={SCRIPT_PATH}')
SCRIPT_DIR: str = os.path.dirname(SCRIPT_PATH)
print(f'SCRIPT_DIR={SCRIPT_DIR}')
MAIN_DIR: str = os.path.dirname(SCRIPT_DIR)
print(f'MAIN_DIR={MAIN_DIR}')

DATASET_PATH = os.path.join(MAIN_DIR, f"codegeex/benchmark/humaneval-x/{LANGUAGE}/data/humaneval_{LANGUAGE}.jsonl.gz")
print(f'DATASET_PATH={DATASET_PATH}')

TMP_DIR=os.path.join(MAIN_DIR, "/codegeex/benchmark/humaneval-x/")
print(f'TMP_DIR={TMP_DIR}')


#Debugging
# GENERATIONS_PATH='/home/rog0d/Escritorio/CodeGeeX/generations/humaneval_rust_generations.jsonl.gz'
# LANGUAGE='rust'
# DATASET_PATH=os.path.join(MAIN_DIR,"codegeex/benchmark/humaneval-x/" + LANGUAGE + "/data/humaneval_" + LANGUAGE + ".jsonl.gz")

"""
        generations_path: str = None,
        tmp_dir: str = "./",
        n_workers: int = 32,
        timeout: float = 5.0,
        dataset_path: str = "../data/humaneval_python.jsonl.gz",
        out_dir: str = None,
        k: List[int] = [1, 10, 100],
        test_groundtruth: bool = False,
        example_test: bool = False,

"""

evaluate_functional_correctness(generations_path=GENERATIONS_PATH,
                                n_workers=N_WORKERS,
                                tmp_dir=TMP_DIR,
                                dataset_path=DATASET_PATH,
                                timeout=300.0)


