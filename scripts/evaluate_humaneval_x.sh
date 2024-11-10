#!/bin/bash

# This script is for evaluating the functional correctness of the generated codes of HumanEval-X.

LOAD_GENERATIONS_PATH=$1  # Path to the .jsonl file that contains the generated codes.
LANGUAGE=$2  # Target programming language, currently support one of ["python", "java", "cpp", "js", "go"]
N_WORKERS=$3  # Number of parallel workers.
TIMEOUT=$4  # Timeout in seconds.

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")

echo "$LOAD_GENERATIONS_PATH"

if [ -z "$N_WORKERS" ]
then
    N_WORKERS=64
fi

if [ -z "$LANGUAGE" ]
then
    LANGUAGE=python
fi

if [ -z "$TIMEOUT" ]
then
    TIMEOUT=5
fi

DATASET_PATH="$MAIN_DIR/codegeex/benchmark/humaneval-x/$LANGUAGE/data/humaneval_$LANGUAGE.jsonl.gz"

if [ "$LANGUAGE" = go ]; then
  export PATH=$PATH:/usr/local/go/bin
fi

if [ "$LANGUAGE" = cpp ]; then
  export PATH=$PATH:/usr/bin/openssl
fi

CMD="python $MAIN_DIR/codegeex/benchmark/humaneval-x/evaluate_humaneval_x.py \
    --load_generations_path $LOAD_GENERATIONS_PATH \
    --n_workers $N_WORKERS \
    --tmp_dir $MAIN_DIR/codegeex/benchmark/humaneval-x/ \
    --dataset_path $DATASET_PATH \
    --timeout $TIMEOUT"

echo "$CMD"
eval "$CMD"