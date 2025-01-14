#!/bin/bash

# this is the official evaluation scripts but running in a venv
EVAL_REPO_PATH=$1
RESULTS_DIR_PATH=$2
MODEL_PATH=$3
cd $EVAL_REPO_PATH

source $EVAL_REPO_PATH/.venv/bin/activate

MODEL_BASENAME=$(basename $MODEL_PATH)

python -m lm_eval --model hf-mlm \
    --model_args pretrained=$MODEL_PATH,backend="mlm" \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:2 \
    --batch_size 1 \
    --log_samples \
    --output_path ${RESULTS_DIR_PATH}/blimp/${MODEL_BASENAME}/blimp_results.json

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.


python -m lm_eval --model hf-mlm \
    --model_args pretrained=$MODEL_PATH,backend="mlm" \
    --tasks ewok_filtered \
    --device cuda:2\
    --batch_size 1 \
    --log_samples \
    --output_path ${RESULTS_DIR_PATH}/ewok/${MODEL_BASENAME}/ewok_results.json

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.