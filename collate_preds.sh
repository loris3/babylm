#!/bin/bash

python -m evaluation_pipeline.collate_preds --model_path_or_name=babylm-anon/TICL --backend=mlm --fast --track=strict-small --results_dir=/srv/home/users/loriss21cs/babylm/results