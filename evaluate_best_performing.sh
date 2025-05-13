#!/bin/bash 
#SBATCH --job-name='[BabyLM] Evaluation of best performing model'
#SBATCH --container-image=loris3/babylm:eval
#SBATCH --container-mount-home 
#SBATCH --mem=32GB 
#SBATCH --cpus-per-task=24  
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dgx-h100-em2


python3 --version
export $(grep -v '^#' /srv/home/users/loriss21cs/babylm/.env | xargs) && huggingface-cli login --token $HF_TOKEN

pwd
mkdir /srv/home/users/loriss21cs/babylm/results




# full for main

python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --task blimp --data_path "evaluation_data/full_eval/blimp_filtered" --save_predictions --output_dir=/srv/home/users/loriss21cs/babylm/results
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --task blimp --data_path "evaluation_data/full_eval/supplement_filtered" --save_predictions --output_dir=/srv/home/users/loriss21cs/babylm/results
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --task ewok --data_path "evaluation_data/full_eval/ewok_filtered" --save_predictions --output_dir=/srv/home/users/loriss21cs/babylm/results
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --task entity_tracking --data_path "evaluation_data/full_eval/entity_tracking" --save_predictions --output_dir=/srv/home/users/loriss21cs/babylm/results
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --task wug --data_path "evaluation_data/full_eval/wug_adj_nominalization" --save_predictions --output_dir=/srv/home/users/loriss21cs/babylm/results
python -m evaluation_pipeline.reading.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --data_path "evaluation_data/full_eval/reading/reading_data.csv" --output_dir=/srv/home/users/loriss21cs/babylm/results



# fast for other checkpoints

for revision_name in "chck_1M" "chck_2M" "chck_3M" "chck_4M" "chck_5M" "chck_6M" "chck_7M" "chck_8M" "chck_9M" "chck_10M" "chck_20M" "chck_30M" "chck_40M" "chck_50M" "chck_60M" "chck_70M" "chck_80M" "chck_90M"; do
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --task blimp --data_path "evaluation_data/fast_eval/blimp_fast" --save_predictions --revision_name "$revision_name" --output_dir=/srv/home/users/loriss21cs/babylm/results
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --task blimp --data_path "evaluation_data/fast_eval/supplement_fast" --save_predictions --revision_name "$revision_name" --output_dir=/srv/home/users/loriss21cs/babylm/results
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --task ewok --data_path "evaluation_data/fast_eval/ewok_fast" --save_predictions --revision_name "$revision_name" --output_dir=/srv/home/users/loriss21cs/babylm/results
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --task wug --data_path "evaluation_data/fast_eval/wug_adj_nominalization" --save_predictions --revision_name "$revision_name" --output_dir=/srv/home/users/loriss21cs/babylm/results
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --task entity_tracking --data_path "evaluation_data/fast_eval/entity_tracking_fast" --save_predictions --revision_name "$revision_name" --output_dir=/srv/home/users/loriss21cs/babylm/results
    python -m evaluation_pipeline.reading.run --model_path_or_name loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm --backend mlm --data_path "evaluation_data/fast_eval/reading/reading_data.csv" --revision_name "$revision_name" --output_dir=/srv/home/users/loriss21cs/babylm/results
done



# finetuning 

MODEL_PATH=loris3/stratified_10m_curriculum_roberta_roberta_influence_incr_bins_lognorm_babylm
LR=3e-5           
BSZ=32            
BIG_BSZ=16    
MAX_EPOCHS=10   
WSC_EPOCHS=30   
SEED=42         

model_basename=$(basename $MODEL_PATH)


        
python -m evaluation_pipeline.finetune.run     --model_name_or_path "$MODEL_PATH"     --train_data "evaluation_data/full_eval/glue_filtered/boolq.train.jsonl"     --valid_data "evaluation_data/full_eval/glue_filtered/boolq.valid.jsonl"     --predict_data "evaluation_data/full_eval/glue_filtered/boolq.valid.jsonl"     --task boolq     --num_labels 2     --batch_size $BIG_BSZ     --learning_rate $LR     --num_epochs $MAX_EPOCHS     --sequence_length 512     --results_dir /srv/home/users/loriss21cs/babylm/results     --save     --save_dir /srv/home/users/loriss21cs/babylm/models     --metrics accuracy f1 mcc     --metric_for_valid accuracy     --seed $SEED     --verbose
python -m evaluation_pipeline.finetune.run     --model_name_or_path "$MODEL_PATH"     --train_data "evaluation_data/full_eval/glue_filtered/multirc.train.jsonl"     --valid_data "evaluation_data/full_eval/glue_filtered/multirc.valid.jsonl"     --predict_data "evaluation_data/full_eval/glue_filtered/multirc.valid.jsonl"     --task multirc     --num_labels 2     --batch_size $BIG_BSZ     --learning_rate $LR     --num_epochs $MAX_EPOCHS     --sequence_length 512     --results_dir /srv/home/users/loriss21cs/babylm/results     --save     --save_dir /srv/home/users/loriss21cs/babylm/models     --metrics accuracy f1 mcc     --metric_for_valid accuracy     --seed $SEED     --verbose

python -m evaluation_pipeline.finetune.run     --model_name_or_path "$MODEL_PATH"     --train_data "evaluation_data/full_eval/glue_filtered/rte.train.jsonl"     --valid_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl"     --predict_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl"     --task rte     --num_labels 2     --batch_size $BSZ     --learning_rate $LR     --num_epochs $MAX_EPOCHS     --sequence_length 512     --results_dir /srv/home/users/loriss21cs/babylm/results     --save     --save_dir /srv/home/users/loriss21cs/babylm/models     --metrics accuracy f1 mcc     --metric_for_valid accuracy     --seed $SEED     --verbose

python -m evaluation_pipeline.finetune.run     --model_name_or_path "$MODEL_PATH"     --train_data "evaluation_data/full_eval/glue_filtered/wsc.train.jsonl"     --valid_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl"     --predict_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl"     --task wsc     --num_labels 2     --batch_size $BSZ     --learning_rate $LR     --num_epochs $WSC_EPOCHS     --sequence_length 512     --results_dir /srv/home/users/loriss21cs/babylm/results     --save     --save_dir /srv/home/users/loriss21cs/babylm/models     --metrics accuracy     --metric_for_valid accuracy     --seed $SEED     --verbose


        
python -m evaluation_pipeline.finetune.run     --model_name_or_path "$MODEL_PATH"     --train_data "evaluation_data/full_eval/glue_filtered/mrpc.train.jsonl"     --valid_data "evaluation_data/full_eval/glue_filtered/mrpc.valid.jsonl"     --predict_data "evaluation_data/full_eval/glue_filtered/mrpc.valid.jsonl"     --task mrpc     --num_labels 2     --batch_size $BSZ     --learning_rate $LR     --num_epochs $MAX_EPOCHS     --sequence_length 512     --results_dir /srv/home/users/loriss21cs/babylm/results     --save     --save_dir /srv/home/users/loriss21cs/babylm/models     --metrics accuracy f1 mcc     --metric_for_valid f1     --seed $SEED     --verbose

python -m evaluation_pipeline.finetune.run     --model_name_or_path "$MODEL_PATH"     --train_data "evaluation_data/full_eval/glue_filtered/qqp.train.jsonl"     --valid_data "evaluation_data/full_eval/glue_filtered/qqp.valid.jsonl"     --predict_data "evaluation_data/full_eval/glue_filtered/qqp.valid.jsonl"     --task qqp     --num_labels 2     --batch_size $BSZ     --learning_rate $LR     --num_epochs $MAX_EPOCHS     --sequence_length 512     --results_dir /srv/home/users/loriss21cs/babylm/results     --save     --save_dir /srv/home/users/loriss21cs/babylm/models     --metrics accuracy f1 mcc     --metric_for_valid f1     --seed $SEED     --verbose

python -m evaluation_pipeline.finetune.run     --model_name_or_path "$MODEL_PATH"     --train_data "evaluation_data/full_eval/glue_filtered/mnli.train.jsonl"     --valid_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl"     --predict_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl"     --task mnli     --num_labels 3     --batch_size $BSZ     --learning_rate $LR     --num_epochs $MAX_EPOCHS     --sequence_length 512     --results_dir /srv/home/users/loriss21cs/babylm/results     --save     --save_dir /srv/home/users/loriss21cs/babylm/models     --metrics accuracy     --metric_for_valid accuracy     --seed $SEED     --verbose

