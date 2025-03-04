#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="hotfix canceled jobs"
#SBATCH --comment="Calculation of training data influence. Loads gradients from network share to /tmp/gradients, assumes /tmp is cleared after job completion"
#SBATCH --time=0-10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=325GB
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=galadriel,dgx-h100-em2
#SBATCH --gres=gpu:4


source /etc/profile.d/modules.sh

# Create a permanent environment with the name "my_new_permanent_environment"
export ENV_MODE="permanent"
export ENV_NAME="babylm_venv"
# This environment is now stored in $HOME/venvs/my_new_permanent_environment
 
# Load module miniforge3
module load miniforge

# df -h $TMPDIR/gradients
# rm -rf $TMPDIR/gradients


# df -h $TMPDIR/gradients

# conda env update --file environment.yml

# Run your python script



python extract_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_split=train[89%:90%] --paradigm=pre --mode=store --num_processes_gradients=4
python process_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_train_split=train[89%:90%] --dataset_test=anasedova/olmes_tulu_3_unseen --dataset_test_split=test[0%:100%] --mode=mean  --batch_size=10

python extract_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_split=train[54%:55%] --paradigm=pre --mode=store --num_processes_gradients=4
python process_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_train_split=train[54%:55%] --dataset_test=anasedova/tulu_3_underspecified_input_errors --dataset_test_split=train[0%:100%] --mode=mean  --batch_size=10


python extract_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_split=train[99%:100%] --paradigm=pre --mode=store --num_processes_gradients=4
python process_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_train_split=train[99%:100%] --dataset_test=anasedova/olmes_tulu_3_unseen --dataset_test_split=test[0%:100%] --mode=mean  --batch_size=10



python extract_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_split=train[5%:6%] --paradigm=pre --mode=store --num_processes_gradients=4
python process_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_train_split=train[5%:6%] --dataset_test=anasedova/tulu_3_incorrect_output_errors --dataset_test_split=train[0%:100%] --mode=mean  --batch_size=10


python extract_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_split=train[65%:66%] --paradigm=pre --mode=store --num_processes_gradients=4
python process_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_train_split=train[65%:66%] --dataset_test=anasedova/tulu_3_underspecified_input_errors --dataset_test_split=train[0%:100%] --mode=mean  --batch_size=10


python extract_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_split=train[7%:8%] --paradigm=pre --mode=store --num_processes_gradients=4
python process_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_train_split=train[7%:8%] --dataset_test=anasedova/tulu_3_incorrect_output_errors --dataset_test_split=train[0%:100%] --mode=mean  --batch_size=10


python extract_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_split=train[65%:66%] --paradigm=pre --mode=store --num_processes_gradients=4
python process_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_train_split=train[65%:66%] --dataset_test=anasedova/tulu_3_underspecified_input_errors --dataset_test_split=train[0%:100%] --mode=mean  --batch_size=10