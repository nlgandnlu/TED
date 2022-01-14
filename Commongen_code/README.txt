# TED

code for paper 'TED: Generating Consistent and Diverse Essays from User Specified Topics'



1.Software environment

1)use these commands to install:
conda create -n commongen python=3.7
conda activate commongen
pip install torch==1.4.0
git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext
cd transformers_local ; pip install --editable .

2.Experiment:



1)use these commands to train TED:
cd .../TED/
python transformers_local/examples/run_language_modeling.py   --output_dir tmp/ted_contrastce   --model_type=gpt2   --model_name_or_path=gpt2   --do_train   --do_eval   --evaluate_during_training    --train_data_file=train.txt   --eval_data_file=dev.txt  --line_by_line --block_size 128   --num_train_epochs 20   --learning_rate 5e-5   --warmup_steps 2000   --logging_steps 50   --save_steps 50   --per_gpu_train_batch_size 120  --gradient_accumulation_steps 1 --overwrite_output_dir --contrast_ce

2)use this command to generate sentences:
python transformers_local/examples/run_generation.py   --model_type=gpt2   --model_name_or_path=tmp/ted_contrastce --input_file inference_test.txt --output_file decode_result.txt

3)use these commands to evaluate results:
source activate nlpscore
cd .../evaluation/Traditional/eval_metrics/
python eval.py --key_file ../../../dataset/final_data/commongen/commongen.test.src_alpha.txt --gts_file ../../../dataset/final_data/commongen/commongen.test.tgt.txt --res_file ../../../methods/TED/new.gpt2.test