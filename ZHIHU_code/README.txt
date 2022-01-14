# TED

code for paper 'TED: Generating Consistent and Diverse Essays from User Specified Topics'



1.Software environment: see requirements.txt



2.Experiment:



1)train:  

1.see https://github.com/ghosthamlet/gpt2-ml-torch to download pretrained gpt-2 model SHA256:971f187fd72dc6bea547259daa92b2b5e73732825cc49b82c2d34bc20e39460d. Place it under TED/models/mega-clue-tok/checkpoint/  

2.dataset path is default in finetune_lm.py, it is not need to edit.
first stage for fine-tune:  

deepspeed --num_nodes 1 --num_gpus 1 finetune_lm.py --log_name finetune_large_stage1 --seq_len 300 --epochs 1 --batch_size 1 --lr 5e-5 --device_ids 0 --pretrained_path models/mega-clue-tok/checkpoint/ --freeze_body  

second stage for fine-tune:  
deepspeed --num_nodes 1 --num_gpus 1 finetune_lm.py --log_name finetune_large_stage2 --seq_len 300 --epochs 1 --batch_size 1 --lr 6e-5 --device_ids 0 --pretrained_path models/finetune_large_stage1_epoch_1



2)test:  
1.edit config.py to config: MODEL_PATH, TEST_OUTPUT_PATH, DETECTED_TOPIC_PATH, where:
#MODEL_PATH: when you run generatezhihu.py, this is the path need to be edited before to load trained model.
#TEST_OUTPUT_PATH: when you run evaluate.py and generatezhihu.py, this is the path need to be edited before to tell the program where is the generated essays
#DETECTED_TOPIC_PATH: when you run evaluate.py, this is the path need to be edited before to tell the program where is the predicted topic words of generated essays detected by SGM model
2.To get generated essays, run these commands:
cd  .../TED/ZHIHU_code/code
python generatezhihu.py
3.To get evaluation results, run these commands:
cd  .../TED/ZHIHU_code/code
python evaluate.py  
