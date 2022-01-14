#use absolute path to test, complete xxx with true path on your machine.
#MODEL_PATH: when you run generatezhihu.py, this is the path need to be edited before to load trained model
MODEL_PATH = '../models/mega-clue-tok/checkpoint/'
#TEST_OUTPUT_PATH: when you run bleu.py and generatezhihu.py, this is the path need to be edited before to tell the program where is the generated essays
TEST_OUTPUT_PATH = '../models/WT-GPT-2/output_data/output.json'
#DETECTED_TOPIC_PATH: when you run bleu.py, this is the path need to be edited before to tell the program where is the predicted topic words of generated essays detected by SGM model
DETECTED_TOPIC_PATH = '../models/WT-GPT-2/output_data/topics.txt'
#TEST_TOPIC_PATH: ground truth topic words of test dataset
TEST_TOPIC_PATH = '../datasets/test_topic.txt'
#TEST_REFERENCE_PATH: test dataset 
TEST_REFERENCE_PATH = '../datasets/test.json'
#TRAIN_REFERENCE_PATH: train dataset
TRAIN_REFERENCE_PATH = '../datasets/train.json'
