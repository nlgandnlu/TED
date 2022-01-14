from code.config import MODEL_PATH
from code.config import TEST_TOPIC_PATH
from code.config import TEST_OUTPUT_PATH
from code.modeling_gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer, pipeline
import json
import torch
def readtest(filename):
    with open(filename, 'r', encoding ='utf-8') as f:
        ob=json.load(f)
    topic=ob.get('topic')
    realvalue = ob.get('content')
    length=ob.get('length')
    return topic,realvalue,length
def readtxt(filename):
    l=[]
    with open(filename, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            l.append(line)
    return l
def generate(**kwargs):
    model_path = kwargs.pop('model_path')
    model, tokenizer, info = build_model(model_path)
    return build_output(model, tokenizer, **kwargs)
def build_model(model_path):
    model, info = GPT2LMHeadModel.from_pretrained(model_path, output_loading_info=True)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer, info
def build_output(model, tokenizer,prompt, n_seq=3, max_len=300, gpu=-1, **kwargs):
    nlp = pipeline('text-generation',
            model=model, tokenizer=tokenizer,
            device=gpu)
    res = nlp(
        prompt,
        num_return_sequences=n_seq,
        max_length=max_len,
        do_sample=True,
        return_dict=False,
        **kwargs
    )

    return res

def generate_all(topicfile,resultfile,gpu,model_path):
    model, tokenizer, info = build_model(model_path)
    minlen=70
    max_repeat=5
    topic = readtxt(topicfile)
    memory = topic
    length = len(topic)
    now = 0
    dic = {}
    outputs = []

    for (topic_t, memory_t) in zip(topic, memory):
        #input of topic encoder
        topic_ids = torch.tensor(tokenizer.encode(topic_t, add_special_tokens=False)).unsqueeze(0).cuda(gpu)
        lent=0
        #construct prompt, input of decoder
        topic_t = topic_t.replace('[CLS][SEP]', '')
        topic_t = topic_t.replace('[SEP]','#')
        topic_t = topic_t+'[SEP]'
        num=0
        # use lent to easily control generating, use this way to control length of generated essays
        while lent<=minlen and num<max_repeat:
            res = build_output(model, tokenizer, prompt=topic_t, n_seq=1, max_len=120, gpu=gpu, zhuti_ids=topic_ids
                            ,local_rank=gpu)
            value = res[0]
            result = value['generated_text'][len(topic_t):]
            result = result.replace(' ', '')
            lent = len(result)
            num += 1
        print("now is (%d),and total is (%d)" % (now, length))
        print(topic_t)
        print(result)
        outputs.append(result)

        if now % 500 == 0:
            dic['length'] = now + 1
            dic['outputs'] = outputs
            with open(resultfile, 'w', encoding='utf-8') as f:
                json.dump(dic, f, ensure_ascii=False)
        if now == length - 1:
            dic['length'] = now + 1
            dic['outputs'] = outputs
            with open(resultfile, 'w', encoding='utf-8') as f:
                json.dump(dic, f, ensure_ascii=False)
        now = now + 1

if __name__ == '__main__':
    topicfile = TEST_TOPIC_PATH
    resultfile= TEST_OUTPUT_PATH
    model_path = MODEL_PATH
    gpu=0
    generate_all(topicfile,resultfile,gpu,model_path)
