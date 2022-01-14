from code.config import MODEL_PATH
from code.config import TEST_OUTPUT_PATH
from code.config import TEST_REFERENCE_PATH
from code.config import DETECTED_TOPIC_PATH
from code.config import TRAIN_REFERENCE_PATH
from transformers import BertTokenizer
import json
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
import jieba

def tihuan(l):
    new=[]
    for item in l:
        s="".join(item.split())
        s = s.replace('<UNK>', '')
        new.append(s)
    return new
def tihuan1(l):
    new=[]
    for item in l:
        s=item
        s=s.replace('、',' ')
        s=s.replace('。','')
        new.append(s)
    return new
def readsamples(filename):
    with open(filename, 'r', encoding ='utf-8') as f:
        ob = json.load(f)
    content = ob.get('outputs')
    content = tihuan(content)
    return content
def readtest(filename):
    with open(filename, 'r', encoding ='utf-8') as f:
        ob=json.load(f)
    topic=ob.get('topic')
    realvalue = ob.get('content')
    realvalue=tihuan(realvalue)
    topic=tihuan(topic)
    return topic,realvalue
def gettrain_worddic(tokenizer, train_txt):
    topic_train, train_value = readtest(train_txt)
    topictoken = []
    worddic = defaultdict(list)
    for t in topic_train:
        t=t.replace('、','')
        t=t.replace('。','')
        res = tokenizer.tokenize(t)
        res = tokenizer.convert_tokens_to_ids([x for x in res])
        res = sorted(res)
        topictoken.append(res)
    for w, real in zip(topictoken, train_value):
        s = "-".join([str(x) for x in w])
        worddic[s].append(real)
    return worddic
def bleu(tokenizer, sample_txt, test_txt):
    samples = readsamples(sample_txt)
    topic, realvalue = readtest(test_txt)
    samples=samples[:len(topic)]
    topictoken = []
    realtoken = []
    samplestoken = []
    bleu_value=[]
    # data prase
    for v in realvalue:
        res = tokenizer.tokenize(v)
        res = tokenizer.convert_tokens_to_ids([x for x in res])
        realtoken.append(res)
    for t in topic:
        res = tokenizer.tokenize(t)
        res = tokenizer.convert_tokens_to_ids([x for x in res])
        res = sorted(res)
        topictoken.append(res)
    for s in samples:
        res1 = tokenizer.tokenize(s)
        res1 = tokenizer.convert_tokens_to_ids([x for x in res1])
        samplestoken.append(res1)
    worddic = defaultdict(list)
    worddic2 = defaultdict(list)
    for w, real, real1 in zip(topictoken, realtoken, realvalue):
        s = "-".join([str(x) for x in w])
        worddic[s].append(real)
        worddic2[s].append(real1)
    # cacu bleu
    total_bleuo1 = 0
    total_bleuo2 = 0
    sm = SmoothingFunction()
    for w, v in zip(topic, samplestoken):
        ww = tokenizer.tokenize(w)
        ww = tokenizer.convert_tokens_to_ids([x for x in ww])
        ww = sorted(ww)
        ww = '-'.join([str(x) for x in ww])
        temp1= sentence_bleu(worddic[ww], v, weights=(1, 0, 0, 0), smoothing_function=sm.method1)
        total_bleuo1 += temp1
        temp2 = sentence_bleu(worddic[ww], v, weights=(0, 1, 0, 0), smoothing_function=sm.method1)
        total_bleuo2 += temp2
    total_bleuo1 = (total_bleuo1 / len(topic)) * 100
    total_bleuo2 = (total_bleuo2 / len(topic)) * 100
    print("BLEU-1 (1 0 0 0):(%f)" % total_bleuo1)
    print("BLEU-2 (0 1 0 0):(%f)" % total_bleuo2)
    return topic,samples,bleu_value,worddic2,samplestoken
def dist(outputfile):
    samples = readsamples(outputfile)
    unigrams = dict()
    bigrams = dict()
    wordsNum = 0
    sentencesNum = 0
    for resultLine in samples:
        resultWords = resultLine.split()
        if len(resultWords) == 0:
            continue
        sentencesNum += 1
        if sentencesNum % 100 == 0:
            print('finish:%d' % (sentencesNum))
        # caculate Distinct1 and Distinct2
        words = resultWords
        wlen = len(words)
        wordsNum += wlen
        for i in range(wlen):
            word = words[i]
            # caculate number of unigrams
            if word not in unigrams:
                print(word)
                unigrams[word] = 1
            else:
                unigrams[word] = unigrams[word] + 1
            # caculate number of bigrams
            if i < wlen - 1:
                word = words[i] + ' ' + words[i + 1]
            if word not in bigrams:
                bigrams[word] = 1
            else:
                bigrams[word] = bigrams[word] + 1
    dist1=len(unigrams)/wordsNum
    dist2=len(bigrams)/wordsNum
    return dist1,dist2
def distinct(seqs):
    """ Calculate intra/inter distinct 1/2. """
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)
    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    print("Diversity-1 :(%f)" % inter_dist1)
    print("Diversity-2 :(%f)" % inter_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2
def Jaccrad(model, reference):
    terms_reference= jieba.cut(reference)
    terms_model= jieba.cut(model)
    grams_reference = set(terms_reference)
    grams_model = set(terms_model)
    temp=0
    for i in grams_reference:
        if i in grams_model:
            temp=temp+1
    fenmu=len(grams_model)+len(grams_reference)-temp
    jaccard_coefficient=float(temp/fenmu)
    return jaccard_coefficient
def novelty(topic, samples1, worddic_train, tokenizer):
    novel_all = 0
    now=0
    worddic = defaultdict(list)
    print('novelty-stage1, this need some time')
    for t in topic:
        t = t.replace('、', '')
        t = t.replace('。', '')
        ww = tokenizer.tokenize(t)
        ww = tokenizer.convert_tokens_to_ids([x for x in ww])
        ww = sorted(ww)
        ww = '-'.join([str(x) for x in ww])
        for w in worddic_train.keys():
            if Jaccrad(w, ww) > 0.8:
                worddic[ww].extend(worddic_train[w])
        now = now + 1
    print('novelty-stage2, this need some time')
    for t,s in zip(topic,samples1):
        t=t.replace('、','')
        t=t.replace('。','')
        ww = tokenizer.tokenize(t)
        ww = tokenizer.convert_tokens_to_ids([x for x in ww])
        ww = sorted(ww)
        ww = '-'.join([str(x) for x in ww])
        max=0
        for refer in worddic[ww]:
            jaccard_coefficient = Jaccrad(s, refer)
            if jaccard_coefficient > max:
                max = jaccard_coefficient
        novel_all += 1-max
    novel_ave = novel_all / len(topic)
    print("Novelty :(%f)" % novel_ave)
    return novel_ave
def consistence(pre_t, topic):
    pre_tlist=[]
    topiclist=[]
    consis_all=0
    for item1,item2 in zip(pre_t, topic):
        pre_tlist.append([x for x in item1.split()])
        topiclist.append([x for x in item2.split()])
    for t,s in zip(pre_tlist,topiclist):
        ret1= [x for x in t if x in s]
        ret2= list(set(t).union(set(s)))
        consis_all += len(ret1)/len(ret2)
    consis_ave = consis_all / len(topic)
    print("Consistency :(%f)" % consis_ave)
if __name__ == '__main__':

    #build tokenlizer from pre-trained path
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    # bleu
    topic, samples1, bleu1, worddic_test, samplestoken=bleu(tokenizer,TEST_OUTPUT_PATH,TEST_REFERENCE_PATH)

    # consitence : when you get the topic_file(containing topics detected by SGM) at position of DETECTED_TOPIC_PATH,
    # you can use these codes to caculate topic-sonsistency
    with open(DETECTED_TOPIC_PATH, 'r', encoding='utf-8') as f:
        pre_topic = f.readlines()
    consistence(pre_topic, tihuan1(topic))

    #dist1,2
    a, b, c, d = distinct(samplestoken)

    #novelty
    worddic_train = gettrain_worddic(tokenizer,TRAIN_REFERENCE_PATH)
    novelty(topic, samples1, worddic_train, tokenizer)






