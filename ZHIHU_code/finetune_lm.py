import os
import json
import time
import random
import argparse
import shutil
import logging

import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.engine import DeepSpeedEngine
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer
import transformers
from transformers import GPT2Config, get_linear_schedule_with_warmup, \
    BertTokenizer, BertTokenizerFast

from gpt2_ml_torch.modeling_gpt2 import GPT2LMHeadModel
import torch.nn.functional as FF
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
"""
文本生成任务微调

必须先安装 deepspeed==0.3.7

数据格式查看get_configs函数内train_data命令行参数的注释
datasets/目录下有示例数据文件


测试代码：
deepspeed --num_nodes 1 --num_gpus 1 finetune_lm.py --log_name testtest --seq_len 300 --epochs 2 --batch_size 1 --lr 5e-5 --device_ids 0,1 --train_data datasets/test_train.txt --valid_data datasets/test_val.txt --model_config configs/small.json --vocab models/mega-clue-tok/vocab.txt --max_data_len 1000 --no_cache


微调第一阶段：（正式实验阶段：无知识）
编码器权重初始化，门控机制，batch设计，知识图。
ps aux|grep python|grep -v grep|cut -c 9-15|xargs kill -9
deepspeed --num_nodes 1 --num_gpus 2 finetune_lm.py --log_name finetune_large_stage1_e74_enc2 --seq_len 100 --epochs 3 --batch_size 2 --lr 10e-8 --device_ids 0,1 --pretrained_path models/mega-clue-tok --freeze_body 门控机制
deepspeed --num_nodes 1 --num_gpus 2 finetune_lm.py --log_name finetune_large_stage2_2e8 --seq_len 100 --epochs 2 --batch_size 1 --lr 2e-8 --device_ids 0,1 --pretrained_path models/finetune_large_stage1_e7_epoch_1 --freeze_body 门控机制
deepspeed --num_nodes 1 --num_gpus 2 finetune_lm.py --log_name finetune_large_stage3_2e7 --seq_len 100 --epochs 5 --batch_size 1 --lr 10e-8 --device_ids 0,1 --pretrained_path models/finetune_large_stage2_2e8_epoch_2
微调第二阶段：
deepspeed --num_nodes 1 --num_gpus 2 finetune_lm.py --log_name finetune_large_stage32 --seq_len 100 --epochs 10 --batch_size 4 --lr 5e-5 --device_ids 0,1 --pretrained_path models/finetune_large_stage1_e74_enc4_epoch_1

"""


def get_configs():
    parser = argparse.ArgumentParser(description='GPT2')
    parser.add_argument("--lr", type=float, default=5e-5, metavar="N", help="学习率")
    parser.add_argument('--warmup_steps', default=200, type=int, required=False, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, metavar="N", help="")

    parser.add_argument('--model_config', type=str, default='/home1/htbai/gongwei/gpt2-ml-torch-master/configs/config.json', help="测试用的模型配置文件")
    parser.add_argument('--vocab', type=str, default='/home1/htbai/gongwei/gpt2-ml-torch-master/configs/vocab.txt', help="测试用的字典")
    parser.add_argument('--pretrained_path', type=str, default='',
                        help="预训练模型目录，默认为空时，用model_config和vocab参数初始化模型从头训练，可用于快速测试代码")
    parser.add_argument('--train_labeldata', type=str, default='datasets/train_label.txt', required=False,
                        help="训练标签数据文件，普通的以\n分行的txt文件，必须是utf8格式")
    parser.add_argument('--valid_labeldata', type=str, default='datasets/val_label.txt', required=False,
                        help="验证标签数据文件，普通的以\n分行的txt文件，必须是utf8格式")
    parser.add_argument('--train_zhutidata', type=str, default='datasets/train_topic.txt', required=False,
                        help="训练主题数据文件，普通的以\n分行的txt文件，必须是utf8格式")
    parser.add_argument('--valid_zhutidata', type=str, default='datasets/val_topic.txt', required=False,
                        help="验证主题数据文件，普通的以\n分行的txt文件，必须是utf8格式")
    parser.add_argument('--train_memorydata', type=str, default='datasets/train_indexs.txt', required=False,
                        help="训练知识数据文件，普通的以\n分行的txt文件，必须是utf8格式")
    parser.add_argument('--valid_memorydata', type=str, default='datasets/val_indexs.txt', required=False,
                        help="验证知识数据文件，普通的以\n分行的txt文件，必须是utf8格式")
    parser.add_argument('--train_memorymask', type=str, default='datasets/train_memory_mask.txt', required=False,
                        help="训练知识掩码文件，普通的以\n分行的txt文件，必须是utf8格式")
    parser.add_argument('--valid_memorymask', type=str, default='datasets/val_memory_mask.txt', required=False,
                        help="验证知识掩码文件，普通的以\n分行的txt文件，必须是utf8格式")
    parser.add_argument('--train_zhutismask', type=str, default='datasets/train_topic_mask.txt', required=False,
                        help="训练主题掩码文件，普通的以\n分行的txt文件，必须是utf8格式")
    parser.add_argument('--valid_zhutismask', type=str, default='datasets/val_topic_mask.txt', required=False,
                        help="验证主题掩码文件，普通的以\n分行的txt文件，必须是utf8格式")
    parser.add_argument('--freeze_body', action='store_true', help="是否禁止微调模型主体，只微调编码器和最后一层。微调可分为两个阶段，第一阶段启用这个参数")
    parser.add_argument('--contrast', action='store_true', help="Whether to use contrast learning.")
    parser.add_argument('--contrast_ce', action='store_true', help="Whether to use contrast learning and ce loss together.")
    parser.add_argument('--gradient_accumulation_step', default=1, type=int, required=False, help="")

    parser.add_argument("--max_data_len", type=int, metavar="N", help="最大训练多少份数据，默认全部，输入较小的数字以快速测试代码")
    parser.add_argument('--log_name', type=str, required=True, help="日志名字，字母或数字，不包含特殊字符或中文")

    parser.add_argument('--no_cache', action='store_true', help="是否禁止缓存数据集的预处理操作")

    parser.add_argument('--device_ids', default='0', type=str, required=False, help="可见的GPU设备，如：0,1,2")
    parser.add_argument('--no_cuda', action='store_true', help="禁止GPU")

    parser.add_argument("--seq_len", type=int, default=300, metavar="N", help="输入长度")
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="训练轮次")
    parser.add_argument(
        "--batch_size", type=int, default=1, metavar="N", help="单个GPU上的批次大小"
    )
    parser.add_argument('--seed', type=int, default=62, help='')

    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    os.environ.setdefault('MASTER_PORT', '3600')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    # deepspeed launcher will setup these, so default values no effects
    os.environ.setdefault('WORLD_SIZE', str(len(args.device_ids.split(','))))
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.device_ids)

    args.deepspeed = True
    args.cpu_optimizer = True

    args.rank = int(os.getenv('RANK'))
    args.world_size = int(os.getenv("WORLD_SIZE"))

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = 'cuda' if args.cuda else 'cpu'

    ds_config = {
        'zero_optimization': {
            'stage': 2,
            'cpu_offload': True,
            'contiguous_gradients': True,
            # https://github.com/microsoft/DeepSpeed/issues/467
            'overlap_comm': False,
            # 'reduce_bucket_size': 50000000
            # too small will failed with large dimension size
            'reduce_bucket_size': 10000000,
            'allgather_bucket_size': 10000000
        },
        'train_batch_size': args.batch_size * args.world_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        'fp16': {
            'enabled': True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": True
        },
        "wall_clock_breakdown": False,
    }

    return args, ds_config


def set_random_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        # Disable CuDNN
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        # XXX: for LM with almost same length input/output all the time, enable this
        torch.backends.cudnn.benchmark = False


def create_logger(log_path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    log_fname = log_path + '/' + name + '.log'
    file_handler = logging.FileHandler(filename=log_fname)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


class LMDataset(torch_data.Dataset):
    def __init__(self, args, mode, data_path, label_path, memory_path ,memory_mask_path, zhutis_mask_path, tokenizer):
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.tokenizer = tokenizer
        self.max_data_len = args.max_data_len
        self.mode = mode
        self.data_path = data_path

        dist.barrier()
        if args.local_rank != 0:
            dist.barrier()
        self.zhutis = list(self._convert_to_features(self._get_exmples(data_path), cls=True))
        self.labels = list(self._convert_to_features(self._get_exmples(label_path), cls=False))
        self.indexs = self._get_indexs(memory_path)
        if args.local_rank == 0:
            dist.barrier()
    def _get_indexs(self, path):
        with open(path, encoding='utf-8') as f:
            lines=f.readlines()
        lines=[[int(line)] for line in lines]
        return lines
    def _get_exmples(self, path1):
        path = path1
        with open(path, encoding="utf-8") as f:
            for line in tqdm(f.read().split('\n'), ascii=True):
                line = line.strip()
                if line == '':
                    continue
                yield line

    def _convert_to_features(self, examples, cls=True):
        def fn():
            for i, line in enumerate(tqdm(examples, ascii=True)):
                if self.max_data_len is not None and i == self.max_data_len:
                    break
                yield self.tokenizer.tokenize(line)

        for arr in fn():
            if cls:
                ids = self.tokenizer.convert_tokens_to_ids([v for v in arr])
            else:
                ids = self.tokenizer.convert_tokens_to_ids([v for v in arr])
            yield ids

    def __len__(self):
        return len(self.zhutis)

    def __getitem__(self, index):
        return [torch.tensor(self.zhutis[index], dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long), torch.tensor(self.indexs[index], dtype=torch.long)]
        #return [self.zhutis[index], self.labels[index], torch.tensor(self.indexs[index],dtype=torch.long)]


def collate_fn(data):
    zhutis, examples, indexs = zip(*data)
    new_input_zhutis = []
    attention_masks = []
    f_attention_masks = []
    position_ids = []
    eos_position = get_eos(examples)
    eos_position = _tensorize_batch(eos_position)
    batch = _tensorize_batch(examples)
    indexs = _tensorize_batch(indexs)
    for zhuti in zhutis:
        attention_mask, position_id, input_ids = get_mask_and_position(zhuti)
        f_attention_mask = torch.ones(batch.shape[-1], input_ids.shape[-1])
        new_input_zhutis.append(input_ids)
        attention_masks.append(attention_mask)
        position_ids.append(position_id)
        f_attention_masks.append(f_attention_mask)
    final_zhutis = _tensorize_batch(new_input_zhutis)
    final_attention_mask = _pad2d_batch(attention_masks)
    finalf_attention_mask = _pad2d_batch(f_attention_masks)
    final_position_ids = _pad_batch(position_ids)
    # print(type(batch))
    # print(batch.shape)
    # print(type(final_zhutis))
    # print('input_ids')
    # print(batch)
    # print('zhutis')
    # print(final_zhutis)
    # print('e_attention_mask')
    # print(final_attention_mask)
    # print("e_position_ids")
    # print(final_position_ids)
    # print("f_attention_mask")
    # print(finalf_attention_mask)
    # print("indexs")
    # print(indexs)
    # print('eos_position')
    # print(eos_position)
    if args.contrast_ce:
        return {"input_ids": batch, "labels": batch, "zhutis": final_zhutis,
                "e_attention_mask": final_attention_mask, "e_position_ids": final_position_ids,
                "f_attention_mask": finalf_attention_mask, "indexs": indexs, 'eos_position': eos_position}
    elif args.contrast:
        return {"input_ids": batch, "zhutis": final_zhutis,
                "e_attention_mask": final_attention_mask, "e_position_ids": final_position_ids,
                "f_attention_mask": finalf_attention_mask, "indexs": indexs, 'eos_position': eos_position}
    else:
        return {"input_ids": batch, "labels": batch, "zhutis": final_zhutis, "e_attention_mask": final_attention_mask,
                "e_position_ids": final_position_ids, "f_attention_mask": finalf_attention_mask}

def get_eos(examples):
    return [torch.tensor([x.shape[0]-1],dtype=torch.long) for x in examples]

def _pad2d_batch(examples) -> torch.Tensor:
    max_len1=0
    max_len0 = 0
    batch_outputs=[]
    for e in examples:
        if e.shape[-1]>max_len1:
            max_len1=e.shape[-1]
        if e.shape[-2]>max_len0:
            max_len0=e.shape[-2]
    for e in examples:

        batch_outputs.append( torch.nn.ZeroPad2d(padding=(0, max_len1-e.shape[1], 0, max_len0 - e.shape[0]))(e).unsqueeze(0))
    cat = torch.cat((batch_outputs), dim=0)
    return cat
def _pad_batch(examples) -> torch.Tensor:
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)
    else:
        return pad_sequence(examples, batch_first=True, padding_value=0)
def _tensorize_batch(examples) -> torch.Tensor:
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)
    else:
        return pad_sequence(examples, batch_first=True, padding_value=8022)

def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

def get_mask_and_position(ids):
    # 输入topic信息而不是单独的tokenembedding,顿号544,句号545
    length=ids.shape[-1]
    num_sep=0
    for i in range(length):
        if ids[i].item() == 102:
            num_sep+=1
    length_withoutsep=length-num_sep
    #id groups
    id_groups=[]
    #token groups
    token_groups=[]
    # print('local_rank')
    # print(local_rank)
    mask=torch.zeros([length_withoutsep,length_withoutsep])
    position=torch.arange(0, length_withoutsep, dtype=torch.long)

    #to get relation_position encoding
    relative_position=0
    # to determine whether come to the last token
    end_flag=True

    seps_now=0
    #[cls]
    id_groups.append(torch.tensor([0]))
    token_groups.append(ids.index_select(-1, torch.tensor([0])))
    mask[0, :] = 1.0
    mask[:, 0] = 1.0

    for i in range(length):
        if ids[i].item() == 102:
            seps_now+=1
            ids_before=[]
            ids_convert=[]
            head=i
            for tail in range(i+1,length):
                if ids[tail].item() == 102:
                    end_flag=False
                    break
                ids_convert.append(tail - seps_now)
                ids_before.append(tail)
            id_groups.append(torch.tensor(ids_convert))
            token_groups.append(ids.index_select(-1, torch.tensor(ids_before)))
            if end_flag==False:
                mask[head+1-seps_now:tail-seps_now,head+1-seps_now:tail-seps_now]=1.0
            else:
                mask[head+1-seps_now:tail+1-seps_now, head+1-seps_now:tail+1-seps_now] = 1.0

            # relative position record starts again
            relative_position = 1
            end_flag=True
        else:
            position[i-seps_now]=relative_position
            relative_position+=1
    new_input_ids=torch.cat((token_groups),-1)
    return mask,position,new_input_ids

def build_model(args):
    if args.pretrained_path == '':
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # XXX: must add this, or can't tokenize special token in string to single char
        special_tokens_dict = {'cls_token':'[CLS]','bos_token': '[BOS]','sep_token' : '[SEP]'}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        print('We have added', num_added_toks, 'tokens')
        model.resize_token_embeddings(len(tokenizer))
        print(len(tokenizer))
        model.set_input_embeddings(model.transformer.wte)
        info=None
    else:
        config = GPT2Config.from_pretrained(args.pretrained_path)
        model, info = GPT2LMHeadModel.from_pretrained(args.pretrained_path,
                                                      config=config, output_loading_info=True)
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_path)
        #tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_path)
        if tokenizer.eos_token==None:
            special_tokens_dict = {'bos_token': '[BOS]','eos_token': '[EOS]'}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print(tokenizer.eos_token_id)
            print(tokenizer.bos_token_id)
            print('We have added', num_added_toks, 'tokens')
            # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
            model.resize_token_embeddings(len(tokenizer))
            print(len(tokenizer))
            model.set_input_embeddings(model.transformer.wte)
            assert tokenizer.eos_token == '[EOS]'

    return model, tokenizer, info


def get_model_tokenizer_optimizer(args):
    model, tokenizer, _ = build_model(args)
    model.half()
    model.cuda(args.local_rank)

    model = DDP(model, device_ids=[args.local_rank],
                output_device=args.local_rank)

    model_obj = model.module
    # print("****************************************************************************************")
    # for p in model_obj.state_dict():
    #     print(p)

    if args.freeze_body:
        model_obj.transformer.requires_grad_(False)
        model_obj.transformer.wpe.requires_grad_(False)
        model_obj.transformer.emb_norm.requires_grad_(False)
        model_obj.lm_head.weight.requires_grad_(False)
        model_obj.encoder.h.requires_grad_(True)
        model_obj.encoder.emb_norm.requires_grad_(True)
        model_obj.transformer2.requires_grad_(True)   #has buxuyao chuan di de canshu
        params = [dict(params=v)
                  for v in [
                      # wte is tie with lm_head, no need run requires_grad_
                      # don't put wte in optim, params can't dup,
                      # and autodiff will calc grads two times on params in lm_head
                      # model.module.transformer.wte.parameters(),
                      #model_obj.transformer.wpe.parameters(),
                      #model_obj.transformer.emb_norm.parameters(),
                      #model_obj.lm_head.weight,
                      model_obj.encoder.h.parameters(),
                      model_obj.encoder.emb_norm.parameters(),
                      #model_obj.knowledge.h.parameters(),
                      #model_obj.knowledge.emb_norm.parameters(),
                      model_obj.transformer2.parameters(),
                  ]]
    else:
        model.requires_grad_(True)
        params = model_obj.parameters()

    optimizer = DeepSpeedCPUAdam(params, lr=args.lr)
    return model, tokenizer, optimizer


def get_data_loader(args, tokenizer):
    def fn(mode, data_path, label_path, memory_path, memory_mask_path, zhutis_mask_path):
        dataset = LMDataset(args, mode, data_path, label_path, memory_path, memory_mask_path, zhutis_mask_path,tokenizer)
        sampler = torch_data.distributed.DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True,
        )
        data_loader = torch_data.DataLoader(
            dataset=dataset,
            batch_size=int(args.batch_size / args.gradient_accumulation_steps),
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=sampler)

        return data_loader, sampler

    train = fn('train', args.train_zhutidata, args.train_labeldata, args.train_memorydata, args.train_memorymask, args.train_zhutismask)
    valid = fn('valid', args.valid_zhutidata, args.valid_labeldata, args.valid_memorydata, args.valid_memorymask, args.valid_zhutismask)
    return (*train, *valid)


def get_scheduler(args, optimizer, data_loader):
    total_steps = int(len(data_loader.dataset) * args.epochs / args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    return scheduler


def train_epoch(args, logger, model, data_loader, valid_data_loader, start, writer, optimizer,
                scheduler, epoch=0, log_interval=10):
    model.train()

    loss_acc = 0
    oom_time = 0
    samples_so_far = 0
    batch_len = len(data_loader)
    data_len = len(data_loader.dataset)
    ce_loss=None
    global_step = 0
    loss_batch=0

    valid_loss = eval_epoch(args, logger, model, valid_data_loader,epoch)
    end = time.time()
    if args.local_rank == 0:
        logger.info("Epoch took: {:.3f}s, valid loss: {:.6f}".format(
            end - start, valid_loss))
    for batch_idx, inputs in enumerate(data_loader):
        # if args.contrast_ce:
        #     pass
        # elif args.contrast:
        #     inputs.pop('labels')
        # else:
        #     inputs.pop('indexs')
        #     inputs.pop('eos_position')

        for k, v in inputs.items():
            # print(k)
            # print(v)
            inputs[k] = v.to(args.local_rank)
        inputs['local_rank']=args.local_rank
        if args.contrast_ce:
            loss,ce_loss = compute_contrast_ce(model, inputs, return_outputs=False)
        elif args.contrast:
            loss = compute_loss(model, inputs, return_outputs=False)
        else:
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        loss_batch+= loss.item()/args.gradient_accumulation_step
        loss_acc += loss.item()/args.gradient_accumulation_step
        model.backward(loss/args.gradient_accumulation_step)
        if (batch_idx + 1) % args.gradient_accumulation_step == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(data_loader) <= args.gradient_accumulation_step
                and (batch_idx + 1) == len(data_loader)
        ):
            global_step += 1
            model.step()
            print(loss_batch)
            if args.local_rank == 0:
                samples_so_far += inputs['input_ids'].shape[0]
                if global_step % log_interval == 0:
                    logger.info(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch + 1,
                            samples_so_far,
                            data_len,
                            100 * samples_so_far / data_len,
                            loss_batch,
                        )
                    )
                if ce_loss != None:
                    print('ce_loss')
                    print(ce_loss)
                writer.add_scalar('train_loss', loss_batch, batch_idx)

                if global_step != 0 and global_step % 500 == 0:
                    save_model(args, logger, None, model, epoch, global_step)
            if global_step % 500 == 0:
                valid_loss = eval_epoch(args, logger, model, valid_data_loader,
                                        epoch)
                end = time.time()
                if args.local_rank == 0:
                    logger.info("Epoch took: {:.3f}s, valid loss: {:.6f}".format(
                        end - start, valid_loss))
            loss_batch=0
        del inputs, loss
    return loss_acc / global_step

def compute_contrast_ce(
        model: nn,
        inputs,
        return_outputs= False,
        pooling_strategy='pooler',
        w_drop_out = [0.2],
        temperature = 0.05,
        def_drop_out = 0.1,
):
    """ Computes loss given model and its inputs.

    :param model: The model to be finetuned.

    :param inputs: The inputs

    """
    labels = inputs["indexs"]
    #print(inputs)
    # ----- Default p = 0.1 ---------#
    output = model(**inputs)

    loss=output['ce_loss']
    loss1=loss.clone()
    if pooling_strategy == 'pooler':
        try:
            logits = output['pooler_output'].unsqueeze(1)
        except:
            logits = output['last_hidden_state'].mean(dim=1, keepdim=True)
    else:
        logits = output['last_hidden_state'].mean(dim=1, keepdim=True)

    # ---- iteratively create dropouts -----#
    for p_dpr in w_drop_out:
        # -- Set models dropout --#
        if p_dpr != def_drop_out:
            model = set_dropout_mf(model, w=p_dpr)
        # ---- concat logits ------#
        if pooling_strategy == 'pooler':
            # --------- If model does offer pooler output --------#
            try:
                logits = torch.cat((logits, model(**inputs)['pooler_output'].unsqueeze(1)), 1)
            except:
                logits = torch.cat((logits, model(**inputs)['last_hidden_state'].mean(dim=1, keepdim=True)), 1)
        else:
            logits = torch.cat((logits, model(**inputs)['last_hidden_state'].mean(dim=1, keepdim=True)), 1)

    # ---- L2 norm ---------#
    logits = FF.normalize(logits, p=2, dim=2)

    # ----- Set model back to dropout = 0.1 -----#
    if p_dpr != def_drop_out: model = set_dropout_mf(model, w=0.1) #之前是这句model = set_dropout_mf(model, w=0.1)

    # SupContrast
    loss_fn = SupConLoss(temperature=temperature)  # temperature=0.1

    loss += 0.1*loss_fn(logits, labels)  # added rounding for stsb

    return (loss, output) if return_outputs else (loss,loss1)

def compute_loss(
        model: nn,
        inputs,
        return_outputs = False,
        pooling_strategy='pooler',
        w_drop_out= [0.2],
        temperature= 0.05,
        def_drop_out= 0.1,
) :
    """ Computes loss given model and its inputs.

    :param model: The model to be finetuned.

    :param inputs: The inputs

    """
    labels = inputs["indexs"]
    #print(inputs)
    # ----- Default p = 0.1 ---------#
    output = model(**inputs)
    if pooling_strategy == 'pooler':
        try:
            logits = output['pooler_output'].unsqueeze(1)
        except:
            logits = output['last_hidden_state'].mean(dim=1, keepdim=True)
    else:
        logits = output['last_hidden_state'].mean(dim=1, keepdim=True)

    # ---- iteratively create dropouts -----#
    for p_dpr in w_drop_out:
        #print(model.module.module.encoder.drop.p)
        # -- Set models dropout --#
        if p_dpr != def_drop_out:
            #print(model.module.module.encoder.drop.p)
            model = set_dropout_mf(model, w=p_dpr)
            #print(model.module.module.encoder.drop.p)
        # ---- concat logits ------#
        if pooling_strategy == 'pooler':
            #print(model.transformer.drop.p)
            # --------- If model does offer pooler output --------#
            try:
                #print(model.module.module.encoder.drop.p)
                #print(logits)
                #print(model(**inputs)['pooler_output'])
                logits = torch.cat((logits, model(**inputs)['pooler_output'].unsqueeze(1)), 1)
            except:
                logits = torch.cat((logits, model(**inputs)['last_hidden_state'].mean(dim=1, keepdim=True)), 1)
        else:
            logits = torch.cat((logits, model(**inputs)['last_hidden_state'].mean(dim=1, keepdim=True)), 1)

    # ---- L2 norm ---------#
    logits = FF.normalize(logits, p=2, dim=2)

    # ----- Set model back to dropout = 0.1 -----#
    if p_dpr != def_drop_out: model = set_dropout_mf(model, w=0.1)

    # SupContrast
    loss_fn = SupConLoss(temperature=temperature)  # temperature=0.1

    loss = loss_fn(logits, labels)  # added rounding for stsb

    return (loss, output) if return_outputs else loss

def set_dropout_mf(
        model: nn,
        w
):
    """Alters the dropouts in the embeddings.
    """
    # ------ set hidden dropout -------#

    # print("****************************************************************************************")
    # for p in model_obj.state_dict():
    #     print(p)
    if hasattr(model, 'module'):
        model_obj = model.module.module
        model_obj.transformer.drop.p = w
        model_obj.encoder.drop.p = w
        for i in model_obj.transformer.h:
            i.attn.attn_dropout.p = w
            i.attn.resid_dropout.p = w
            i.mlp.dropout.p = w
        for i in model_obj.encoder.h:
            i.attn.attn_dropout.p = w
            i.attn.resid_dropout.p = w
            i.mlp.dropout.p = w
        for i in model_obj.transformer2:
            i.attn1.attn_dropout.p = w
            i.attn1.resid_dropout.p = w
            i.attn21.attn_dropout.p = w
            i.attn21.resid_dropout.p = w
            i.mlp21.dropout.p = w
            i.mlp3.dropout.p = w
    else:
        model.transformer.drop.p = w
        model.encoder.drop.p=w
        for i in model.transformer.h:
            i.attn.attn_dropout.p = w
            i.attn.resid_dropout.p = w
            i.mlp.dropout.p = w
        for i in model.encoder.h:
            i.attn.attn_dropout.p = w
            i.attn.resid_dropout.p = w
            i.mlp.dropout.p = w
        for i in model.transformer2:
            i.attn1.attn_dropout.p = w
            i.attn1.resid_dropout.p = w
            i.attn21.attn_dropout.p = w
            i.attn21.resid_dropout.p = w
            i.mlp21.dropout.p = w
            i.mlp3.dropout.p = w

    return model

def eval_epoch(args, logger, model, data_loader,
               epoch=0, log_interval=10):
    #model.eval()

    loss_acc = 0
    samples_so_far = 0
    ce_loss=None
    batch_len = len(data_loader)
    data_len = len(data_loader.dataset)
    with torch.no_grad():
        for batch_idx, inputs in enumerate(data_loader):
            # if args.contrast_ce:
            #     pass
            # elif args.contrast:
            #     inputs.pop('labels')
            # else:
            #     inputs.pop('indexs')
            #     inputs.pop('eos_position')
            for k, v in inputs.items():
                # print(k)
                # print(v)
                inputs[k] = v.to(args.local_rank)
            inputs['local_rank'] = args.local_rank
            if args.contrast_ce:
                loss, ce_loss = compute_contrast_ce(model, inputs, return_outputs=False)
            elif args.contrast:
                loss = compute_loss(model, inputs, return_outputs=False)
            else:
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            loss = average_distributed_scalar(args, loss.item())
            loss_acc += loss
            if args.local_rank == 0:
                samples_so_far += inputs['input_ids'].shape[0]
                if batch_idx % log_interval == 0:
                    logger.info(
                        "Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch + 1,
                            samples_so_far,
                            data_len,
                            100 * samples_so_far / data_len,
                            loss,
                        )
                    )
                if ce_loss != None:
                    print('ce_loss')
                    print(ce_loss)
    torch.cuda.empty_cache()
    return loss_acc/ batch_len


def average_distributed_scalar(args, scalar):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / dist.get_world_size()
    dist.all_reduce(scalar_t, op=dist.ReduceOp.SUM)
    return scalar_t.item()


def train(args, ds_config):

    dist_init(args)

    set_random_seed(args)

    logger = create_logger('logs/', args.log_name)

    model, tokenizer, optimizer = get_model_tokenizer_optimizer(args)


    # tokenizer.add_special_tokens({'bos_token':'<BOS>'})
    data_loader, sampler, valid_data_loader, valid_sampler = get_data_loader(args, tokenizer)

    scheduler = get_scheduler(args, optimizer, data_loader)

    model, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        args=args,
        config_params=ds_config
    )
    writer = SummaryWriter()
    for epoch in range(args.epochs):
        start = time.time()
        sampler.set_epoch(epoch)

        if args.local_rank == 0:
            logger.info("\nEpoch %s" % (epoch + 1))

        train_loss = train_epoch(args, logger, model, data_loader, valid_data_loader, start, writer,
                                 optimizer, scheduler, epoch)
        valid_loss = eval_epoch(args, logger, model, valid_data_loader,
                                epoch)

        if args.local_rank == 0:
            end = time.time()
            logger.info("Epoch took: {:.3f}s, train loss: {:.6f}, valid loss: {:.6f}".format(
                end - start, train_loss, valid_loss))

            save_model(args, logger, tokenizer,  model, epoch, None)
    writer.close()
    dist_cleanup()


def dist_init(args):
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR')
    master_port = os.getenv('MASTER_PORT')
    init_method += master_ip + ':' + master_port
    dist.init_process_group(
        backend='nccl',
        world_size=args.world_size,
        rank=args.local_rank,
        init_method=init_method)


def dist_cleanup():
    dist.destroy_process_group()


def save_model(args, logger,tokenizer, model, epoch, batch):
    parent = 'models'
    if not os.path.exists(parent):
        os.mkdir(parent)

    path = '{}/{}_epoch_{}'.format(parent, args.log_name, epoch + 1)
    if not os.path.exists(path):
        os.mkdir(path)
    if batch is None:
        model_filename = path + '/pytorch_model.bin'
    else:
        model_filename = path + '/pytorch_model_{}.bin'.format(batch + 1)

    model_obj = model
    while isinstance(model_obj, (DeepSpeedEngine, DDP)):
        model_obj = model_obj.module

    if args.pretrained_path != '':
        config_file = args.pretrained_path + '/config.json'
        vocab_file = args.pretrained_path + '/vocab.txt'
    else:
        config_file = args.model_config
        vocab_file = args.vocab
    if tokenizer!= None:
        tokenizer.save_pretrained(path)
    torch.save(model_obj.state_dict(), model_filename)

    if not os.path.exists(path + '/config.json') \
            or not os.path.samefile(config_file, path + '/config.json'):
        shutil.copy(config_file, path + '/config.json')
    if not os.path.exists(path + '/vocab.txt') \
            or not os.path.samefile(vocab_file, path + '/vocab.txt'):
        shutil.copy(vocab_file, path)

    torch.save(args, path + '/model_training_args.bin')

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        # print(mask)
        # print(logits)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

if __name__ == "__main__":
    args, ds_config = get_configs()

    log_path = 'logs'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger = create_logger('logs/', args.log_name)


    # 把参数配置从配置文件中读取出来写入log日志文件，把ds_配置也写入日志
    def log_configs():
        logger.info(
            json.dumps({k: v for k, v in os.environ.items() if k not in ['LS_COLORS']})
        )
        logger.info(json.dumps(args.__dict__, indent=True))
        model_config_file = args.model_config if args.pretrained_path == '' \
            else args.pretrained_path + '/config.json'
        with open(model_config_file, encoding='utf-8') as f:
            model_config = json.loads(f.read())
        logger.info(json.dumps(model_config, indent=True))
        logger.info(json.dumps(ds_config, indent=True))

    log_configs()

    # 传入args和ds——config构建模型和训练
    train(args, ds_config)

