# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch OpenAI GPT-2 model.
Adapted from https://github.com/huggingface/transformers/blob/v2.11.0/src/transformers/modeling_gpt2.py
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
# Difference from Transformers 2.11.0 is code in _USE_GROVER
_USE_GROVER = True

if _USE_GROVER:

    from transformers.activations import ACT2FN
    from transformers.configuration_gpt2 import GPT2Config
    from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
    from transformers.modeling_utils import Conv1D, PreTrainedModel, SequenceSummary, prune_conv1d_layer
    from transformers import CONFIG_NAME, WEIGHTS_NAME, GPT2Config, GPT2Model

    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from transformers.file_utils import ModelOutput
else:
    from .activations import ACT2FN
    from .configuration_gpt2 import GPT2Config
    from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
    from .modeling_utils import Conv1D, PreTrainedModel, SequenceSummary, prune_conv1d_layer

    from .modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from .file_utils import ModelOutput

logger = logging.getLogger(__name__)

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]

if _USE_GROVER:
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    _GPT2_ML_TF_TO_TORCH = {
        'LayerNorm_embed_norm': 'emb_norm',
        'pos_embed': 'wpe.weight',
        'word_embed': 'wte.weight',

        'layer': 'h',
        # Most importently This two layer norm must be put on the same position as gpt2-ml
        # or generated data is bad, just repeat the last token
        'LayerNorm_mlp_ln0': 'ln_1',
        'LayerNorm_mlp_ln1': 'ln_2',
        'intermediate': 'mlp.c_fc',
        'output': 'mlp.c_proj',
        'query_layer': 'attn.c_attn',
        'key_layer': 'attn.c_attn',
        'value_layer': 'attn.c_attn',
        'context_projection_layer': 'attn.c_proj',

        'gamma': 'weight',
        'kernel': 'weight',
        'beta': 'bias',
        'bias': 'bias',
    }


    def convert_gpt2_checkpoint_to_pytorch(gpt2_checkpoint_path, gpt2_config_file, pytorch_dump_folder_path):
        # Construct model
        if gpt2_config_file == "":
            config = GPT2Config()
        else:
            config = GPT2Config.from_json_file(gpt2_config_file)
        model = GPT2Model(config)

        # Load weights from numpy
        load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path)

        # Save pytorch-model
        pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME
        pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME
        print("Save PyTorch model to {}".format(pytorch_weights_dump_path))
        torch.save(model.state_dict(), pytorch_weights_dump_path)
        print("Save configuration file to {}".format(pytorch_config_dump_path))
        with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
            f.write(config.to_json_string())


    # XXX: MUST do like: convert_gpt2_checkpoint_to_pytorch('./model.ckpt-100000', './mega.json', './')
    #      https://github.com/tensorflow/models/issues/2675#issuecomment-516595597
    def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
        """ Load tf checkpoints in a pytorch model
        """
        try:
            import re
            import tensorflow as tf
        except ImportError:
            logger.error(
                "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                "https://www.tensorflow.org/install/ for installation instructions."
            )
            raise
        tf_path = os.path.abspath(gpt2_checkpoint_path)
        logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
        # Load weights from TF model
        init_vars = tf.train.list_variables(tf_path)
        names = []
        arrays = []
        for name, shape in init_vars:
            logger.info("Loading TF weight {} with shape {}".format(name, shape))
            array = tf.train.load_variable(tf_path, name)
            names.append(name)
            arrays.append(array.squeeze())

        import copy
        orig_model = copy.deepcopy(model)

        for name, array in zip(names, arrays):
            name = name[6:]  # skip "model/"
            name = name.split("/")
            pointer = model

            attn_layer = ''
            for m_name in name:
                if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                    scope_names = re.split(r"(\d+)", m_name)
                else:
                    scope_names = [m_name]
                sname = scope_names[0]

                if sname == '' or sname == 'embeddings':
                    continue
                elif sname not in _GPT2_ML_TF_TO_TORCH:
                    print('=========================================================')
                    logger.info('Skip var name {}'.format(scope_names))
                    pointer = None
                    break
                else:
                    tname = _GPT2_ML_TF_TO_TORCH[sname]
                    if '.' in tname:
                        parent, child = tname.split('.')
                        pointer = getattr(pointer, parent)
                        pointer = getattr(pointer, child)
                    else:
                        pointer = getattr(pointer, tname)

                    if tname == 'attn.c_attn':
                        attn_layer = sname

                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]

            if pointer is None:
                continue
            if attn_layer == '':
                try:
                    assert pointer.shape == array.shape
                except AssertionError as e:
                    e.args += (pointer.shape, array.shape)
                    raise
            logger.info("Initialize PyTorch weight {}, {}, {}".format(name, array.mean(), pointer.mean()))
            if attn_layer == '':
                pointer.data = torch.from_numpy(array)
            else:
                shape = pointer.shape
                d = torch.from_numpy(array)
                is_bias = len(shape) == 1
                end = int(shape[0 if is_bias else 1] / 3)
                m = dict(
                    query_layer=0,
                    key_layer=end,
                    value_layer=end * 2,
                )
                start = m[attn_layer]
                end = start + end
                if is_bias:
                    pointer.data[start:end] = d
                else:
                    pointer.data[:, start:end] = d

        for name, params in orig_model.named_parameters():
            for n, p in model.named_parameters():
                if name == n:
                    if params.equal(p):
                        print('--------------------------')
                        print(' %s not changed!' % n)
        return model
else:
    def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
        """ Load tf checkpoints in a pytorch model
        """
        try:
            import re
            import tensorflow as tf
        except ImportError:
            logger.error(
                "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                "https://www.tensorflow.org/install/ for installation instructions."
            )
            raise
        tf_path = os.path.abspath(gpt2_checkpoint_path)
        logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
        # Load weights from TF model
        init_vars = tf.train.list_variables(tf_path)
        names = []
        arrays = []
        for name, shape in init_vars:
            logger.info("Loading TF weight {} with shape {}".format(name, shape))
            array = tf.train.load_variable(tf_path, name)
            names.append(name)
            arrays.append(array.squeeze())

        for name, array in zip(names, arrays):
            name = name[6:]  # skip "model/"
            name = name.split("/")
            pointer = model
            for m_name in name:
                if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                    scope_names = re.split(r"(\d+)", m_name)
                else:
                    scope_names = [m_name]
                if scope_names[0] == "w" or scope_names[0] == "g":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "b":
                    pointer = getattr(pointer, "bias")
                elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                    pointer = getattr(pointer, scope_names[0])
                    pointer = getattr(pointer, "weight")
                else:
                    pointer = getattr(pointer, scope_names[0])
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
        return model


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, gpt_mask=True):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1))**0.5)
        if gpt_mask==True:
            nd, ns = w.size(-2), w.size(-1)
            mask = self.bias[:, :, ns - nd: ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
        else:
            pass
        if attention_mask is not None:
            # Apply the attention mask
            # print(w.shape)
            # print(attention_mask.shape)
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask
        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, query, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, gpt_mask=True):
        x = self.c_attn(x)
        #query, key, value = x.split(self.split_size, dim=2)
        _, key, value = x.split(self.split_size, dim=2)
        y = self.c_attn(query)
        query, _, _=y.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        # print(query.shape)
        # print(key.shape)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)
        # print(query.shape)
        # print(key.shape)
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, gpt_mask)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)
class Attention1(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, gpt_mask=True):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1))**0.5)
        if gpt_mask==True:
            nd, ns = w.size(-2), w.size(-1)
            mask = self.bias[:, :, ns - nd: ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
        else:
            pass
        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask
        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, query, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, gpt_mask=True):
        x = self.c_attn(x)
        _, key, value = x.split(self.split_size, dim=2)
        x = self.c_attn(query)
        query, _, _ = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        # print(query.shape)
        # print(key.shape)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)
        # print(query.shape)
        # print(key.shape)
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, gpt_mask)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)
class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
class MLP1(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, 2*nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, query, encoder_embeds=None,layer_past=None, attention_mask=None, head_mask=None, use_cache=False, gpt_mask=True):
        if encoder_embeds == None:
            pass
        else:
            x=x+encoder_embeds
        output_attn = self.attn(
            x if _USE_GROVER else self.ln_1(x),
            layer_past=layer_past,
            query=query if _USE_GROVER else self.ln_1(query),
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            gpt_mask=gpt_mask,
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)
        x = x + a
        m = self.mlp(self.ln_1(x) if _USE_GROVER else self.ln_2(x))

        x = x + m
        if _USE_GROVER:
            x = self.ln_2(x)

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)
class Block1(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd
        #print(nx)
        self.attn1 = Attention(nx, n_ctx, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn21 = Attention1(nx, n_ctx, config, scale)

        self.mlp21 = MLP(nx, config)

        self.ln_21 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

        #???????????????????????????????????????attention
        self.mlp3 = MLP1(nx, config)
        self.ln_3 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)




    def forward(self, x, key_value1, encoder_embeds=None,layer_past=None, attention_mask=None, head_mask=None, use_cache=False, gpt_mask=True):

        #input layer
        output_attn1 = self.attn1(
            x,
            layer_past=layer_past,
            query=x,
            attention_mask=None,
            head_mask=head_mask,
            use_cache=use_cache,
            gpt_mask=True,
        )
        a1=output_attn1[0]
        x = x + a1
        x=self.ln_1(x)

        # layer topic
        output_attn21 = self.attn21(
            key_value1,
            layer_past=None,
            query=x,
            attention_mask=attention_mask,
            head_mask=None,
            use_cache=None,
            gpt_mask=False,
        )
        output_mlp21 = self.mlp21(output_attn21[0])
        output_ln21 = self.ln_21(output_mlp21+output_attn21[0])

        # layer concat and foward
        x = torch.cat((x, output_ln21), -1)
        x = self.mlp3(x)
        x = self.ln_3(x)

        return [x] # x, present, (attentions)

class GPT2PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`mc_labels` is provided):
            Multiple choice classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            ``past_key_values`` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


GPT2_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length`` if ``past`` is ``None`` else ``past[0].shape[-2]`` (``sequence_length`` of input past key value states).
            Indices of input sequence tokens in the vocabulary.

            If `past` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.

            Indices can be obtained using :class:`transformers.GPT2Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__

        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding.
            The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`, defaults to :obj:`None`):
            `input_ids_length` = `sequence_length if `past` is None else 1
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
            If `past` is used, optionally only the last `inputs_embeds` have to be input (see `past`).
        use_cache (:obj:`bool`):
            If `use_cache` is True, `past` key value states are returned and can be used to speed up decoding (see `past`). Defaults to `True`.
"""

@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        if _USE_GROVER:
            self.emb_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        if not _USE_GROVER:
            self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            encoder_embeds=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            use_cache=True,
            return_dict=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
            If `past` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import GPT2Tokenizer, GPT2Model
        import torch

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        #print(input_ids)
        #print(inputs_embeds)
        #print(self.wte(torch.tensor([[50256]]).cuda()))
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)
        if _USE_GROVER:
            hidden_states = self.emb_norm(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)
        #print(input_ids)
        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            #torch.Size([2, 3, 24, 298, 64])
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)
            #print(hidden_states.shape)
            outputs = block(
                hidden_states,
                query=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
            )
            hidden_states, present = outputs[:2]
            #print(hidden_states.shape)
            #print(present.shape)
            if use_cache is True:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        if not _USE_GROVER:
            hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)


        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # outputs = (hidden_states,)
        # if use_cache is True:
        #    outputs = outputs + (presents,)
        # if self.output_hidden_states:
        #    outputs = outputs + (all_hidden_states,)
        # if self.output_attentions:
        #    # let the number of heads free (-1) so we can extract attention even after head pruning
        #    attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
        #    all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
        #    outputs = outputs + (all_attentions,)
        # return outputs  # last hidden state, (presents), (all hidden_states), (attentions)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class Encoder_gpt(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.emb_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.drop = nn.Dropout(config.embd_pdrop)
        #self.conv=nn.Conv1d(in_channels = config.n_embd, out_channels = config.n_embd, kernel_size = 2, bias=True, stride=1, padding=1)
        #self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(2)])
        self.init_weights()

    def gettp(self,wte,wpe):
        self.wte=wte
        self.wpe=wpe
    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)
    def get_embeds(self,input_ids,local_rank):
        # ??????topic????????????????????????tokenembedding,??????544,??????545
        id_embeds = self.wte(input_ids)
        num = 1
        id_list = []
        now = 0
        for id in input_ids[0]:
            if id.item() == 544:
                num += 1
                id_list.append(now)
            now = now + 1
        inputs_embeds = torch.zeros([1, num, 1536], dtype=torch.float16).cuda(local_rank, non_blocking=True)
        point = 0
        length = 0
        for now in range(input_ids.size()[-1]):
            if now not in id_list:
                length += 1
                inputs_embeds[0][point] += id_embeds[0][now]
            else:
                # inputs_embeds[0][point]=inputs_embeds[0][point]/length
                length = 0
                point += 1
        return inputs_embeds,num
    def get_mask_and_position(self,ids,local_rank):
        # ??????topic????????????????????????tokenembedding,??????544,??????545
        # ???????????????????????????token
        key_position=[]
        length=ids.shape[-1]
        mask=torch.zeros([length,length]).cuda(local_rank, non_blocking=True)
        position=torch.arange(0, length, dtype=torch.long).cuda(local_rank, non_blocking=True)
        cls_position=0
        sep_position=1
        relative_position=2
        # to determine whether come to the last token
        flag=True

        for i in range(length):
            if ids[0][i].item() == 50257:
                key_position.append(i)
                position[i]=cls_position
                # [CLS] can view [CLS]
                mask[i, i] = 1.0
            elif ids[0][i].item() == 50259:
                key_position.append(i)
                position[i] = sep_position
                relative_position = sep_position + 1
                #[CLS] can view [SEP]
                mask[0, i] = 1.0
                mask[i, 0] = 1.0
                #[SEP] can view surround
                head=i
                # find next [SEP]
                for tail in range(i+1,length):
                    if ids[0][tail].item() == 50259:
                        flag=False
                        break
                if flag==False:
                    mask[head:tail,head:tail]=1.0
                else:
                    mask[head:tail+1, head:tail+1] = 1.0
                flag=True
            else:
                position[i]=relative_position
                relative_position+=1
        key_position = torch.tensor(key_position).cuda(local_rank, non_blocking=True)
        return mask,position,key_position
    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids,
            memory=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            use_cache=True,
            return_dict=None,
            local_rank='cuda:0'
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
            If `past` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import GPT2Tokenizer, GPT2Model
        import torch

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        #attention_mask,position_ids,key_position = self.get_mask_and_position(input_ids, local_rank)
        #print(attention_mask)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            # attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1)
            # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + token_type_embeds+ position_embeds
        hidden_states = self.drop(hidden_states)
        hidden_states = self.emb_norm(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                query=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                gpt_mask=False,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        # print('hidden_states')  # 1.3
        # print(hidden_states[:, 0, :])
        # print(torch.norm(hidden_states[:, 0, :], p=2))  # 13
        # embed_perentity=[hidden_states.index_select(-2, id_entity).permute(0,2,1) for id_entity in id_groups]
        # print('embed_perentity')
        # print(embed_perentity)
        # convs=[self.conv(embeds) for embeds in embed_perentity]
        # print('convs')  # 1.3
        # print(convs[0])
        # print(torch.norm(convs[0], p=2))  # 13
        # print('convs')
        # print(convs)
        # pooled = [F.avg_pool1d(cv, cv.shape[2]).squeeze(2) for cv in convs]
        # print('pooled')  # 1.3
        # print(pooled[0])
        # print(torch.norm(pooled[0], p=2))  # 13
        # print('pooled')
        # print(pooled)
        # pooled=[self.ln_f(x) for x in pooled]
        # print('pooled2')  # 1.3
        # print(pooled[0])
        # print(torch.norm(pooled[0], p=2))  # 13
        # cat=torch.cat((pooled), dim=0)
        # print('cat')
        # print(cat)

        # hidden_states = cat

        return  hidden_states
@add_start_docstrings(
    """The GPT2 Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.transformer2 = nn.ModuleList([Block1(config.n_ctx, config, scale=True) for _ in range(4)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.encoder=Encoder_gpt(config)
        self.encoder.gettp(self.transformer.wte, self.transformer.wpe)
        self.init_weights()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        #pass
        self.encoder.set_input_embeddings(new_embeddings)
        #self.knowledge.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids,past, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        past = None
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {"input_ids": input_ids, "zhutis": kwargs['zhutis'], "e_attention_mask": kwargs['e_attention_mask'],
                "e_position_ids": kwargs['e_position_ids'], "past": past, "use_cache": kwargs["use_cache"], }

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids,
            zhutis,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_cache=True,
            return_dict=None,
            e_attention_mask=None,
            e_token_type_ids=None,
            e_position_ids=None,
            e_head_mask=None,
            e_inputs_embeds=None,
            e_use_cache=True,
            e_return_dict=None,
            f_attention_mask=None,
            memory=None,
            memory_mask=None,
            local_rank='cuda:0',
            topic_content=None,
            phase=True,
            indexs=None,
            eos_position=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            encoder_embeds=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        #get batch_encoder_outputs and concat to cat from batch_zhuti_ids
        encoder_outouts = self.encoder(
            zhutis,
            past=None,
            attention_mask=e_attention_mask,
            token_type_ids=e_token_type_ids,
            position_ids=e_position_ids,
            head_mask=e_head_mask,
            inputs_embeds=e_inputs_embeds,
            use_cache=e_use_cache,
            local_rank=local_rank
        )
        if f_attention_mask is not None:
            f_attention_mask = f_attention_mask.unsqueeze(1)
            f_attention_mask = f_attention_mask.to(dtype=next(self.parameters()).dtype)
            f_attention_mask = (1.0 - f_attention_mask) * -10000.0

        for i, r in enumerate(self.transformer2):
            outputs = r(
                hidden_states,
                key_value1=encoder_outouts,
                attention_mask=f_attention_mask,
                encoder_embeds=None,
                use_cache=use_cache,
            )
            hidden_states = outputs[0]
        self.last_output = hidden_states

        lm_logits = self.lm_head(hidden_states)

        loss = None
        outputs = (lm_logits,) + transformer_outputs[1:]
        if not return_dict:
            if labels is not None and indexs is not None:
                dic = {}
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                dic['ce_loss'] = loss

                eos_output = []
                eos_position = eos_position.squeeze(1)
                for i in range(eos_position.shape[0]):
                    eos_output.append(self.last_output[i, eos_position[i], :].unsqueeze(0))
                dic['pooler_output'] = torch.cat(eos_output, 0)
                dic['last_hidden_state'] = self.last_output
                return dic
            elif labels is not None:
                # Shift so that tokens < n predict n
                # length = labels.shape[-1]
                # start = 0
                # for i in range(length):
                #     if labels[0][i].item() == 796:
                #         start = i + 1
                #         break
                #
                # # if now is phase1, we predict [0,start-1), else we predict [start-1, -1)
                # if phase:
                #     shift_logits = lm_logits[..., : start - 1, :].contiguous()
                #     shift_labels = labels[..., 1: start].contiguous()
                # else:
                #     shift_logits = lm_logits[..., start - 1:-1, :].contiguous()
                #     shift_labels = labels[..., start:].contiguous()
                # Flatten the tokens
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                outputs = (loss,) + outputs
                return outputs
            elif indexs is not None:
                dic = {}
                # # print(eos_position.squeeze(1))
                eos_output = []
                eos_position = eos_position.squeeze(1)
                for i in range(eos_position.shape[0]):
                    eos_output.append(self.last_output[i, eos_position[i], :].unsqueeze(0))
                dic['pooler_output'] = torch.cat(eos_output, 0)
                # print(dic['pooler_output'].shape)
                # print(self.last_output[eos_position])
                # print(self.last_output[eos_position].shape)
                dic['last_hidden_state'] = self.last_output
                return dic
            else:
                return outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """The GPT2 Model transformer with a language modeling and a multiple-choice classification
    head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.
    The language modeling head has its weights tied to the input embeddings,
    the classification head takes as input the input of a specified classification token index in the input sequence).
""",
    GPT2_START_DOCSTRING,
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            lm_labels=None,
            mc_labels=None,
            use_cache=True,
            return_dict=None,
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`)
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`, defaults to :obj:`None`)
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``lm_labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`multiple_choice_labels` is provided):
            Multiple choice classification loss.
        lm_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        mc_loss = None
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        lm_loss = None
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)

        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

# from transformers import GPT2Tokenizer, pipeline, set_seed
# set_seed(42)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
# res = generator("stand look field", max_length=30, num_return_sequences=5)
# for i, v in enumerate(res):
#     print(v['generated_text'])