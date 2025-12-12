from __future__ import absolute_import, division, print_function, unicode_literals

import math
import torch
from torch import nn
from model.bert_self_encoder import BertSelfEncoder
from model.bert_cross_encoder import BertCrossEncoder


class TextAblationNet(nn.Module):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """

    def __init__(self, config, args, layer_num1=1, layer_num2=1, layer_num3=1, layer_num4=1, num_labels=4):
        super(TextAblationNet, self).__init__()
        self.num_labels = num_labels
        self.device = args.device

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.vismap2text = nn.Linear(768, config.hidden_size)
        self.vismap2text_v2 = nn.Linear(768, config.hidden_size)

        self.self_txt_attention_encoder = BertSelfEncoder(config)

        self.aug_txt_cross_attention_encoder = BertCrossEncoder(config, layer_num1)

        self.dropout_audio_input = nn.Dropout(0.1)


        self.weights = nn.Parameter(torch.zeros(13))

        self.fuse_type = 'max'

        if self.fuse_type == 'att':
            self.output_attention_audio = nn.Sequential(
                nn.Linear(768, 768 // 2),
                ActivateFun("gelu"),
                nn.Linear(768 // 2, 1)
            )
            self.output_attention_multimodal = nn.Sequential(
                nn.Linear(768 * 2, 768 * 2 // 2),
                ActivateFun("gelu"),
                nn.Linear(768 * 2 // 2, 1)
            )
            # self.output_attention_text = nn.Sequential(
            #     nn.Linear(768, 768 // 2),
            #     ActivateFun("gelu"),
            #     nn.Linear(768 // 2, 1)
            # )

    def forward(self, text_embedding, bert_attention_mask):

        text_embedding = self.dropout(text_embedding)

        extended_txt_mask = bert_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        txt_self_attention_encoder = self.self_txt_attention_encoder(text_embedding, extended_txt_mask)
        txt_self_attention_output = txt_self_attention_encoder[-1]

        aug_txt_encoder = self.aug_txt_cross_attention_encoder(text_embedding, txt_self_attention_output, extended_txt_mask)
        aug_txt_output = aug_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim

        # pooling
        if self.fuse_type == 'max':
            # text
            padding_mask_text = bert_attention_mask > 0
            aug_txt_output_cloned = aug_txt_output.clone()
            aug_txt_output_cloned[~padding_mask_text] = -9999.9999  # max
            aug_txt_feats, _ = torch.max(aug_txt_output_cloned, dim=1)  # max

        # concat audio_for_test and text branch
        # final_output = torch.cat((classification_feats_multimodal_txt, classification_feats_multimodal_audio), dim=-1)

        return aug_txt_feats # [batch_size, 768]




class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)
