from __future__ import absolute_import, division, print_function, unicode_literals

import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, Wav2Vec2Model, WavLMModel, HubertModel
from model.bert_self_encoder import BertSelfEncoder
from model.bert_cross_encoder import BertCrossEncoder
from utils.utils import create_mask

"""
Ablation Study: after the intra-modal interacting, concat the audio feature and text feature to verify SCMI-Net's modality fusion module
"""
class ConcatAblationNet(nn.Module):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """

    def __init__(self, config, args, layer_num1=1, layer_num2=1, layer_num3=1, layer_num4=1, num_labels=4):
        super(ConcatAblationNet, self).__init__()
        self.num_labels = num_labels
        self.device = args.device
        # self.bert = BertModel.from_pretrained('bert-base-cased')
        # 将 "facebook/wav2vec2-base" 改成本地路径
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("../pretained_model/wav2vec2-base", output_hidden_states=True,
                                                      return_dict=True, apply_spec_augment=False)
        # 冻结wav2vec2参数
        self.wav2vec2.feature_extractor._freeze_parameters()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.vismap2text = nn.Linear(768, config.hidden_size)
        self.vismap2text_v2 = nn.Linear(768, config.hidden_size)

        self.self_txt_attention_encoder = BertSelfEncoder(config)
        self.self_audio_attention_encoder = BertSelfEncoder(config)

        self.aug_txt_cross_attention_encoder = BertCrossEncoder(config, layer_num1)
        self.aug_audio_cross_attention_encoder = BertCrossEncoder(config, layer_num2)
        self.txt2audio_cross_attention_encoder = BertCrossEncoder(config, layer_num3)
        self.audio2txt_cross_attention_encoder = BertCrossEncoder(config, layer_num4)

        self.gate_txt_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.gate_audio_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

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


    def forward(self, text_embedding, bert_attention_mask, audio_input, audio_length):

        text_embedding = self.dropout(text_embedding)
        # audio_output_wav2vec2_all = self.wav2vec2(audio_input) #only in average
        # audio_output_wav2vec2 = audio_output_wav2vec2_all[0] #only in average
        audio_output_wav2vec2 = self.wav2vec2(audio_input)[0]  # imp

        # print(audio_output_wav2vec2_layers.shape)
        # audio_output_wav2vec2_2 = self._weighted_sum([f for f in audio_output_wav2vec2_all["hidden_states"]], True) #weighted mean
        # audio_output_wav2vec2_2 = torch.mean(torch.stack(audio_output_wav2vec2_all["hidden_states"][-8:], axis = 0), axis = 0) #8 average

        # -----------------------------------------------------------------------------------------------------------#
        # create raw audio_for_test, FBank and wav2vec2 hidden state attention masks
        # create raw audio_for_test, FBank and wav2vec2 hidden state attention masks
        audio_attention_mask, fbank_attention_mask, wav2vec2_attention_mask, input_lengths = None, None, None, None

        audio_attention_mask = create_mask(audio_input.shape[0], audio_input.shape[1], audio_length)
        input_lengths = self.wav2vec2._get_feat_extract_output_lengths(audio_attention_mask.sum(-1)).type(torch.IntTensor)
        wav2vec2_attention_mask = create_mask(audio_output_wav2vec2.shape[0], audio_output_wav2vec2.shape[1], input_lengths)
        wav2vec2_attention_mask = wav2vec2_attention_mask.to(self.device)

        # -----------------------------------------------------------------------------------------------------------#

        # audio_output_dropout = self.dropout_audio_input(audio_output_wav2vec2)
        # logits_ctc = self.ctc_linear(audio_output_dropout)

        extended_txt_mask = bert_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        wav2vec2_attention_mask_back = wav2vec2_attention_mask.clone()
        # subsample the frames to 1/4th of the number
        audio_output = audio_output_wav2vec2.clone()
        # audio_output, wav2vec2_attention_mask = self.conv2d_subsample(audio_output_wav2vec2,wav2vec2_attention_mask.unsqueeze(1)) # remove _2

        # project audio_for_test embeddings to a smaller space
        audio_embedding = self.vismap2text(audio_output)

        # --------------------applying txt2audio attention mechanism to obtain audio_for_test-based text representations----------------------------#
        # calculate added attention mask
        # img_mask = added_attention_mask = torch.ones([audio_output.shape[0],audio_output.shape[1]]).cuda()

        # calculate added attention mask
        audio_mask = wav2vec2_attention_mask.squeeze(1).clone()
        # calculate extended_img_mask required for cross-attention
        extended_audio_mask = audio_mask.unsqueeze(1).unsqueeze(2)
        extended_audio_mask = extended_audio_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_audio_mask = (1.0 - extended_audio_mask) * -10000.0

        #
        txt_self_attention_encoder = self.self_txt_attention_encoder(text_embedding, extended_txt_mask)
        txt_self_attention_output = txt_self_attention_encoder[-1]

        audio_self_attention_encoder = self.self_audio_attention_encoder(audio_embedding, extended_audio_mask)
        audio_self_attention_output = audio_self_attention_encoder[-1]

        aug_txt_encoder = self.aug_txt_cross_attention_encoder(text_embedding, txt_self_attention_output, extended_txt_mask)
        aug_txt_output = aug_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim

        aug_audio_encoder = self.aug_txt_cross_attention_encoder(audio_embedding, audio_self_attention_output, extended_audio_mask)
        aug_audio_output = aug_audio_encoder[-1]  # self.batch_size * text_len * hidden_dim


        # pooling
        if self.fuse_type == 'max':
            # text
            padding_mask_text = bert_attention_mask > 0

            aug_txt_output_cloned = aug_txt_output.clone()
            aug_txt_output_cloned[~padding_mask_text] = -9999.9999  # max
            aug_txt_feats, _ = torch.max(aug_txt_output_cloned, dim=1)  # max

            # audio_for_test
            padding_mask_audio = self.wav2vec2._get_feature_vector_attention_mask(audio_output_wav2vec2.shape[1],audio_attention_mask).to(audio_output_wav2vec2.device)

            aug_audio_output_cloned = aug_audio_output.clone()
            aug_audio_output_cloned[~padding_mask_audio] = -9999.9999  # max
            aug_audio_feats, _ = torch.max(aug_audio_output_cloned, dim=1)  # max

        # concat audio_for_test and text branch
        final_output = torch.cat((aug_txt_feats, aug_audio_feats), dim=-1)

        return final_output # [batch_size, 1536]

    def _ctc_loss(self, logits, labels, input_lengths, attention_mask=None):

        loss = None
        if labels is not None:

            # # retrieve loss input_lengths from attention_mask
            # attention_mask = (
            #     attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            # )
            if attention_mask is not None:
                input_lengths = self.wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(-1)).type(
                    torch.IntTensor)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction="sum",
                    zero_infinity=False,
                )

        return loss

    def _cls_loss(self, logits,
                  cls_labels):  # sum hidden_states over dim 1 (the sequence length), then feed into self.cls
        loss = None
        if cls_labels is not None:
            loss = F.cross_entropy(logits, cls_labels.to(logits.device))
        return loss

    def _weighted_sum(self, feature, normalize):

        stacked_feature = torch.stack(feature, dim=0)

        if normalize:
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],))

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(13, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature


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
