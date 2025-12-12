from __future__ import absolute_import, division, print_function, unicode_literals

import math
import torch
from torch import nn
from transformers import BertModel, Wav2Vec2Model, WavLMModel, HubertModel
from model.bert_self_encoder import BertSelfEncoder
from model.bert_cross_encoder import BertCrossEncoder
from utils.utils import create_mask


class AudioAblationNet(nn.Module):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """

    def __init__(self, config, args, layer_num1=1, layer_num2=1, layer_num3=1, layer_num4=1, num_labels=4):
        super(AudioAblationNet, self).__init__()
        self.num_labels = num_labels
        self.device = args.device
        # self.bert = BertModel.from_pretrained('bert-base-cased')
        # 将 "facebook/wav2vec2-base" 改成本地路径
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("../pretained_model/wav2vec2-base", output_hidden_states=True,
                                                      return_dict=True, apply_spec_augment=False)

        # 冻结模型的参数
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        # 冻结wav2vec2特征提取器的参数
        # self.wav2vec2.feature_extractor._freeze_parameters()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.vismap2text = nn.Linear(768, config.hidden_size)
        self.vismap2text_v2 = nn.Linear(768, config.hidden_size)

        self.self_audio_attention_encoder = BertSelfEncoder(config)

        self.aug_audio_cross_attention_encoder = BertCrossEncoder(config, layer_num2)

        self.dropout_audio_input = nn.Dropout(0.1)


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


    def forward(self, audio_input, audio_length):

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

        wav2vec2_attention_mask_back = wav2vec2_attention_mask.clone()
        # subsample the frames to 1/4th of the number
        audio_output = audio_output_wav2vec2.clone()
        # audio_output, wav2vec2_attention_mask = self.conv2d_subsample(audio_output_wav2vec2,wav2vec2_attention_mask.unsqueeze(1)) # remove _2

        # project audio_for_test embeddings to a smaller space
        audio_embedding = self.vismap2text(audio_output)


        # calculate added attention mask
        audio_mask = wav2vec2_attention_mask.squeeze(1).clone()
        # calculate extended_img_mask required for cross-attention
        extended_audio_mask = audio_mask.unsqueeze(1).unsqueeze(2)
        extended_audio_mask = extended_audio_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_audio_mask = (1.0 - extended_audio_mask) * -10000.0


        audio_self_attention_encoder = self.self_audio_attention_encoder(audio_embedding, extended_audio_mask)
        audio_self_attention_output = audio_self_attention_encoder[-1]


        aug_audio_encoder = self.aug_audio_cross_attention_encoder(audio_embedding, audio_self_attention_output, extended_audio_mask)
        aug_audio_output = aug_audio_encoder[-1]  # self.batch_size * text_len * hidden_dim


        # pooling
        if self.fuse_type == 'max':
            # audio_for_test
            padding_mask_audio = self.wav2vec2._get_feature_vector_attention_mask(audio_output_wav2vec2.shape[1],audio_attention_mask).to(audio_output_wav2vec2.device)

            aug_audio_output_cloned = aug_audio_output.clone()
            aug_audio_output_cloned[~padding_mask_audio] = -9999.9999  # max
            aug_audio_feats, _ = torch.max(aug_audio_output_cloned, dim=1)  # max


        # concat audio_for_test and text branch
        # final_output = torch.cat((classification_feats_multimodal_txt, classification_feats_multimodal_audio), dim=-1)

        return aug_audio_feats # [batch_size, 768]



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
