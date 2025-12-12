from __future__ import absolute_import, division, print_function, unicode_literals

import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, Wav2Vec2Model, WavLMModel, HubertModel
from model.bert_self_encoder import BertSelfEncoder
from model.bert_cross_encoder import BertCrossEncoder
from utils.utils import create_mask


class SCMI_NET(nn.Module):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """

    def __init__(self, config, args, layer_num1=1, layer_num2=1, layer_num3=1, layer_num4=1):
        super(SCMI_NET, self).__init__()
        self.device = args.device
        # self.bert = BertModel.from_pretrained('bert-base-cased')
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("../pretained_model/wav2vec2-base", output_hidden_states=True,
                                                      return_dict=True, apply_spec_augment=False)
        # self.wav2vec2 = WavLMModel.from_pretrained("microsoft/wavlm-base-plus-sv",output_hidden_states=True,return_dict=True,apply_spec_augment=False) # 作者尝试过使用其他的语音预训练模型
        # self.wav2vec2 = HubertModel.from_pretrained("facebook/hubert-base-ls960",output_hidden_states=True,return_dict=True,apply_spec_augment=False)
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

        # self.audio_encoder = TransformerEncoder(config_audio)

        # self.semantic_excite = nn.Linear(768, 768)
        # self.acoustic_excite = nn.Linear(768,768)
        # self.subsample_final = nn.Linear(768 * 4, 768 * 2) #only in case of stats pooling

        self.subsample_final_txt = nn.Linear(768 * 2, 768)
        self.subsample_final_audio = nn.Linear(768 * 2, 768)

        # self.conv2d_subsample = Conv2dSubsampling2(768, 768, 0.1)
        # self.conv2d_subsample_fbank = Conv2dSubsampling(768, 768, 0.1)

        # self.aux_crf = CRF(auxnum_labels, batch_first=True)

        # self.apply(self.init_bert_weights)

        self.weights = nn.Parameter(torch.zeros(13))

        # self.fuse_type = 'max'
        self.fuse_type = 'mean'

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

    def forward(self, text_embedding, text_attention_mask, audio_input, audio_length):

        text_embedding = self.dropout(text_embedding)
        audio_output_wav2vec2 = self.wav2vec2(audio_input)[0]  # imp

        audio_attention_mask = create_mask(audio_input.shape[0], audio_input.shape[1], audio_length)
        input_lengths = self.wav2vec2._get_feat_extract_output_lengths(audio_attention_mask.sum(-1)).type(
            torch.IntTensor)
        wav2vec2_attention_mask = create_mask(audio_output_wav2vec2.shape[0], audio_output_wav2vec2.shape[1],
                                              input_lengths).to(self.device)

        extended_text_mask = text_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_text_mask = extended_text_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0

        # subsample the frames to 1/4th of the number
        audio_output = audio_output_wav2vec2.clone()

        # project audio embeddings to a smaller space
        audio_embedding = self.vismap2text(audio_output)

        # calculate added attention mask
        audio_mask = wav2vec2_attention_mask.squeeze(1).clone()

        # calculate extended_audio_mask required for cross-attention
        extended_audio_mask = audio_mask.unsqueeze(1).unsqueeze(2)
        extended_audio_mask = extended_audio_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_audio_mask = (1.0 - extended_audio_mask) * -10000.0

        # intra-modal interaction module
        text_self_attention_encoder = self.self_txt_attention_encoder(text_embedding, extended_text_mask)
        text_self_attention_output = text_self_attention_encoder[-1]

        audio_self_attention_encoder = self.self_audio_attention_encoder(audio_embedding, extended_audio_mask)
        audio_self_attention_output = audio_self_attention_encoder[-1]

        aug_txt_encoder = self.aug_txt_cross_attention_encoder(text_embedding, text_self_attention_output,
                                                               extended_text_mask)
        aug_txt_output = aug_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim

        aug_audio_encoder = self.aug_audio_cross_attention_encoder(audio_embedding, audio_self_attention_output,
                                                                   extended_audio_mask)
        aug_audio_output = aug_audio_encoder[-1]  # self.batch_size * text_len * hidden_dim

        # modal fusion module
        cross_txt_encoder = self.txt2audio_cross_attention_encoder(aug_txt_output, aug_audio_output,
                                                                   extended_audio_mask)
        cross_txt_output = cross_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim

        cross_audio_encoder = self.audio2txt_cross_attention_encoder(aug_audio_output, aug_txt_output,
                                                                     extended_text_mask)
        cross_audio_output = cross_audio_encoder[-1]  # self.batch_size * audio_length * hidden_dim

        # apply visual gate and get final representations
        # merge_text_representation = torch.cat((cross_txt_output, aug_txt_output), dim=-1)
        # gate_value = torch.sigmoid(self.gate_txt_linear(merge_text_representation))  # batch_size, text_len, hidden_dim
        # gated_text_output = torch.mul(gate_value, cross_txt_output)
        # final_text_output = torch.cat((cross_txt_output, gated_text_output), dim=-1)
        #
        # merge_audio_representation = torch.cat((cross_audio_output, aug_audio_output), dim=-1)
        # gate_value = torch.sigmoid(
        #     self.gate_audio_linear(merge_audio_representation))  # batch_size, text_len, hidden_dim
        # gated_audio_output = torch.mul(gate_value, cross_audio_output)
        # final_audio_output = torch.cat((cross_audio_output, gated_audio_output), dim=-1)



        # 消融实验：验证GFM模块的性能

        padding_mask_text = text_attention_mask > 0
        cross_txt_output[~padding_mask_text] = -9999.9999  # max
        classification_feats_multimodal_txt, _ = torch.max(cross_txt_output, dim=1)  # max

        aug_txt_output_cloned = aug_txt_output.clone()
        aug_txt_output_cloned[~padding_mask_text] = -9999.9999  # max
        aug_txt_feats, _ = torch.max(aug_txt_output_cloned, dim=1)  # max

        # audio
        padding_mask_audio = self.wav2vec2._get_feature_vector_attention_mask(audio_output_wav2vec2.shape[1],
                                                                              audio_attention_mask).to(
            audio_output_wav2vec2.device)
        cross_audio_output[~padding_mask_audio] = -9999.9999  # max
        classification_feats_multimodal_audio, _ = torch.max(cross_audio_output, dim=1)  # max

        aug_audio_output_cloned = aug_audio_output.clone()
        aug_audio_output_cloned[~padding_mask_audio] = -9999.9999  # max
        aug_audio_feats, _ = torch.max(aug_audio_output_cloned, dim=1)  # max

        final_output = torch.cat((aug_txt_feats, aug_audio_feats), dim=-1)

        return aug_txt_feats, aug_audio_feats, final_output  # [batch_size, 1536]



        # ablation expriment：GFM

        # multimodal_output = final_text_output.clone()

        # pooling
        # if self.fuse_type == 'mean':
        #     # text
        #     padding_mask_text = text_attention_mask > 0
        #     final_text_output[~padding_mask_text] = 0  # mean
        #     classification_feats_multimodal_txt = torch.mean(final_text_output, dim=1)  # mean
        #
        #     aug_txt_output_cloned = aug_txt_output.clone()
        #     aug_txt_output_cloned[~padding_mask_text] = 0  # mean
        #     aug_txt_feats = torch.mean(aug_txt_output_cloned, dim=1)  # mean
        #
        #     # audio
        #     padding_mask_audio = self.wav2vec2._get_feature_vector_attention_mask(audio_output_wav2vec2.shape[1],
        #                                                                           audio_attention_mask).to(
        #         audio_output_wav2vec2.device)
        #     final_audio_output[~padding_mask_audio] = 0  # mean
        #     classification_feats_multimodal_audio = torch.mean(final_audio_output, dim=1)  # mean
        #
        #     aug_audio_output_cloned = aug_audio_output.clone()
        #     aug_audio_output_cloned[~padding_mask_audio] = 0  # mean
        #     aug_audio_feats = torch.mean(aug_audio_output_cloned, dim=1)  # mean
        #
        # elif self.fuse_type == 'max':
        #     # text
        #     padding_mask_text = text_attention_mask > 0
        #     final_text_output[~padding_mask_text] = -9999.9999  # max
        #     classification_feats_multimodal_txt, _ = torch.max(final_text_output, dim=1)  # max
        #
        #     aug_txt_output_cloned = aug_txt_output.clone()
        #     aug_txt_output_cloned[~padding_mask_text] = -9999.9999  # max
        #     aug_txt_feats, _ = torch.max(aug_txt_output_cloned, dim=1)  # max
        #
        #     # audio
        #     padding_mask_audio = self.wav2vec2._get_feature_vector_attention_mask(audio_output_wav2vec2.shape[1],
        #                                                                           audio_attention_mask).to(
        #         audio_output_wav2vec2.device)
        #     final_audio_output[~padding_mask_audio] = -9999.9999  # max
        #     classification_feats_multimodal_audio, _ = torch.max(final_audio_output, dim=1)  # max
        #
        #     aug_audio_output_cloned = aug_audio_output.clone()
        #     aug_audio_output_cloned[~padding_mask_audio] = -9999.9999  # max
        #     aug_audio_feats, _ = torch.max(aug_audio_output_cloned, dim=1)  # max

        # elif self.fuse_type == 'att':
        #     multimodal_mask = text_attention_mask.permute(1, 0).contiguous()
        #     multimodal_mask = multimodal_mask[0:multimodal_output.size(1)]
        #     multimodal_mask = multimodal_mask.permute(1, 0).contiguous()
        #
        #     multimodal_alpha = self.output_attention_multimodal(multimodal_output)
        #     multimodal_alpha = multimodal_alpha.squeeze(-1).masked_fill(multimodal_mask == 0, -1e9)
        #     multimodal_alpha = torch.softmax(multimodal_alpha, dim=-1)
        #     classification_feats_multimodal_1 = (multimodal_alpha.unsqueeze(-1) * multimodal_output).sum(dim=1)
        #
        # elif self.fuse_type == 'stats':
        #     classification_feats_multimodal_1 = torch.cat(
        #         (torch.mean(multimodal_output, dim=1), torch.std(multimodal_output, dim=1)), dim=-1)

        # sub-sampling
        # classification_feats_multimodal_txt = self.subsample_final_txt(classification_feats_multimodal_txt)
        # classification_feats_multimodal_audio = self.subsample_final_audio(classification_feats_multimodal_audio)

        # concat audio and text
        # final_output = torch.cat((classification_feats_multimodal_txt, classification_feats_multimodal_audio), dim=-1)
        #
        # return aug_txt_feats, aug_audio_feats, final_output  # [batch_size, 1536]

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
