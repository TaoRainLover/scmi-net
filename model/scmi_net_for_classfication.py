from torch import nn
from collections import OrderedDict
from model.scmi_net import SCMI_NET as SCMI
from model.ablation_for_audio import AudioAblationNet
from model.ablation_for_text import TextAblationNet
from model.ablation_mmi_module_for_concat import ConcatAblationNet
from transformers import BertConfig
import torch.nn.functional as F
from utils.adversarial import LabelSmoothSoftmaxCEV1
from model.supConLoss import SupConLoss


class Multimodal_MMI(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.config_mmi = BertConfig('config.json')

        self.ablation = args.ablation
        self.ablation_type = args.ablation_type

        self.ablation_net = None
        self.scmi = None

        if args.ablation:
            if args.ablation_type == 'audio':
                self.ablation_net = AudioAblationNet(self.config_mmi, args)
            elif args.ablation_type == 'text':
                self.ablation_net = TextAblationNet(self.config_mmi, args)
            elif args.ablation_type == 'concat':
                self.ablation_net = ConcatAblationNet(self.config_mmi, args)

        self.scmi = SCMI(self.config_mmi, args)

        self.num_labels = args.num_labels

        self.multimodal_classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(config.hidden_size * 2, config.hidden_size)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(config.hidden_size, config.hidden_size)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(config.hidden_size, args.num_labels))
        ]))

        self.txt_classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(config.hidden_size, config.hidden_size)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(config.hidden_size, args.num_labels))
        ]))

        self.audio_classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(config.hidden_size, config.hidden_size)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(config.hidden_size, args.num_labels))
        ]))

        self.supcon_loss = SupConLoss()
        self.supcon_loss_weight = args.supcon_loss_weight

        self.loss_type = args.loss_type
        self.criterion = LabelSmoothSoftmaxCEV1(lb_smooth=0.1)

        self.text_modal_weight = args.text_modal_loss_weight
        self.audio_modal_weight = args.audio_modal_loss_weight

    def forward(self, text_output, attention_mask, audio_inputs, audio_length, emotion_labels, mode="train"):
        if self.ablation:
            # only audio
            if self.ablation_type == 'audio':
                audio_feats = self.ablation_net(audio_inputs, audio_length)
                audio_feats_pooled = self.audio_classifier(audio_feats)
                loss = None
                if self.loss_type == 'cross_entropy':
                    loss = F.cross_entropy(audio_feats_pooled, emotion_labels)
                elif self.loss_type == 'LabelSmoothSoftmaxCEV1':
                    loss = self.criterion(audio_feats_pooled, emotion_labels)
                if mode == "train":
                    return audio_feats_pooled, loss, 0, loss, 0, 0
                else:
                    return audio_feats_pooled, audio_feats, None

            # only text
            elif self.ablation_type == 'text':
                txt_feats = self.ablation_net(text_output, attention_mask)
                txt_feats_pooled = self.txt_classifier(txt_feats)
                loss = None
                if self.loss_type == 'cross_entropy':
                    loss = F.cross_entropy(txt_feats_pooled, emotion_labels)
                elif self.loss_type == 'LabelSmoothSoftmaxCEV1':
                    loss = self.criterion(txt_feats_pooled, emotion_labels)
                if mode == "train":
                    return txt_feats_pooled, loss, loss, 0, 0, 0
                else:
                    return txt_feats_pooled, txt_feats, None

            # concatenate
            elif self.ablation_type == 'concat':
                multimodal_feats = self.ablation_net(text_output, attention_mask, audio_inputs, audio_length)
                multimodal_feats_pooled = self.multimodal_classifier(multimodal_feats)
                loss = None
                if self.loss_type == 'cross_entropy':
                    loss = F.cross_entropy(multimodal_feats_pooled, emotion_labels)
                elif self.loss_type == 'LabelSmoothSoftmaxCEV1':
                    loss = self.criterion(multimodal_feats_pooled, emotion_labels)
                if mode == "train":
                    return multimodal_feats_pooled, loss, 0, 0, 0, 0
                else:
                    return multimodal_feats_pooled, multimodal_feats, None

        else:
            aug_txt_feats, aug_audio_feats, multimodal_feature = self.scmi(text_output, attention_mask, audio_inputs,
                                                                           audio_length)
            txt_feats_pooled = self.txt_classifier(aug_txt_feats)
            audio_feats_pooled = self.audio_classifier(aug_audio_feats)
            multimodal_feats_pooled = self.multimodal_classifier(multimodal_feature)
            loss_supcon = self.supcon_loss(multimodal_feature, emotion_labels)

            loss_txt, loss_audio, loss_multimodal, loss_all = None, None, None, None
            if self.loss_type == 'cross_entropy':
                loss_txt = F.cross_entropy(txt_feats_pooled, emotion_labels)
                loss_audio = F.cross_entropy(audio_feats_pooled, emotion_labels)
                loss_multimodal = F.cross_entropy(multimodal_feats_pooled, emotion_labels)
                loss_all = loss_multimodal + loss_supcon * self.supcon_loss_weight + self.text_modal_weight * loss_txt + self.audio_modal_weight * loss_audio
                # w/o audio constraint
                # loss_all = loss_multimodal + loss_supcon * self.supcon_loss_weight + self.text_modal_weight * loss_txt
                #  w/o text constraint
                # loss_all = loss_multimodal + loss_supcon * self.supcon_loss_weight + self.audio_modal_weight * loss_audio
                # w/o constraint&SCL
                # loss_all = loss_multimodal
                # w/o constraint
                # loss_all = loss_multimodal + loss_supcon * self.supcon_loss_weight
                # opy(multimodal_feats_pooled, emotion_labels)

            elif self.loss_type == 'LabelSmoothSoftmaxCEV1':
                loss_txt = self.criterion(txt_feats_pooled, emotion_labels)
                loss_audio = self.criterion(audio_feats_pooled, emotion_labels)
                loss_multimodal = self.criterion(multimodal_feats_pooled, emotion_labels)
                loss_all = loss_multimodal + loss_supcon * self.supcon_loss_weight + self.text_modal_weight * loss_txt + self.audio_modal_weight * loss_audio
                # w/o audio constraint
                # loss_all = loss_multimodal + loss_supcon * self.supcon_loss_weight + self.text_modal_weight * loss_txt
                # w/o text constraint
                # loss_all = loss_multimodal + loss_supcon * self.supcon_loss_weight + self.audio_modal_weight * loss_audio
                # w/o constraint&SCL(Baseline)
                # loss_all = loss_multimodal
                # w/o constraint
                # loss_all = loss_multimodal + loss_supcon * self.supcon_loss_weight
            # emotion_logits, logits, loss_cls

            if mode == "train":
                return multimodal_feats_pooled, loss_all, loss_txt, loss_audio, loss_multimodal, loss_supcon,
            else:
                return multimodal_feats_pooled, multimodal_feature, loss_all
