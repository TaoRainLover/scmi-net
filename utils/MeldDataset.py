# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import random
import re
import librosa
import warnings

from utils.utils import create_processor
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore")
path_roberta_base = '../pretained_model/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(path_roberta_base)
path_wav2vec2_base = "../pretained_model/wav2vec2-base"
# audio_processor = create_processor("facebook/wav2vec2-base")
audio_processor = create_processor(path_wav2vec2_base)

vocabulary_chars_str = "".join(t for t in audio_processor.tokenizer.get_vocab().keys() if len(t) == 1)

vocabulary_text_cleaner = re.compile(  # remove characters not in vocabulary
    f"[^\s{re.escape(vocabulary_chars_str)}]",  # allow space in addition to chars in vocabulary
    flags=re.IGNORECASE if audio_processor.tokenizer.do_lower_case else 0,
)

# 7 classes

label2idx = {
    'neutral': 0,
    'surprise': 1,
    'joy': 2,
    'sadness': 3,
    'anger': 4,
    'fear': 5,
    'disgust': 6
}


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # text = text.replace('', '\'')

    # Remove '@name'
    text = text.replace('\x92', '\'')
    text = re.sub("[\(\[].*?[\)\]]", '', text)

    # Replace '&amp;' with '&'
    text = re.sub(" +", ' ', text).strip()

    return text


class MELDDataset(object):
    def __init__(self, config, data_list, args):
        self.data_list = data_list
        self.num_labels = args.num_labels

        # self.unit_length = int(8 * 16000)
        # self.audio_length = config['acoustic']['audio_length']
        # self.feature_name = config['acoustic']['feature_name']
        # self.feature_dim = config['acoustic']['embedding_dim']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        file_name, audio_file_path, text_embedding_path, utterance, emotion, sentiment = self.data_list[index]
        audio_name = os.path.basename(audio_file_path)
        # print(audio_name)`
        # ------------- extract the audio_for_test features -------------#
        wave, sr = librosa.core.load(audio_file_path + ".wav", sr=None)

        # precautionary measure to fit in a 24GB gpu, feel free to comment the next 2 lines
        if len(wave) > 210000:
            wave = wave[:210000]

        audio_length = len(wave)

        # ------------- extract the text contexts -------------#
        tokenized_word = np.load(text_embedding_path + ".npy")
        tokenized_word = torch.from_numpy(tokenized_word).squeeze(0)
        text_length = tokenized_word.shape[0]

        bert_text = text_preprocessing(utterance)

        # ------------- labels to id -------------#
        label = label2idx[emotion]

        # ------------- wrap up all the output info the dict format -------------#
        return {'audio_input': wave, 'text': bert_text, 'audio_length': audio_length,
                'text_length': text_length, 'label': label, 'audio_name': audio_name, 'bert_output': tokenized_word}


def meld_collate(sample_list):
    batch_audio = [x['audio_input'] for x in sample_list]
    batch_origin_text = [x['text'] for x in sample_list]

    # ----------------tokenize and pad the audio_for_test----------------------#

    batch_audio = audio_processor(batch_audio, sampling_rate=16000).input_values

    batch_audio = [{"input_values": audio} for audio in batch_audio]
    batch_audio = audio_processor.pad(batch_audio, padding=True, return_tensors="pt")

    # with audio_processor.as_target_processor():
    #     label_features = audio_processor(batch_asr_text).input_ids
    #
    # label_features = [{"input_ids": labels} for labels in label_features]
    #
    # with audio_processor.as_target_processor():
    #     labels_batch = audio_processor.pad(
    #             label_features,
    #             padding=True,
    #             return_tensors="pt",
    #         )
    #
    # ctc_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    # ----------------tokenize and pad the text----------------------#
    batch_text = tokenizer(batch_origin_text, padding=True, truncation=True,
                           return_tensors="pt")  # padding到batch中最长句的长度
    batch_text_input_ids = batch_text['input_ids']
    batch_text_attention = batch_text['attention_mask']

    # -----------------pad the pre-generated bert embeddings----------#
    bert_output = [x['bert_output'] for x in sample_list]
    bert_output = pad_sequence(bert_output, batch_first=True)  # (batch_size, max_sequence_length, dim_length)

    # ----------------tokenize and pad the extras----------------------#
    audio_length = torch.LongTensor(
        [x['audio_length'] for x in sample_list])  # tensor([item_auidio_length, item2_audio_length ...])
    text_length = torch.LongTensor([x['text_length'] for x in sample_list])

    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
    # batch_name = [x['audio_name'] for x in sample_list]

    # target_labels = []
    #
    # for label_idx in range(4):
    #     temp_labels = []
    #     for idx, _label in enumerate(batch_label):
    #         if _label == label_idx:
    #             temp_labels.append(idx)
    #
    #     target_labels.append(torch.LongTensor(temp_labels[:]))

    return (batch_text_input_ids, batch_text_attention, text_length, bert_output), (batch_audio, audio_length), (
        batch_label)
