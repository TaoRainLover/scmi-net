from model.mmi_module_v1 import MMI_Model
import librosa
from transformers import BertConfig
from utils.utils import create_processor
# from transformers import AutoTokenizer, BertConfig, AutoConfig, Wav2Vec2Model, RobertaModel

from transformers import BertTokenizer,BertModel,AutoTokenizer, BertConfig, RobertaModel, RobertaTokenizer, RobertaConfig
from torch.nn.utils.rnn import pad_sequence
import torch

# path_roberta_base = '../pretained_model/roberta-base'

path_wav2vec2_base = "../pretained_model/wav2vec2-base"
audio_processor = create_processor(path_wav2vec2_base)
model_name = '../pretained_model/roberta-base'
config = RobertaConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name) # 分词器
roberta_base_model = RobertaModel.from_pretrained(model_name).to('cuda')

def get_roberta_output(text):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    encoded_input = tokenizer(text, return_tensors='pt').to('cuda')

    batch_text_attention = encoded_input['attention_mask'] # 掩码位置
    output = roberta_base_model(**encoded_input)
    # last_hidden_state = output['last_hidden_state'].squeeze(0)
    last_hidden_state = output['last_hidden_state']
    last_hidden_state = pad_sequence(last_hidden_state, batch_first=True)
    text_length = last_hidden_state.shape[1]
    text_length_list = [text_length]
    text_length = torch.LongTensor([x for x in text_length_list])
    return last_hidden_state, text_length, batch_text_attention

def get_audio_output(audio_path):
    wave, sr = librosa.core.load(audio_path, sr=None)
    if len(wave) > 210000:
        wave = wave[:210000]
    audio_length = len(wave)
    audio_length = torch.LongTensor([audio_length])
    batch_audio = wave
    batch_audio = audio_processor(batch_audio, sampling_rate=16000).input_values
    batch_audio = [{"input_values": audio} for audio in batch_audio]
    batch_audio = audio_processor.pad(
        batch_audio,
        padding=True,
        return_tensors="pt",
    )
    return batch_audio['input_values'], audio_length

def collate(sample_list):

    batch_audio = [x['audio_input'] for x in sample_list]
    batch_augmented_audio = [x['augmented_audio_input'] for x in sample_list]
    batch_bert_text = [x['text_input'] for x in sample_list]
    batch_augmented_text = [x['augmented_text_input'] for x in sample_list]
    batch_asr_text = [x['asr_target'] for x in sample_list]

    #----------------tokenize and pad the audio_for_test----------------------#

    batch_audio = audio_processor(batch_audio, sampling_rate=16000).input_values

    batch_audio = [{"input_values": audio} for audio in batch_audio]
    batch_audio = audio_processor.pad(
            batch_audio,
            padding=True,
            return_tensors="pt",
        )

    with audio_processor.as_target_processor():
        label_features = audio_processor(batch_asr_text).input_ids

    label_features = [{"input_ids": labels} for labels in label_features]

    with audio_processor.as_target_processor():
        labels_batch = audio_processor.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )

    ctc_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    #-------------tokenize and pad augmented audio_for_test--------------------#

    batch_augmented_audio = audio_processor(batch_augmented_audio, sampling_rate=16000).input_values

    batch_augmented_audio = [{"input_values": audio} for audio in batch_augmented_audio]
    batch_augmented_audio = audio_processor.pad(
            batch_augmented_audio,
            padding=True,
            return_tensors="pt",
        )

    #----------------tokenize and pad the text----------------------#
    batch_text = tokenizer(batch_bert_text, padding=True, truncation=True, return_tensors="pt")
    batch_text_inputids = batch_text['input_ids']
    batch_text_attention = batch_text['attention_mask']

    #----------------tokenize and pad the augmented text-------------#
    batch_augmented_text = tokenizer(batch_augmented_text, padding=True, truncation=True, return_tensors="pt")
    batch_augmented_text_inputids = batch_augmented_text['input_ids']
    batch_augmented_text_attention = batch_augmented_text['attention_mask']

    #-----------------pad the pre-generated bert embeddings----------#
    bert_output = [x['bert_output'] for x in sample_list]
    bert_output = pad_sequence(bert_output,batch_first = True)

    #-----------------pad the pre-generated augmented bert embeddings----------#
    bert_output_augment = [x['bert_output_augment'] for x in sample_list]
    bert_output_augment = pad_sequence(bert_output_augment,batch_first = True)

    #----------------tokenize and pad the extras----------------------#
    audio_length = torch.LongTensor([x['audio_length'] for x in sample_list])
    text_length = torch.LongTensor([x['text_length'] for x in sample_list])

    augmented_audio_length = torch.LongTensor([x['augmented_audio_length'] for x in sample_list])
    # augmented_text_length = torch.LongTensor([x['text_length'] for x in sample_list])

    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
    batch_name = [x['audio_name'] for x in sample_list]

    target_labels = []

    for label_idx in range(4):
        temp_labels = []
        for idx, _label in enumerate(batch_label):
            if _label == label_idx:
                temp_labels.append(idx)

        target_labels.append(torch.LongTensor(temp_labels[:]))

    return (batch_text_inputids, batch_text_attention, text_length, bert_output), (batch_audio, audio_length), (ctc_labels, batch_label, target_labels), (bert_output_augment,batch_augmented_text_attention), \
        (batch_augmented_audio, augmented_audio_length)


if __name__ == '__main__':
    config_mmi = BertConfig('config.json')
    mmi = MMI_Model(config_mmi).cuda()
    print(mmi)
    text = 'Clearly. You know, do you have like a supervisor or something?'
    audio_file_path = '../dataset/audio_for_test/Ses01F_impro01/Ses01F_impro01_F014.wav'
    roberta_output, text_length, batch_text_attention = get_roberta_output(text)
    print(roberta_output.shape)
    batch_audio, audio_length = get_audio_output(audio_file_path)

    # print(batch_audio['input_values'].shape())
    roberta_output = roberta_output.cuda()
    batch_text_attention = batch_text_attention.cuda()
    batch_audio = batch_audio.cuda()
    audio_length = audio_length.cuda()
    output = mmi(roberta_output, batch_text_attention, batch_audio, audio_length)
    print(output.shape)
