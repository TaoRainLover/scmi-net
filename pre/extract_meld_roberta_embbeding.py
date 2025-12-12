import numpy as np
import torch
import csv
from transformers import BertTokenizer,BertModel, BertConfig, RobertaModel, RobertaTokenizer, RobertaConfig
from utils.MeldDataset import text_preprocessing

model_name = '../pretained_model/roberta-base'
config = RobertaConfig.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name).to('cuda')
print(model)

types = ['train', 'dev', 'test']

for type in types:
    if type == 'train':
        csv_data_path = 'dataset/MELD/train_sent_emo.csv'
    elif type == 'dev':
        csv_data_path = 'dataset/MELD/dev/dev_sent_emo.csv'
    elif type == 'test':
        csv_data_path = 'dataset/MELD/test/test_sent_emo.csv'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device: ', device)

    dialogue_id_list = []
    utterance_id_list = []
    utterance_list = []

    with open(csv_data_path, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dialogue_id_list.append(row['Dialogue_ID'])
            utterance_id_list.append(row['Utterance_ID'])
            utterance_list.append(row['Utterance'])

    for i in range(len(utterance_list)):
        utterance_list[i] = text_preprocessing(utterance_list[i])


    print(f'{type}set: {len(utterance_list)}, {len(dialogue_id_list)}, {len(utterance_id_list)}')

    for dialogue_id, utterance_id, utterance in zip(dialogue_id_list, utterance_id_list, utterance_list):
        print(f'{dialogue_id}-{utterance_id}: {utterance}')
        encoded_input = tokenizer(utterance, return_tensors='pt').to('cuda')
        text_embedding = model(**encoded_input)['last_hidden_state']
        save_path = '../dataset/MELD/roberta/' + type + '/dia' + dialogue_id + '_utt' + utterance_id
        np.save(save_path, text_embedding.cpu().detach().numpy())


