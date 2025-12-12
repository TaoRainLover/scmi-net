import numpy as np
import torch
import csv
from transformers import BertTokenizer,BertModel, BertConfig, RobertaModel, RobertaTokenizer, RobertaConfig
from utils.MeldDataset import text_preprocessing

model_name = '../pretained_model/roberta-base'
config = RobertaConfig.from_pretrained(model_name)	# 这个方法会自动从官方的s3数据库下载模型配置、参数等信息（代码中已配置好位置）
tokenizer = RobertaTokenizer.from_pretrained(model_name)	 # 这个方法会自动从官方的s3数据库读取文件下的vocab.txt文件
model = RobertaModel.from_pretrained(model_name).to('cuda')		# 这个方法会自动从官方的s3数据库下载模型信息
print(model)

csv_data_path = '../dataset/SAVEE/SAVEE_large.csv'

device = "cuda" if torch.cuda.is_available() else "cpu"
print('device: ', device)

file_name_list = []
utterance_list = []

with open(csv_data_path, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        file_name_list.append(row['file_name'])
        utterance_list.append(row['text'])


print(f'dataset length: {len(utterance_list)} - {len(file_name_list)}')

for file_name, utterance in zip(file_name_list, utterance_list):
    file_name = file_name.split('.')[0]
    print(f'{file_name}: {utterance}')
    encoded_input = tokenizer(utterance, return_tensors='pt').to('cuda')
    text_embedding = model(**encoded_input)['last_hidden_state']
    save_path = '../dataset/SAVEE/roberta/' + file_name
    np.save(save_path, text_embedding.cpu().detach().numpy())


