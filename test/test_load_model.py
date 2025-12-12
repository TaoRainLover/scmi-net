#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/2/27 20:05
# @Author : Tao Zhang
# @Email: 2637050370@qq.com


import argparse
import os
import time
from multiprocessing import freeze_support

import pandas as pd
import yaml
import torch
from tqdm import tqdm
from utils.IEMOCAPDataset import IEMOCAPDataset, collate


def generate_classification_feats(args, config):
    # 加载预训练模型的参数
    batch_size = args.batch_size
    num_workers = args.num_workers
    model_path = args.model_path  # 替换为你的模型文件的路径
    model = torch.load(model_path)

    # 打印模型的所有层信息
    for name, module in model.named_modules():
        print(name)

    # 设置模型为评估模式
    model.eval()
    pred_y, true_y = [], []
    classification_feats = []
    valid_data = []
    df_emotion = pd.read_csv(args.csv_path)

    valid_session = "Ses0" + str(args.session_id)
    valid_data_csv = df_emotion[df_emotion["FileName"].str.match(valid_session)]
    train_data_csv = pd.DataFrame(df_emotion, index=list(
        set(df_emotion.index).difference(set(valid_data_csv.index)))).reset_index(drop=True)
    valid_data_csv.reset_index(drop=True, inplace=True)

    for row in valid_data_csv.itertuples():
        file_name = os.path.join(args.data_path_audio + row.FileName)
        bert_path = args.data_path_roberta + row.FileName
        valid_data.append((file_name, bert_path, row.Sentences, row.Label, row.text))

    valid_dataset = IEMOCAPDataset(config, valid_data)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, collate_fn=collate,
        shuffle=False, num_workers=num_workers
    )

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        time.sleep(2)  # avoid the deadlock during the switch between the different dataloaders
        for bert_input, audio_input, label_input in tqdm(valid_loader):
            torch.cuda.empty_cache()
            attention_mask, text_length, bert_output = bert_input[1].to(device), bert_input[2].to(device), bert_input[
                3].to(
                device)
            acoustic_input, acoustic_length = audio_input[0]['input_values'].to(device), audio_input[1].to(device)
            emotion_labels = label_input[0].to(device)



            logits, cls_loss = model(bert_output, attention_mask, acoustic_input, acoustic_length, emotion_labels,
                                     mode="valid")
            print('loss:{}'.format(cls_loss))
            classification_feats.extend(list(logits.cpu().numpy()))
            true_y.extend(list(emotion_labels.cpu().numpy()))

    # 生成最后用于分类的特征向量表示
    print(classification_feats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='../config/iemocap-ours.yaml', help='configuration file path')
    parser.add_argument("--bert_config", default='../config/config.json', type=str,
                        help='configuration file path for BERT')
    parser.add_argument("--epochs", type=int, default=50, help="training epoches")

    parser.add_argument("--num_labels", type=int, default=4, help="the number of classification label")
    parser.add_argument("--csv_path", type=str, default='../dataset/IEMOCAP/iemocap.csv', help="path of csv")
    parser.add_argument("--save_path", type=str, default="../results/iemocap/",
                        help="report or ckpt save path")  # checkpoint保存路径
    parser.add_argument("--dataset", type=str, default="iemocap", help="the using dataset")
    parser.add_argument("--data_path_audio", type=str, default='../dataset/IEMOCAP/audio/',
                        help="path to raw audio_for_test wav files")
    parser.add_argument("--data_path_roberta", type=str, default='../dataset/IEMOCAP/roberta/',
                        help="path to roberta embeddings for text")
    # TODO：修改学习率
    # parser.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate for the specific run")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate for the specific run")
    parser.add_argument("--batch_size", type=int, default=12, help="batch size")
    parser.add_argument("--accum_grad", type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers in data loader")
    # TODO: model version
    parser.add_argument("--model_version", type=str, default="v4", help="model version of mmi_module")
    parser.add_argument("--loss_type", type=str, default="cross_entropy", help="training loss type")
    # parser.add_argument("--loss_type", type=str, default="LabelSmoothSoftmaxCEV1", help="training loss type")
    parser.add_argument("--device", type=str, default="cpu", help="training device(cpu or cuda)")
    parser.add_argument("--text_modal_loss_weight", type=float, default=0.5,
                        help="weight of text-modal loss function")
    parser.add_argument("--audio_modal_loss_weight", type=float, default=0.5,
                        help="weight of audio-modal loss function")

    parser.add_argument("--supcon_loss_weight", type=float, default=0.1, help="weight of supcon loss")
    # 消融实验参数
    parser.add_argument("--ablation", type=bool, default=False, help="whether doing the ablation study")
    parser.add_argument("--ablation_type", type=str, default="concat", help="audio/text/concat")
    # parser.add_argument("--save-model", type=bool, default=False, help="save model pt")
    parser.add_argument("--session_id", type=int, default=2, help="the id of classification feats of session")
    parser.add_argument("--model_path", type=str, default='C:/研生活/IDEA/MODEL/results/iemocap/20240227-SCMI模型特征可视化分析/2_model.pt')

    # mmi.downsample_final_txt
    # mmi.downsample_final_audio

    args = parser.parse_args()

    seed = 6
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    freeze_support()

    generate_classification_feats(args, config)

