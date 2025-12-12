import argparse
import os
import time
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import yaml
import torch
from tqdm import tqdm
from utils.IEMOCAPDataset import IEMOCAPDataset, collate
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def generate_classification_feats(args, config):
    # 加载预训练模型的参数
    batch_size = args.batch_size
    num_workers = args.num_workers
    model_path = args.model_path + args.session_id + args.model_name  # 替换为你的模型文件的路径
    model = torch.load(model_path)

    # 打印模型信息
    for name, module in model.named_modules():
        print(name)

    # 设置模型为评估模式
    model.eval()
    pred_y, true_y = [], []
    classification_feats = []
    classification_pooled_feats = []
    valid_data = []
    df_emotion = pd.read_csv(args.csv_path)

    valid_session = "Ses0" + args.session_id
    valid_data_csv = df_emotion[df_emotion["FileName"].str.match(valid_session)]
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

            multimodal_feats_pooled, multimodal_feature, loss_all = model(bert_output, attention_mask, acoustic_input, acoustic_length, emotion_labels,
                                     mode="valid")

            classification_feats.extend(list(multimodal_feature.cpu().numpy()))
            classification_pooled_feats.extend(list(multimodal_feats_pooled.cpu().numpy()))
            true_y.extend(list(emotion_labels.cpu().numpy()))

    # 将数组保存为NumPy文件
    np.save(args.model_path + args.session_id + '_clas sification_feats.npy', classification_feats)
    np.save(args.model_path + args.session_id + '_true_y.npy', true_y)

    # 生成最后用于分类的特征向量表示
    return classification_pooled_feats, classification_feats, true_y



def plot(classification_feats):
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(classification_feats)
    # 定义标签映射
    idx2label = {0:'Hap&Exc', 1:'Ang', 2:'Neu', 3:'Sad'}
    # 定义颜色映射，您可以根据需要修改颜色
    color_map = {0: '#440154', 1: '#31688e', 2: '#35b779', 3: '#fde726'}
    # 定义形状映射，您可以根据需要修改形状
    marker_map = {0: 'o', 1: 's', 2: '^', 3: 'D'}

    # 可视化 t-SNE 降维结果
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(true_y))

    # 绘制散点图，并使用真实标签作为颜色
    for label in unique_labels:
        indices = [i for i, y in enumerate(true_y) if y == label]
        plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], marker=marker_map[label], c=color_map[label],
                    label=idx2label[label])

    # 添加图例
    plt.legend()

    # 添加标题
    plt.title('t-SNE Visualization of Model Features')

    # 移除坐标值
    plt.xticks([])
    plt.yticks([])
    # 保存图形为PNG格式
    plt.savefig(args.model_path+str(args.session_id)+'_feature_visualization.png', dpi=260)
    # 显示图形
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",            type=str, default='../config/iemocap-ours.yaml', help='configuration file path')
    parser.add_argument("--bert_config",       type=str, default='../config/config.json',
                        help='configuration file path for BERT')

    parser.add_argument("--num_labels",        type=int, default=4, help="the number of classification label")
    parser.add_argument("--csv_path",          type=str, default='../../MODEL/dataset/IEMOCAP/iemocap.csv', help="path of csv")
    parser.add_argument("--dataset",           type=str, default="iemocap", help="the using dataset")
    parser.add_argument("--data_path_audio",   type=str, default='../../MODEL/dataset/IEMOCAP/audio/',
                        help="path to raw audio_for_test wav files")
    parser.add_argument("--data_path_roberta", type=str, default='../../MODEL/dataset/IEMOCAP/roberta/',
                        help="path to roberta embeddings for text")

    parser.add_argument("--batch_size",  type=int, default=12, help="batch size")
    parser.add_argument("--accum_grad",  type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers in data loader")

    parser.add_argument("--device",      type=str, default="cpu", help="training device(cpu or cuda)")

    # TODO: 可视化参数
    parser.add_argument("--session_id",  type=str, default='2', help="the id of classification feats of session")
    parser.add_argument("--model_path",  type=str, default='C:/研生活/IDEA/MODEL/results/iemocap/20240301-语音模态可视化分析/')
    parser.add_argument('--model_name',  type=str, default='_model.pt')
    parser.add_argument('--npy_name',    type=str, default='_classification_feats.npy')
    parser.add_argument('--true_y_name', type=str, default='_true_y.npy')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    freeze_support()
    # 获取模型生成的分类特征
    npy_file_path = args.model_path + args.session_id + args.npy_name
    true_y_file_path = args.model_path + args.session_id + args.true_y_name
    classification_feats, true_y = None, None
    if os.path.exists(npy_file_path):
        # 加载NumPy文件
        classification_feats = np.load(npy_file_path)
        true_y = np.load(true_y_file_path)
    else:
        # 加载模型生成模型分类特征
        classification_logits_feats, classification_feats, true_y = generate_classification_feats(args, config)
    # 可视化
    plot(classification_feats)