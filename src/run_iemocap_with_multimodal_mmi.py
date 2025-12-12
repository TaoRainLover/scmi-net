import os
import time
import datetime
import json
import argparse
import numpy as np
from tqdm import tqdm
import yaml
import pandas as pd
import pickle
import torch
import torch.optim as optim
from torchsummary import summary
from transformers import BertConfig
from utils.IEMOCAPDataset import IEMOCAPDataset, collate
from model.scmi_net_for_classfication import Multimodal_MMI
from utils.printer import goodluck
from sklearn import metrics
from src.metric_evaluation import confusion_matrix_iemocap

from utils.plot_confusion_matrix import plot_confusion_matrix, iemocap_classes, meld_classes


def run(args, config, train_data, valid_data, session):
    # PARAMETER SETTING
    num_workers = args.num_workers
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    accum_iter = args.accum_grad
    final_save_path = args.save_path
    device = args.device
    stats_file = open(os.path.join(final_save_path, session) + '_' + 'stats.txt', 'a', buffering=1)

    # PREPARE DATASET
    train_dataset = IEMOCAPDataset(config, train_data)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, collate_fn=collate,
        shuffle=True, num_workers=num_workers
    )
    valid_dataset = IEMOCAPDataset(config, valid_data)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, collate_fn=collate,
        shuffle=False, num_workers=num_workers
    )

    # CREATE MODEL
    print("*" * 40)
    print("Create model")
    print("*" * 40)
    config_mmi = BertConfig(args.bert_config)
    model = Multimodal_MMI(config_mmi, args)
    del config_mmi

    # MODEL PARAMS COUNT
    print("*" * 40)
    print(sum(p.numel() for p in model.parameters()))
    print(f"Total Trainable Model's Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("*" * 40)

    print("*" * 40)
    print("Load to CUDA")
    print("*" * 40)

    model.to(device)

    print("*" * 40)
    print("Loaded to CUDA ...")
    print("*" * 40)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 日常迷信 #
    goodluck()
    # 日常迷信 #

    # Training
    count, best_metric, save_metric, best_epoch = 0, -np.inf, None, 0
    save_metric_old, save_metric = None, None
    best_validation_result_old, best_validation_result = 0, 0

    for epoch in range(epochs):
        epoch_train_loss = []
        epoch_train_cls_loss = []

        model.train()
        start_time = time.time()  # start time
        batch_idx = 0
        time.sleep(2)  # avoid the deadlock during the switch between the different dataloaders
        progress = tqdm(train_loader, desc='Training Epoch {:0>3d}'.format(epoch))

        for bert_input, audio_input, label_input in progress:
            # prepare data
            attention_mask, text_length, bert_output = bert_input[1].to(device), bert_input[2].to(device), bert_input[
                3].to(device)
            acoustic_input, acoustic_length = audio_input[0]['input_values'].to(device), audio_input[1].to(device)
            emotion_labels = label_input[0].to(device)

            # forward
            logits, cls_loss, loss_txt, loss_audio, loss_multimodal, loss_supcon = model(bert_output, attention_mask,
                                                                                         acoustic_input,
                                                                                         acoustic_length,
                                                                                         emotion_labels)

            # backward
            loss = cls_loss / accum_iter
            loss.backward()

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                epoch_train_loss.append(loss)
                epoch_train_cls_loss.append(cls_loss)
                optimizer.step()
                optimizer.zero_grad()  # 清除梯度

            batch_idx += 1
            count += 1
            acc_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
            cls_loss = torch.mean(torch.tensor(epoch_train_cls_loss)).cpu().detach().numpy()

            # step
            progress.set_description("Training Epoch {:0>3d} - total_loss {:.4f} - (multimodal_Loss {:.4f} - "
                                     "txt_loss {:.4f} - audio_loss {:.4f} - supcon_loss {:.4f})"
                                     .format(epoch, acc_train_loss, loss_multimodal, loss_txt, loss_audio, loss_supcon))

        # valid set
        del progress
        model.eval()
        pred_y, true_y = [], []
        with torch.no_grad():
            time.sleep(2)  # avoid the deadlock during the switch between the different dataloaders

            for bert_input, audio_input, label_input in tqdm(valid_loader):
                torch.cuda.empty_cache()
                attention_mask, text_length, bert_output = bert_input[1].to(device), bert_input[2].to(device), \
                    bert_input[3].to(device)
                acoustic_input, acoustic_length = audio_input[0]['input_values'].to(device), audio_input[1].to(device)
                emotion_labels = label_input[0].to(device)
                true_y.extend(list(emotion_labels.cpu().numpy()))
                logits, multimodal_feature, cls_loss = model(bert_output, attention_mask, acoustic_input,
                                                             acoustic_length, emotion_labels, mode="valid")
                prediction = torch.argmax(logits, axis=1)
                label_outputs = prediction.cpu().detach().numpy().astype(int)
                pred_y.extend(list(label_outputs))

        # evaluation
        unweighted_precision = metrics.precision_score(true_y, pred_y, average='macro')
        weighted_precision = metrics.precision_score(true_y, pred_y, average='weighted')
        confusion_matrix = confusion_matrix_iemocap(true_y, pred_y)[0]

        epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S",
                                                                                        time.gmtime(elapsed_time)))

        print(
            f"Valid Metric: [WA: {weighted_precision}, UA: {unweighted_precision}, AVG: {(weighted_precision + unweighted_precision) / 2}], Train Loss: {epoch_train_loss}")

        stats = dict(date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch=epoch, wa=weighted_precision,
                     ua=unweighted_precision, avg=(weighted_precision + unweighted_precision) / 2)
        print(json.dumps(stats), file=stats_file)
        print("\n", file=stats_file)

        # record best result of metric
        validation_result = weighted_precision + unweighted_precision
        if validation_result > best_validation_result:
            print('\nBetter Metric(new) found on Test, saving checkpoints')

            # TODO：save best model
            # torch.save(model, os.path.join(final_save_path, session) + '_' + "model.pt")  # save entire model
            best_validation_result, best_epoch = validation_result, epoch
            best_confusion_matrix = confusion_matrix
            plot_confusion_matrix(best_confusion_matrix, classes=iemocap_classes, normalize=True,
                                  title='Normalized Confusion Matrix of IEMOCAP',
                                  save_path=os.path.join(final_save_path) + "/" + session + '_confusion_matrix.png')
            print(
                f"Valid Metric: [wa: {weighted_precision}, ua: {unweighted_precision}, wa+ua: {weighted_precision + unweighted_precision}], Train Loss: {epoch_train_loss}")

            if args.save_model:
                torch.save({'state_dict': model.state_dict()},
                           os.path.join(final_save_path) + "/" + session + "_best_model.pt")

            save_metric = (weighted_precision, unweighted_precision)

    print(
        f"Session-{session} Training End. Best epoch: {best_epoch}, metric(new): [wa: {weighted_precision}, ua: {unweighted_precision}, wa_plus_ua: {weighted_precision + unweighted_precision}]")
    return save_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='../config/iemocap-ours.yaml', help='configuration file path')
    parser.add_argument("--bert_config", default='../config/config.json', type=str, help='config file path for BERT')
    parser.add_argument("--device", type=str, default="cpu", help="training device(cpu or cuda)")

    # dataset setting
    parser.add_argument("--num_labels", type=int, default=4, help="the number of classification label")
    parser.add_argument("--csv_path", type=str, default='../../MODEL/dataset/IEMOCAP/iemocap.csv', help="path of csv")
    parser.add_argument("--save_path", type=str, default="../results/iemocap/",
                        help="report or ckpt save path")  # checkpoint保存路径
    parser.add_argument("--dataset", type=str, default="iemocap", help="the using dataset")
    parser.add_argument("--data_path_audio", type=str, default='../../MODEL/dataset/IEMOCAP/audio/',
                        help="path to raw audio wav files")
    parser.add_argument("--data_path_roberta_embedding", type=str, default='../../MODEL/dataset/IEMOCAP/roberta/',
                        help="path to roberta embeddings for text")

    # TODO：学习率
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate for the specific run")
    parser.add_argument("--epochs", type=int, default=50, help="training epoches")
    parser.add_argument("--batch_size", type=int, default=12, help="batch size")
    parser.add_argument("--accum_grad", type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers in data loader")

    # loss type: cross_entropy or LabelSmoothSoftmaxCEV1
    parser.add_argument("--loss_type", type=str, default="cross_entropy", help="training loss type")  # 交叉熵损失
    # parser.add_argument("--loss_type", type=str, default="LabelSmoothSoftmaxCEV1", help="training loss type") # 带标签平衡处理的交叉熵损失

    # loss weight setting
    parser.add_argument("--text_modal_loss_weight", type=float, default=0.5,
                        help="weight of text constraint loss function")  # 文本模态约束权重
    parser.add_argument("--audio_modal_loss_weight", type=float, default=0.5,
                        help="weight of audio constraint loss function")  # 语音模态约束权重
    parser.add_argument("--supcon_loss_weight", type=float, default=0.1, help="weight of supcon loss")

    # 消融实验参数
    parser.add_argument("--ablation", type=bool, default=False, help="whether doing the ablation study")
    parser.add_argument("--ablation_type", type=str, default="audio", help="audio/text/concat")
    parser.add_argument("--save-model", type=bool, default=False, help="save model pt")

    args = parser.parse_args()

    seed = 6
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    df_emotion = pd.read_csv(args.csv_path)

    # device 默认设置为 CPU，有 cuda 会自动更换为 cuda
    args.device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {args.device}")

    # report result
    report_wa_result, report_ua_result, report_result = [], [], []

    # create saving result dir
    training_date = datetime.datetime.now().strftime('%Y%m%d')
    save_dir = args.save_path + training_date
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_path = save_dir

    # create result file for recording each session result
    new_metric_result_file = open(os.path.join(args.save_path) + '/result(new).txt', 'a', buffering=1)

    # recording training date
    print(json.dumps({"data": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), file=new_metric_result_file)

    # print arguments
    argument_str = ''
    print('\n' + '*' * 40)
    for k, v in sorted(vars(args).items()):
        print(f'{k:>15}: {v}')
        print(json.dumps({k: v}), file=new_metric_result_file)
    print('*' * 40 + '\n')

    # 5-fold validation
    for i in range(1, 6):
        # split data
        valid_session = "Ses0" + str(i)
        valid_data_csv = df_emotion[df_emotion["FileName"].str.match(valid_session)]
        train_data_csv = pd.DataFrame(df_emotion, index=list(
            set(df_emotion.index).difference(set(valid_data_csv.index)))).reset_index(drop=True)
        valid_data_csv.reset_index(drop=True, inplace=True)

        # load data
        train_data = []
        valid_data = []
        for row in train_data_csv.itertuples():
            file_name = os.path.join(args.data_path_audio + row.FileName)
            bert_path = args.data_path_roberta_embedding + row.FileName
            train_data.append((file_name, bert_path, row.Sentences, row.Label, row.text))

        for row in valid_data_csv.itertuples():
            file_name = os.path.join(args.data_path_audio + row.FileName)
            bert_path = args.data_path_roberta_embedding + row.FileName
            valid_data.append((file_name, bert_path, row.Sentences, row.Label, row.text))

        # run
        report_metric = run(args, config, train_data, valid_data, str(i))
        report_result.append(report_metric)

        # save valid result of cur session
        print(report_metric)
        result_dict_new = dict(session_id=i, wa=report_metric[0], ua=report_metric[1])

        report_wa_result.append(report_metric[0])
        report_ua_result.append(report_metric[1])

        print(json.dumps(result_dict_new), file=new_metric_result_file)

        pickle.dump(report_result, open(os.path.join(args.save_path, 'metric_report.pkl'), 'wb'))

    # average result for 5-fold validation
    print(json.dumps({"avg_wa": sum(report_wa_result) / len(report_wa_result),
                      "avg_ua": sum(report_ua_result) / len(report_ua_result)}), file=new_metric_result_file)
