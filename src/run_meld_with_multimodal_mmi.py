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
from transformers import BertConfig
from utils.MeldDataset import MELDDataset, meld_collate
from model.scmi_net_for_classfication import Multimodal_MMI
from sklearn import metrics
from src.metric_evaluation import confusion_matrix_meld
from utils.MeldDataset import label2idx

from utils.plot_confusion_matrix import plot_confusion_matrix, iemocap_classes, meld_classes


def evaluation(model, data_loader, device, num_labels=7):
    model.eval()
    pred_y, true_y = [], []
    with torch.no_grad():
        time.sleep(2)  # avoid the deadlock during the switch between the different dataloaders
        for bert_input, audio_input, label_input in tqdm(data_loader):
            torch.cuda.empty_cache()
            attention_mask, text_length, bert_output = bert_input[1].to(device), bert_input[2].to(device), bert_input[
                3].to(device)
            acoustic_input, acoustic_length = audio_input[0]['input_values'].to(device), audio_input[1].to(device)
            emotion_labels = label_input.to(device)
            true_y.extend(list(emotion_labels.cpu().numpy()))
            logits, multimodal_feature, cls_loss = model(bert_output, attention_mask, acoustic_input, acoustic_length, emotion_labels,
                                     mode="valid")
            prediction = torch.argmax(logits, axis=1)
            label_outputs = prediction.cpu().detach().numpy().astype(int)
            pred_y.extend(list(label_outputs))

    # evaluation
    unweighted_precision = metrics.precision_score(true_y, pred_y, average='macro')
    weighted_precision = metrics.precision_score(true_y, pred_y, average='weighted')
    WA = metrics.recall_score(true_y, pred_y, average='weighted')
    WF1 = metrics.f1_score(true_y, pred_y, average='weighted')
    UF1 = metrics.f1_score(true_y, pred_y, average='macro')
    report = metrics.classification_report(true_y, pred_y, digits=3)
    confusion_matrix = confusion_matrix_meld(true_y, pred_y, num_labels)[0]

    return unweighted_precision, weighted_precision, WA, WF1, UF1, report, confusion_matrix


def run(args, config, train_data, valid_data, test_data, k_fold):
    # PARAMETER SETTING
    num_workers = args.num_workers
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    accum_iter = args.accum_grad
    final_save_path = args.save_path
    device = args.device
    stats_file = open(os.path.join(final_save_path, k_fold) + '_' + 'stats.txt', 'a', buffering=1)

    # PREPARE DATASET 
    train_dataset = MELDDataset(config, train_data, args)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, collate_fn=meld_collate,
        shuffle=True, num_workers=num_workers
    )
    valid_dataset = MELDDataset(config, valid_data, args)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, collate_fn=meld_collate,
        shuffle=False, num_workers=num_workers
    )

    test_dataset = MELDDataset(config, test_data, args)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, collate_fn=meld_collate,
        shuffle=False, num_workers=num_workers
    )
    # CREATE MODEL 

    print("*" * 40)
    print("Create model")
    print("*" * 40)

    config_mmi = BertConfig(args.bert_config)
    model = Multimodal_MMI(config_mmi, args)

    del config_mmi

    print("*" * 40)
    print("Model params")
    print(sum(p.numel() for p in model.parameters()))
    print(f"total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("*" * 40)

    print("*" * 40)
    print("Load to CUDA")
    print("*" * 40)

    model.to(device)

    print("*" * 40)
    print("Loaded to CUDA ...")
    print("*" * 40)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # training
    count, best_metric, save_metric, best_epoch = 0, -np.inf, None, 0
    save_metric_old, save_metric = None, None
    best_validation_result = 0
    best_test_result = 0
    best_confusion_matrix = None

    for epoch in range(epochs):
        epoch_train_loss = []
        epoch_train_cls_loss = []  # loss
        model.train()
        start_time = time.time()  # start time
        batch_idx = 0
        time.sleep(2)  # avoid the deadlock during the switch between the different dataloaders
        # TODO: 调试完记得将 valid_loader 修改回 train_loader
        progress = tqdm(train_loader, desc='Epoch {:0>3d}'.format(epoch))
        for bert_input, audio_input, label_input in progress:
            # prepare data
            attention_mask, text_length, bert_output = bert_input[1].to(device), bert_input[2].to(device), bert_input[
                3].to(device)
            acoustic_input, acoustic_length = audio_input[0]['input_values'].to(device), audio_input[1].to(device)
            emotion_labels = label_input.to(device)

            # forward

            logits, cls_loss, loss_txt, loss_audio, loss_multimodal, loss_supcon = model(bert_output, attention_mask,
                                                                            acoustic_input, acoustic_length,
                                                                            emotion_labels)

            loss = cls_loss / accum_iter
            loss.backward()  # backward

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                epoch_train_loss.append(loss)
                epoch_train_cls_loss.append(cls_loss)
                optimizer.step()
                optimizer.zero_grad()  # 将梯度置为零
            batch_idx += 1
            count += 1
            acc_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
            cls_loss = torch.mean(torch.tensor(epoch_train_cls_loss)).cpu().detach().numpy()

            # step
            progress.set_description("Training Epoch {:0>3d} - total_loss {:.4f} - (multimodal_Loss {:.4f} - txt_loss {:.4f} - audio_loss {:.4f})".format(epoch, cls_loss, loss_multimodal, loss_txt, loss_audio))

        # valid set
        del progress
        unweighted_precision, weighted_precision, WA, WF1, UF1, report, confusion_matrix = evaluation(model,
                                                                                                      valid_loader,
                                                                                                      device,
                                                                                                      args.num_labels)

        epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()

        elapsed_time = time.time() - start_time

        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S",
                                                                                        time.gmtime(elapsed_time)))
        print(
            f"Valid Metric: [weighted_precision: {weighted_precision}, unweighted_precision: {unweighted_precision}, AVG: {(weighted_precision + unweighted_precision) / 2}], Train Loss: {epoch_train_loss}")
        print(f"Valid Metric: [WA: {WA}, WF1: {WF1}, UF1: {UF1}], Train Loss: {epoch_train_loss}")
        print(f"report:\n {report}")

        # record best result of metric
        validation_result = WA + WF1

        if validation_result > best_validation_result:
            print('\nBetter Metric(new) found on Valid, saving checkpoints')
            # record
            stats = dict(type='valid', date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch=epoch,
                         weighted_precision=weighted_precision, unweighted_precision=unweighted_precision,
                         avg_wa_plua_ua=(weighted_precision + unweighted_precision) / 2,
                         WA=WA, WF1=WF1, UF1=UF1)
            print(json.dumps(stats), file=stats_file)
            best_validation_result, best_epoch_new = validation_result, epoch
            print(f"Valid Metric: [WA: {WA}, WF1: {WF1}, UF1: {UF1}], Train Loss: {epoch_train_loss}")

            # test set
            unweighted_precision, weighted_precision, WA, WF1, UF1, report, confusion_matrix = evaluation(model,
                                                                                                          test_loader,
                                                                                                          device,
                                                                                                          args.num_labels)
            test_result = WA + WF1

            if test_result > best_test_result:
                print(
                    f"Test Metric(1): [weighted_precision: {weighted_precision}, unweighted_precision: {unweighted_precision}, AVG: {(weighted_precision + unweighted_precision) / 2}]")
                print(f"Test Metric(2): [WA: {WA}, WF1: {WF1}, UF1: {UF1}]")
                print(f"report:\n {report}")
                # save test result
                stats = dict(type='test', date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch=epoch,
                             weighted_precision=weighted_precision, unweighted_precision=unweighted_precision,
                             avg_wa_plua_ua=(weighted_precision + unweighted_precision) / 2,
                             WA=WA, WF1=WF1, UF1=UF1)
                print(json.dumps(stats), file=stats_file)
                print("\n", file=stats_file)
                save_metric = (weighted_precision, unweighted_precision, WA, WF1, UF1)
                best_test_result = test_result
                best_confusion_matrix = confusion_matrix
                plot_confusion_matrix(best_confusion_matrix, classes=meld_classes[:args.num_labels], normalize=True,
                                      title='Normalized Confusion Matrix of MELD',
                                      save_path=os.path.join(final_save_path) + "/" + k_fold + '_confusion_matrix.png')

                if args.save_model:
                    torch.save({'state_dict': model.state_dict()},
                               os.path.join(final_save_path) + "/" + k_fold + "_best_model.pt")

            print(f"Current best test metric: [WA: {save_metric[2]}, WF1: {save_metric[3]}]")

    print(f"Session-{k_fold} Training End. Best epoch: {best_epoch}, metric(new): [wa: {weighted_precision}, ua: {unweighted_precision}, wa_plus_ua: {weighted_precision + unweighted_precision}]")

    return save_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='../config/iemocap-ours.yaml', help='configuration file path')
    parser.add_argument("--bert_config", default='../config/config.json', type=str,
                        help='configuration file path for BERT')
    parser.add_argument("--epochs", type=int, default=25, help="training epoches")
    parser.add_argument("--num_labels", type=int, default=5, help="the number of classification label")
    # parser.add_argument("--num_labels", type=int, default=7, help="the number of classification label")

    parser.add_argument("--save_path", type=str, default="../results/meld/",
                        help="report or ckpt save path")  

    parser.add_argument("--dataset", type=str, default="MELD", help="the using dataset")
    parser.add_argument("--train_data_path_audio", type=str, default='../../MODEL/dataset/MELD/audio/train/',
                        help="path to raw audio_for_test training wav files")
    parser.add_argument("--dev_data_path_audio", type=str, default='../../MODEL/dataset/MELD/audio/dev/',
                        help="path to raw audio_for_test dev wav files")
    parser.add_argument("--test_data_path_audio", type=str, default='../../MODEL/dataset/MELD/audio/test/',
                        help="path to raw audio_for_test test wav files")
    parser.add_argument("--train_data_path_roberta_embedding", type=str, default='../../MODEL/dataset/MELD/roberta/train/',
                        help="path to roberta embeddings for training text")
    parser.add_argument("--dev_data_path_roberta_embedding", type=str, default='../../MODEL/dataset/MELD/roberta/dev/',
                        help="path to roberta embeddings for dev text")
    parser.add_argument("--test_data_path_roberta_embedding", type=str, default='../../MODEL/dataset/MELD/roberta/test/',
                        help="path to roberta embeddings for test text")
    parser.add_argument("--train_csv_path", type=str, default='../../MODEL/dataset/MELD/train_sent_emo.csv',
                        help="path of training csv")
    parser.add_argument("--dev_csv_path", type=str, default='../../MODEL/dataset/MELD/dev_sent_emo.csv', help="path of dev csv")
    parser.add_argument("--test_csv_path", type=str, default='../../MODEL/dataset/MELD/test_sent_emo.csv',
                        help="path of test csv")

    parser.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate for the specific run")

    parser.add_argument("--batch_size", type=int, default=12, help="batch size")
    parser.add_argument("--accum_grad", type=int, default=4, help="gradient accumulation steps")

    parser.add_argument("--num_workers", type=int, default=4, help="number of workers in data loader")

    # parser.add_argument("--loss_type", type=str, default="cross_entropy", help="training loss type")
    parser.add_argument("--loss_type", type=str, default="LabelSmoothSoftmaxCEV1", help="training loss type")
    parser.add_argument("--device", type=str, default="cpu", help="training device(cpu or cuda)")
    parser.add_argument("--text_modal_loss_weight", type=float, default=0.1, help="weight of text-modal loss function")
    parser.add_argument("--audio_modal_loss_weight", type=float, default=0.1,
                        help="weight of audio-modal loss function")
    parser.add_argument("--supcon_loss_weight", type=float, default=0.1, help="weight of supcon loss")

    # 消融实验参数
    parser.add_argument("--ablation", type=bool, default=True, help="whether doing the ablation study")
    parser.add_argument("--ablation_type", type=str, default="concat", help="audio/text/concat")
    parser.add_argument("--save-model", type=bool, default=False, help="save model pt")

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


    args.device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {args.device}")

    seed = 6
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  

    report_wa_result = []
    report_ua_result = []
    report_result = []

    # create saving result dir
    training_date = datetime.datetime.now().strftime('%Y%m%d')
    save_dir = args.save_path + training_date
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_path = save_dir

    # create result file for recording each session result
    metric_result_file = open(os.path.join(args.save_path) + '/result(best).txt', 'a', buffering=1)

    # recording training date
    print(json.dumps({"data": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), file=metric_result_file)
    # Print arguments
    argument_str = ''
    print('\n' + '*' * 40)
    for k, v in sorted(vars(args).items()):
        print(f'{k:>15}: {v}')
        print(json.dumps({k: v}), file=metric_result_file)
    print('*' * 40 + '\n')

    df_train_emotion = pd.read_csv(args.train_csv_path, encoding='utf-8')
    df_dev_emotion = pd.read_csv(args.dev_csv_path, encoding='utf-8')
    df_test_emotion = pd.read_csv(args.test_csv_path, encoding='utf-8')

    train_data = []
    valid_data = []
    test_data = []

    # 5 class
    if args.num_labels == 5:
        for row in df_train_emotion.itertuples():
            if 0 <= label2idx[row.Emotion] <= 4:
                file_name = f'dia{row.Dialogue_ID}_utt{row.Utterance_ID}'
                audio_file_path = os.path.join(args.train_data_path_audio + file_name)
                text_embedding_path = args.train_data_path_roberta_embedding + file_name
                train_data.append(
                    (file_name, audio_file_path, text_embedding_path, row.Utterance, row.Emotion, row.Sentiment))

        for row in df_dev_emotion.itertuples():
            if 0 <= label2idx[row.Emotion] <= 4:
                file_name = f'dia{row.Dialogue_ID}_utt{row.Utterance_ID}'
                audio_file_path = os.path.join(args.dev_data_path_audio + file_name)
                text_embedding_path = args.dev_data_path_roberta_embedding + file_name
                valid_data.append(
                    (file_name, audio_file_path, text_embedding_path, row.Utterance, row.Emotion, row.Sentiment))

        for row in df_test_emotion.itertuples():
            if 0 <= label2idx[row.Emotion] <= 4:
                file_name = f'dia{row.Dialogue_ID}_utt{row.Utterance_ID}'
                audio_file_path = os.path.join(args.test_data_path_audio + file_name)
                text_embedding_path = args.test_data_path_roberta_embedding + file_name
                test_data.append(
                    (file_name, audio_file_path, text_embedding_path, row.Utterance, row.Emotion, row.Sentiment))

    # 7 class
    elif args.num_labels == 7:
        for row in df_train_emotion.itertuples():
            file_name = f'dia{row.Dialogue_ID}_utt{row.Utterance_ID}'
            audio_file_path = os.path.join(args.train_data_path_audio + file_name)
            text_embedding_path = args.train_data_path_roberta_embedding + file_name
            train_data.append(
                (file_name, audio_file_path, text_embedding_path, row.Utterance, row.Emotion, row.Sentiment))

        for row in df_dev_emotion.itertuples():
            file_name = f'dia{row.Dialogue_ID}_utt{row.Utterance_ID}'
            audio_file_path = os.path.join(args.dev_data_path_audio + file_name)
            text_embedding_path = args.dev_data_path_roberta_embedding_embedding + file_name
            valid_data.append(
                (file_name, audio_file_path, text_embedding_path, row.Utterance, row.Emotion, row.Sentiment))

        for row in df_test_emotion.itertuples():
            file_name = f'dia{row.Dialogue_ID}_utt{row.Utterance_ID}'
            audio_file_path = os.path.join(args.test_data_path_audio + file_name)
            text_embedding_path = args.test_data_path_roberta_embedding + file_name
            test_data.append(
                (file_name, audio_file_path, text_embedding_path, row.Utterance, row.Emotion, row.Sentiment))

    print("The count of training data:", len(train_data))
    print("The count of dev data:", len(valid_data))
    print("The count of test data:", len(test_data))

    # 多跑几次
    for i in range(1, 5):
        # run
        report_metric = run(args, config, train_data, valid_data, test_data, str(i))

        report_result.append(report_metric)

        final_save_path = args.save_path
        # save valid result of cur session
        print(report_metric)
        result_dict = dict(k_fold_id=i, weighted_precision=report_metric[0], unweigted_precision=report_metric[1],
                           WA=report_metric[2], WF1=report_metric[3], UF1=report_metric[4])

        report_wa_result.append(report_metric[0])
        report_ua_result.append(report_metric[1])

        print(json.dumps(result_dict), file=metric_result_file)

        pickle.dump(report_result, open(os.path.join(final_save_path, 'metric_report(best).pkl'), 'wb'))

    # average result
    print(json.dumps({"avg_wa": sum(report_wa_result) / len(report_wa_result),
                      "avg_ua": sum(report_ua_result) / len(report_ua_result)}), file=metric_result_file)
