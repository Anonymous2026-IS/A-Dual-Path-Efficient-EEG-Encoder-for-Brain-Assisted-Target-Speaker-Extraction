import os, sys, shutil, json, time
import argparse
from tqdm import tqdm
import torch
from dataset import cock_tail
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
sys.path.append('../')
import matplotlib.font_manager
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from DP_EEG_TSE import DP_EEG_TSE
import pandas as pd
from tools.VeryCustomSacred import CustomExperiment, ChooseGPU
from tools.utilities import timeStructured

from tools.plotting import evaluations_to_violins, one_plot_test

from tools.calculate_intelligibility import find_intel

import pickle
import numpy as np
from scipy.io import wavfile  # 导入 scipy 的 wavfile 模块

from tools.utilities import setReproducible

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
ex = CustomExperiment(random_string + '-mc-test', base_dir=CDIR, seed=100) 

@ex.config
def test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/experiments.json',
                        help='JSON file for configuration')
    args = parser.parse_args()
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    model_path = config["test"]["model_path"]
    dataset_root = config["test"]["dataset_root"]
    
@ex.automain
def test(model_path, dataset_root):
    testing = True
    exp_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    images_dir = os.path.join(*[exp_dir, 'images'])
    other_dir = os.path.join(*[exp_dir, 'other_outputs'])
    # 创建音频保存目录
    audio_dir = os.path.join(*[exp_dir, 'audio_outputs'])
    os.makedirs(audio_dir, exist_ok=True)
    
    device = "cuda"
    model = DP_EEG_TSE().to('cuda')

    print('loading model')
    check_point = torch.load(model_path)
    print('loading {}'.format(model_path))
    model.load_state_dict(check_point['model_state_dict'], strict=False)
    model.eval()

    ##############################################################
    #                    tests training
    ##############################################################
    with torch.no_grad():
        if testing:
            root_dir = dataset_root
            batch_size = 1
            print('testing the model')
            print('\n loading the test data')

            prediction_metrics = ['sdr','si-sdr', 'stoi', 'estoi', 'pesq']
            noisy_metrics = [m + '_noisy' for m in prediction_metrics]

            inference_time = []
            # 假设采样率为16000，根据实际情况调整
            fs = 16000
            exp_type = 'test'

            subjects = [None] + list(range(51))  # subject None means the whole testset

            for generator_type in ['test']:  # , 'test_unattended']:# ['test']:

                print(generator_type)
                all_subjects_evaluations = {}
                print('going through subjects')
                for subject in subjects:

                    prediction = []
                    df = pd.DataFrame(columns=prediction_metrics + noisy_metrics)
                    try:
                        test_data = cock_tail(root_dir, 'test', subject=subject)
                        test_loader = DataLoader(test_data, batch_size=batch_size)
                    except Exception as e:
                        print(f"Error loading data for subject {subject}: {e}")
                        continue
                        
                    print('Subject {}'.format(subject))
                    try:
                        for batch_sample, (noisy, eeg, clean) in tqdm(enumerate(test_loader)):
                            print('batch_sample {} for subject {}'.format(batch_sample, subject))
                            # 确保数据加载正确
                            if noisy is None or eeg is None or clean is None:
                                print(f"Empty batch for subject {subject}, batch {batch_sample}")
                                continue
                                
                            noisy, eeg, clean = noisy.to(device), eeg.to(device), clean.to(device)

                            noisy_snd, clean = noisy, clean
                            noisy_snd = noisy_snd.squeeze().unsqueeze(0).cpu().detach().numpy()

                            clean = clean.squeeze().unsqueeze(0).cpu().detach().numpy()
                            
                            # 确保初始化 intel_list 和 intel_list_noisy
                            intel_list, intel_list_noisy = [], []
                            
                            inf_start_s = time.time()
                            print('predicting')
                            noisy = torch.cat(torch.split(noisy, 29184, dim=2), dim=0)
                            eeg = torch.cat(torch.split(eeg, 29184, dim=2), dim=0)

                            pred,_ ,_ = model(noisy, eeg)
                            print(pred.shape)
                            inf_t = time.time() - inf_start_s
                            if subject is None:
                                inference_time.append(inf_t)
                            pred = torch.cat(torch.split(pred, 1, dim=0), dim=2)
                            pred = pred.squeeze().unsqueeze(0).cpu().detach().numpy()
                            prediction.append(pred)
                            prediction_concat = np.concatenate(prediction, axis=0)
                            
                            # 保存音频文件 - 使用 scipy.io.wavfile.write
                            pred_audio_path = os.path.join(audio_dir, f'pred_b{batch_sample}_s{subject}_g{generator_type}.wav')
                            clean_audio_path = os.path.join(audio_dir, f'clean_b{batch_sample}_s{subject}_g{generator_type}.wav')
                            noisy_audio_path = os.path.join(audio_dir, f'noisy_b{batch_sample}_s{subject}_g{generator_type}.wav')
                            
                            # 确保音频数据在 -1 到 1 的范围内
                            pred_normalized = np.int16(pred.squeeze() / np.max(np.abs(pred.squeeze())) * 32767)
                            clean_normalized = np.int16(clean.squeeze() / np.max(np.abs(clean.squeeze())) * 32767)
                            noisy_normalized = np.int16(noisy_snd.squeeze() / np.max(np.abs(noisy_snd.squeeze())) * 32767)
                            
                            # 保存音频文件
                            wavfile.write(pred_audio_path, fs, pred_normalized)
                            wavfile.write(clean_audio_path, fs, clean_normalized)
                            wavfile.write(noisy_audio_path, fs, noisy_normalized)
                            
                            fig_path = os.path.join(
                                images_dir,
                                'prediction_b{}_s{}_g{}.png'.format(batch_sample, subject, generator_type))
                            print('saving plot')
                            one_plot_test(pred, clean, noisy_snd, '', fig_path)

                            print('finding metrics')
                            for m in prediction_metrics:
                                print('     ', m)
                                try:
                                    pred_m = find_intel(clean, pred, metric=m)
                                    intel_list.append(pred_m)
                                    
                                    noisy_m = find_intel(clean, noisy_snd, metric=m)
                                    intel_list_noisy.append(noisy_m)
                                except Exception as e:
                                    print(f"Error calculating metric {m}: {e}")
                                    # 添加默认值
                                    intel_list.append(0)
                                    intel_list_noisy.append(0)

                            e_series = pd.Series(intel_list + intel_list_noisy, index=df.columns)
                            df = df.append(e_series, ignore_index=True)

                        if subject is None:
                            prediction_filename = os.path.join(
                                *[images_dir, 'prediction_{}_s{}_g{}.npy'.format('test', subject, generator_type)])
                            print('saving predictions')
                            np.save(prediction_filename, prediction_concat)

                        # 确保所有受试者的评估结果都被添加到字典中
                        subject_key = 'All Subjects' if subject is None else f'Subject {subject}'
                        all_subjects_evaluations[subject_key] = df

                        # 保存每个受试者的评估结果
                        df.to_csv(os.path.join(*[other_dir, 'evaluation_s{}_g{}.csv'.format(subject, generator_type)]),
                                  index=False)

                        # 绘制并保存每个受试者的指标图
                        if not df.empty:
                            fig, axs = plt.subplots(1, len(df.columns), figsize=(9, 4))
                            for ax, column in zip(axs, df.columns):
                                ax.set_title(column)
                                violin_handle = ax.violinplot(df[column])
                                violin_handle['bodies'][0].set_edgecolor('black')
                            fig.savefig(os.path.join(*[images_dir, 'metrics_s{}_g{}.png'.format(subject, generator_type)]),
                                        bbox_inches='tight')
                            plt.close('all')

                    except Exception as e:
                        print(f"Error processing subject {subject}: {e}")
                        # 创建一个空的 DataFrame 作为占位符
                        subject_key = 'All Subjects' if subject is None else f'Subject {subject}'
                        all_subjects_evaluations[subject_key] = pd.DataFrame(columns=prediction_metrics + noisy_metrics)
                        continue

                print('end of code, plotting violins')

                # 检查是否有评估数据
                if not all_subjects_evaluations:
                    print("No evaluation data available for violin plots")
                    continue
                    
                # 保存所有受试者的评估结果
                path_to_test = os.path.join(*[other_dir, 'all_subjects_evaluations_{}.pkl'.format(generator_type)])
                with open(path_to_test, "wb") as a_file:
                    pickle.dump(all_subjects_evaluations, a_file)

                # 加载pickle文件
                with open(path_to_test, "rb") as a_file:
                    all_subjects_evaluations = pickle.load(a_file)
    
                prediction_metrics = ['sdr','si-sdr', 'stoi','estoi','pesq']
                noisy_metrics = [m + '_noisy' for m in prediction_metrics]
                generator_type = 'test'

                # 确保有数据可以绘制
                if all_subjects_evaluations:
                    # 过滤掉空的数据框
                    valid_evaluations = {k: v for k, v in all_subjects_evaluations.items() if not v.empty}
                    if valid_evaluations:
                        evaluations_to_violins({k: v[noisy_metrics] for k, v in valid_evaluations.items()}, images_dir,
                                             generator_type + 'noisy')
                        evaluations_to_violins({k: v[prediction_metrics] for k, v in valid_evaluations.items()}, images_dir,
                                              generator_type + '')
                    else:
                        print("No valid data available for violin plots")
                else:
                    print("No data available for violin plots")

    shutil.make_archive(ex.observers[0].basedir, 'zip', exp_dir)