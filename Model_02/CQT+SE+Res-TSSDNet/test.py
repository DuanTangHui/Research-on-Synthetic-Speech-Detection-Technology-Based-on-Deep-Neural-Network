import torch
from torch.utils.data.dataloader import DataLoader
from data import PrepASV19Dataset
import models
import torch.nn.functional as F
import numpy as np
import eval_metrics as em
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
def asv_cal_accuracies(protocol, path_data, net, device, data_type='time_frame', dataset=19):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        softmax_acc = 0
        num_files = 0
        probs = torch.empty(0, 3).to(device)
        batch_score = torch.empty(0,1).to(device)

        test_set = PrepASV19Dataset(protocol, path_data, data_type=data_type)

        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

        for test_batch in test_loader:
            # load batch and infer
            test_sample, test_label, sub_class = test_batch

            # # sub_class level test, comment if unwanted
            # # train & dev 0~6; eval 7~19
            # # selected_index = torch.nonzero(torch.logical_xor(sub_class == 10, sub_class == 0))[:, 0]
            # selected_index = torch.nonzero(sub_class.ne(10))[:, 0]
            # if len(selected_index) == 0:
            #     continue
            # test_sample = test_sample[selected_index, :, :]
            # test_label = test_label[selected_index]

            num_files += len(test_label)
            test_sample = test_sample.to(device)
            test_label = test_label.to(device)
            
            infer = net(test_sample)
            # 获取分数
            score = (infer[:, 1]).unsqueeze(-1)
            batch_score = torch.cat((batch_score, score), dim=0)
            # obtain output probabilities
            t1 = F.softmax(infer, dim=1)
            t2 = test_label.unsqueeze(-1)
            row = torch.cat((t1, t2), dim=1)
            probs = torch.cat((probs, row), dim=0)
            # print(probs)
            # calculate example level accuracy
            infer = infer.argmax(dim=1)
            batch_acc = infer.eq(test_label).sum().item()
            softmax_acc += batch_acc

        softmax_acc = softmax_acc / num_files
    # 存储分数
    x = batch_score.cpu().numpy()
    x = x.tolist()
    save_path = './score/cmdevscore.trl.txt'
    with open(save_path, "w") as fh:
        for sco in x:
            fh.write("{}\n".format(str(sco)[1:-1]))  # [-3.22324123245] 去除[ ]然后转为str存储
    print("Scores saved to {}".format(save_path))

    return softmax_acc, probs.to('cpu')

def cal_roc_eer(probs, show_plot=True):
    """
    probs: tensor, number of samples * 3, containing softmax probabilities
    row wise: [genuine prob, fake prob, label] 【真概率，假概率，真假布尔值】所有文件的集合
    TP: True Fake
    FP: False Fake
    """
    all_labels = probs[:, 2]
    zero_index = torch.nonzero((all_labels == 0)).squeeze(-1)
    one_index = torch.nonzero(all_labels).squeeze(-1)
    zero_probs = probs[zero_index, 0] # 0的结果就是为假
    one_probs = probs[one_index, 0] # 1的结果为真

    threshold_index = torch.linspace(-0.1, 1.01, 10000)
    tpr = torch.zeros(len(threshold_index),)
    fpr = torch.zeros(len(threshold_index),)
    cnt = 0
    for i in threshold_index:
        tpr[cnt] = one_probs.le(i).sum().item()/len(one_probs)
        fpr[cnt] = zero_probs.le(i).sum().item()/len(zero_probs)
        cnt += 1

    sum_rate = tpr + fpr
    distance_to_one = torch.abs(sum_rate - 1)
    eer_index = distance_to_one.argmin(dim=0).item()
    out_eer = 0.5*(fpr[eer_index] + 1 - tpr[eer_index]).numpy()

    if show_plot:
        print('EER: {:.4f}%.'.format(out_eer * 100))
        plt.figure(1)
        plt.plot(torch.linspace(-0.2, 1.2, 1000), torch.histc(zero_probs, bins=1000, min=-0.2, max=1.2) / len(zero_probs))
        plt.plot(torch.linspace(-0.2, 1.2, 1000), torch.histc(one_probs, bins=1000, min=-0.2, max=1.2) / len(one_probs))
        plt.xlabel("Probability of 'Genuine'")
        plt.ylabel('Per Class Ratio')
        plt.legend(['Real', 'Fake'])
        plt.grid()

        plt.figure(3)
        plt.scatter(fpr, tpr)
        plt.xlabel('False Positive (Fake) Rate')
        plt.ylabel('True Positive (Fake) Rate')
        plt.grid()
        plt.show()

    return out_eer


def compute_eer_and_tdcf(cm_score_file, asv_score_file):

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float64)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_keys = cm_data[:, 4]
    cm_sco = np.genfromtxt('./score/cmdevscore.trl.txt', dtype=str)
    cm_scores = cm_sco[:].astype(np.float64)
    other_cm_scores = -cm_scores

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_keys == 'spoof'])[0]

    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    if eer_cm < other_eer_cm:
        # Compute t-DCF
        tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)

        # Minimum t-DCF
        min_tDCF_index = np.argmin(tDCF_curve)
        min_tDCF = tDCF_curve[min_tDCF_index]

    else:
        tDCF_curve, CM_thresholds = em.compute_tDCF(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_keys == 'spoof'],
                                                    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)

        # Minimum t-DCF
        min_tDCF_index = np.argmin(tDCF_curve)
        min_tDCF = tDCF_curve[min_tDCF_index]


    # print('ASV SYSTEM')
    # print('   EER            = {:8.5f} % (Equal error rate (target vs. nontarget discrimination)'.format(eer_asv * 100))
    # print('   Pfa            = {:8.5f} % (False acceptance rate of nontargets)'.format(Pfa_asv * 100))
    # print('   Pmiss          = {:8.5f} % (False rejection rate of targets)'.format(Pmiss_asv * 100))
    # print('   1-Pmiss,spoof  = {:8.5f} % (Spoof false acceptance rate)'.format((1 - Pmiss_spoof_asv) * 100))

    print('\nCM SYSTEM')
    print('   EER            = {:8.5f} % (Equal error rate for countermeasure)'.format(min(eer_cm, other_eer_cm) * 100))

    print('\nTANDEM')
    print('   min-tDCF       = {:8.5f}'.format(min_tDCF))


    # Visualize ASV scores and CM scores
    plt.figure()
    ax = plt.subplot(121)
    plt.hist(tar_asv, histtype='step', density=True, bins=50, label='Target')
    plt.hist(non_asv, histtype='step', density=True, bins=50, label='Nontarget')
    plt.hist(spoof_asv, histtype='step', density=True, bins=50, label='Spoof')
    plt.plot(asv_threshold, 0, 'o', markersize=10, mfc='none', mew=2, clip_on=False, label='EER threshold')
    plt.legend()
    plt.xlabel('ASV score')
    plt.ylabel('Density')
    plt.title('ASV score histogram')

    ax = plt.subplot(122)
    plt.hist(bona_cm, histtype='step', density=True, bins=50, label='Bona fide')
    plt.hist(spoof_cm, histtype='step', density=True, bins=50, label='Spoof')
    plt.legend()
    plt.xlabel('CM score')
    # plt.ylabel('Density')
    plt.title('CM score histogram')
    plt.savefig(cm_score_file[:-4]+'1.png')


    # Plot t-DCF as function of the CM threshold.
    plt.figure()
    plt.plot(CM_thresholds, tDCF_curve)
    plt.plot(CM_thresholds[min_tDCF_index], min_tDCF, 'o', markersize=10, mfc='none', mew=2)
    plt.xlabel('CM threshold index (operating point)')
    plt.ylabel('Norm t-DCF')
    plt.title('Normalized tandem t-DCF')
    plt.plot([np.min(CM_thresholds), np.max(CM_thresholds)], [1, 1], '--', color='black')
    plt.legend(('t-DCF', 'min t-DCF ({:.5f})'.format(min_tDCF), 'Arbitrarily bad CM (Norm t-DCF=1)'))
    plt.xlim([np.min(CM_thresholds), np.max(CM_thresholds)])
    plt.ylim([0, 1.5])
    plt.savefig(cm_score_file[:-4]+'2.png')

    plt.show()

    return min(eer_cm, other_eer_cm), min_tDCF

if __name__ == '__main__':

    test_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    protocol_file_path = '../LAPA/ASVspoof2019_LAPA_cm_protocols/ASVspoof2019.LAPA.cm.eval.trl.txt'
    asv_scores_file = '../LAPA/ASVspoof2019_LAPA_asv_scores/ASVspoof2019.LAPA.asv.eval.gi.trl.scores.txt'
    data_path = '../LAPA/data/eval_6.4_cqt/'
    path_to_database = '../'


    Net = models.SSDNet1D()
    # Net = models.DilatedNet()
    num_total_learnable_params = sum(i.numel() for i in Net.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))
    # path = [
    #         'LFCC_19_ASVspoof2019_LAPA_Loss_0.0275_dEER_4.39%_eEER_8.42%.pth',
    #         'LFCC_19_ASVspoof2019_LAPA_Loss_0.022_dEER_4.17%_eEER_9.3%.pth',
    #         'LFCC_20_ASVspoof2019_LAPA_Loss_0.0321_dEER_5.5%_eEER_8.49%.pth',
    #         'LFCC_19_ASVspoof2019_LAPA_Loss_0.028_dEER_4.15%_eEER_7.83%.pth',
    #         'LFCC_20_ASVspoof2019_LAPA_Loss_0.0233_dEER_4.12%_eEER_8.22%.pth',
    #         'LFCC_33_ASVspoof2019_LAPA_Loss_0.0065_dEER_4.33%_eEER_8.69%.pth',
    #         'LFCC_27_ASVspoof2019_LAPA_Loss_0.0113_dEER_4.29%_eEER_8.47%.pth',
    #         'LFCC_16_ASVspoof2019_LAPA_Loss_0.0349_dEER_4.26%_eEER_8.0%.pth',
    #         'LFCC_23_ASVspoof2019_LAPA_Loss_0.0204_dEER_3.97%_eEER_7.8%.pth']

    path = ['CQT_44_ASVspoof2019_LAPA_Loss_0.004_dEER_1.67%_eEER_5.67%.pth',
            'CQT_46_ASVspoof2019_LAPA_Loss_0.0039_dEER_1.64%_eEER_5.66%.pth',
            'CQT_49_ASVspoof2019_LAPA_Loss_0.002_dEER_1.42%_eEER_5.73%.pth',
            'CQT_45_ASVspoof2019_LAPA_Loss_0.0025_dEER_1.88%_eEER_6.17%.pth',
            'CQT_20_ASVspoof2019_LAPA_Loss_0.017_dEER_1.75%_eEER_5.07%.pth',
            'CQT_15_ASVspoof2019_LAPA_Loss_0.0278_dEER_1.75%_eEER_5.77%.pth',
            'CQT_40_ASVspoof2019_LAPA_Loss_0.0047_dEER_1.77%_eEER_5.94%.pth',
            'CQT_34_ASVspoof2019_LAPA_Loss_0.008_dEER_1.68%_eEER_5.44%.pth',
            'CQT_24_ASVspoof2019_LAPA_Loss_0.0142_dEER_1.81%_eEER_5.92%.pth',
            'CQT_34_ASVspoof2019_LAPA_Loss_0.0062_dEER_1.44%_eEER_5.38%.pth']

    # path = ['CQCC_22_ASVspoof2019_LAPA_Loss_0.094_dEER_15.89%_eEER_99.0%.pth',
    #         'CQCC_17_ASVspoof2019_LAPA_Loss_0.1456_dEER_16.16%_eEER_99.0%.pth',
    #         'CQCC_27_ASVspoof2019_LAPA_Loss_0.096_dEER_16.66%_eEER_99.0%.pth',
    #         'CQCC_18_ASVspoof2019_LAPA_Loss_0.1269_dEER_15.65%_eEER_99.0%.pth',
    #         'CQCC_12_ASVspoof2019_LAPA_Loss_0.2039_dEER_16.48%_eEER_99.0%.pth',
    #         'CQCC_24_ASVspoof2019_LAPA_Loss_0.0837_dEER_15.8%_eEER_99.0%.pth',
    #         'CQCC_22_ASVspoof2019_LAPA_Loss_0.1285_dEER_16.64%_eEER_99.0%.pth',
    #         'CQCC_18_ASVspoof2019_LAPA_Loss_0.1469_dEER_16.93%_eEER_99.0%.pth',
    #         'CQCC_26_ASVspoof2019_LAPA_Loss_0.0481_dEER_16.13%_eEER_99.0%.pth',
    #         'CQCC_18_ASVspoof2019_LAPA_Loss_0.1306_dEER_16.51%_eEER_99.0%.pth']
    for i in range(10):
        check_point = torch.load('./time_frame/'+path[i])
        Net.load_state_dict(check_point['model_state_dict'])
        accuracy, probabilities = asv_cal_accuracies(protocol_file_path, data_path, Net, test_device, data_type='time_frame', dataset=19)
        eer_cm, min_tDCF = compute_eer_and_tdcf(protocol_file_path, asv_scores_file)
        save_score = './score/score.txt'
        with open(save_score, "a") as fh:
            fh.write("{}\t{}\t{}\n".format(str(accuracy),str(eer_cm),str(min_tDCF)))  # [-3.22324123245] 去除[ ]然后转为str存储
            print("Scores saved to {}".format(save_score))
        print('accuracy',accuracy,'eer_cm:',eer_cm,'min_tDCF:',min_tDCF)
