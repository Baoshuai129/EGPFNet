import time
import argparse

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils.utils import *
from model import Net
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gc
import pywt
import math
# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:5')
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument("--angRes_in", type=int, default=2, help="input angular resolution")
    parser.add_argument("--angRes_out", type=int, default=7, help="output angular resolution")
    parser.add_argument('--trainset_dir', type=str, default='../Data/TrainData_HCI_2x2-7x7/')
    parser.add_argument('--testset_dir', type=str, default='../Data/TestData_RE_HCI_2x2-7x7/')
    parser.add_argument('--model_name', type=str, default='EGPF_HCI_2x2-7x7')
    parser.add_argument('--log_dir', type=str, default='./logtxt/')

    #parser.add_argument('--trainset_dir', type=str, default='../Data/TrainData_Lytro_2x2-7x7/')
    #parser.add_argument('--testset_dir', type=str, default='../Data/TestData_RE_Lytro_2x2-7x7/')
    #parser.add_argument('--model_name', type=str, default='EGPF_Lytro_2x2-7x7')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=70, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches for validation")
    parser.add_argument("--patchsize", type=int, default=128, help="LFs are cropped into patches for validation")
    parser.add_argument("--stride", type=int, default=64, help="LFs are cropped into patches for validation")
    
    parser.add_argument('--load_pretrain', type=bool, default=False)        #加载预训练模型
    parser.add_argument('--model_path', type=str, default='./log/EGPF_HCI_2x2-7x7.pth.tar')

    return parser.parse_args()


def train(cfg, train_loader, test_Names, test_loaders):
    if cfg.parallel:
        cfg.device = 'cuda:5'

    net = Net(cfg.angRes_in, cfg.angRes_out)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0


    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:5': cfg.device})
            net.load_state_dict(model['state_dict'])        #预训练的参数权重加载
            epoch_state = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    if cfg.parallel:
        net = torch.nn.DataParallel(net, device_ids=[5])        #GPU并行处理

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)       #优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)      #每隔setp_size 将学习率乘以gamma替换
    scheduler._step_count = epoch_state     #参数更新
    loss_epoch = []
    loss_list = []
    i = 0
    k = 0
    for idx_epoch in range(epoch_state, cfg.n_epochs):
        for idx_iter, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):      #进度条
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            out = net(data)             #前向传播计算预测值

            epi_loss1 = epi_loss(out, label)
            out_numpy = out.cpu().detach().numpy()
            label_numpy = label.cpu().detach().numpy()
            coeffs_out = pywt.wavedec2(out_numpy[1][0], 'haar', level=1)
            coeffs_label = pywt.wavedec2(label_numpy[1][0], 'haar', level=1)
            # 显示小波系数的图像表示
            LL, (LH, HL, HH) = coeffs_out
            LL_label, (LH_label, HL_label, HH_label) = coeffs_label
            loss_ll = criterion_Loss(torch.from_numpy(LL), torch.from_numpy(LL_label))
            loss_hl = criterion_Loss(torch.from_numpy(HL), torch.from_numpy(HL_label))
            loss = criterion_Loss(out, label) + 0.4*loss_ll +  0.4*loss_hl    #计算损失

            optimizer.zero_grad()       #将模型的参数梯度初始化为0
            loss.backward()             #反向传播计算梯度
            optimizer.step()            # 更新所有参数
            loss_epoch.append(loss.data.cpu())

            i = i+1
            writer.add_scalar('loss11', float(loss), i)

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            txtfile = open('outputlog1.txt', 'a')
            txtfile.write('\n' + time.ctime()[4:-5] +'\n Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            txtfile.close()

            if cfg.parallel:
                save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict(),
            }, save_path='./log/', filename=cfg.model_name + '.pth.tar')
            else:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                }, save_path='./log/', filename=cfg.model_name + '.pth.tar')

            loss_epoch = []

        ''' evaluation '''
        with torch.no_grad():
            psnr_testset = []
            ssim_testset = []

            for index, test_name in enumerate(test_Names):
                test_loader = test_loaders[index]
                psnr_epoch_test, ssim_epoch_test = valid(test_loader, net)
                psnr_testset.append(psnr_epoch_test)
                ssim_testset.append(ssim_epoch_test)
                print(time.ctime()[4:-5] + ' Dataset----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
                txtfile = open('outputlog1.txt', 'a')
                txtfile.write(  '\n Dataset----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
                txtfile.close()
                k = k + 1
                writer.add_scalar('psnr', float(psnr_epoch_test), k)
                writer.add_scalar('ssim', float(ssim_epoch_test), k)

                pass
            pass
        gc.collect()
        torch.cuda.empty_cache()
        scheduler.step()
        pass


def valid(test_loader, net):
    psnr_iter_test = []
    ssim_iter_test = []

    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze().to(cfg.device)


        if cfg.crop == False:
            with torch.no_grad():
                outLF = net(data.unsqueeze(0).unsqueeze(0).to(cfg.device))
                outLF = outLF.squeeze()
        else:
            uh, vw = data.shape
            h0, w0 = uh // cfg.angRes_in, vw // cfg.angRes_in
            subLFin = LFdivide(data, cfg.angRes_in, cfg.patchsize, cfg.patchsize // 2)  # numU, numV, h*angRes, w*angRes
            numU, numV, H, W = subLFin.shape
            subLFout = torch.zeros(numU, numV, cfg.angRes_out * cfg.patchsize, cfg.angRes_out * cfg.patchsize)

            for u in range(numU):
                for v in range(numV):
                    tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        out = net(tmp.to(cfg.device))
                        subLFout[u, v, :, :] = out.squeeze()
            outLF = LFintegrate(subLFout, cfg.angRes_out, cfg.patchsize, cfg.stride, h0, w0)

        psnr, ssim = cal_metrics_RE(label, outLF, cfg.angRes_in, cfg.angRes_out)


        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def epi_loss(pred, label):
    # epi loss
    epi_h_pred, epi_v_pred = lf2epi(pred)
    dx_h_pred, dy_h_pred = gradient(epi_h_pred)
    dx_v_pred, dy_v_pred = gradient(epi_v_pred)

    epi_h_label, epi_v_label = lf2epi(label)
    dx_h_label, dy_h_label = gradient(epi_h_label)
    dx_v_label, dy_v_label = gradient(epi_v_label)

    return (reconstruction_loss(dx_h_pred, dx_h_label) + reconstruction_loss(dy_h_pred, dy_h_label)
            + reconstruction_loss(dx_v_pred, dx_v_label) + reconstruction_loss(dy_v_pred, dy_v_label))


def lf2epi(lf):
    N, an2, h, w = lf.shape
    an = int(math.sqrt(an2))
    epi_h = lf.view(N, an, an, h, w).permute(0, 1, 3, 2, 4).contiguous().view(-1, 1, an, w)
    epi_v = lf.view(N, an, an, h, w).permute(0, 2, 4, 1, 3).contiguous().view(-1, 1, an, h)
    return epi_h, epi_v


def reconstruction_loss(X,Y):
# L1 Charbonnier loss
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt( diff * diff + eps )
    loss = torch.sum(error) / torch.numel(error)
    return loss

def gradient(pred):
    D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy

def main(cfg):
    seed = 10000

    # 设置 Python 的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，也需要设置这个
    torch.backends.cudnn.deterministic = True  # 如果使用了CUDA，这个也需要设置
    torch.backends.cudnn.benchmark = False

    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=cfg.batch_size, shuffle=True)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    train(cfg, train_loader, test_Names, test_Loaders)


if __name__ == '__main__':
    writer: SummaryWriter = SummaryWriter()
    print("1111")
    cfg = parse_args()
    main(cfg)
    writer.close()


