import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# [重要] 匯入我們修改過的模型
# 請確保您的模型檔案名稱為 model_dsc.py，且裡面有 Network 類別
try:
    from model_dsc import Network
except ImportError:
    # 如果您沒改檔名，還是叫 model.py，則嘗試從 model 匯入
    from model import Network

from multi_read_data import MemoryFriendlyLoader

# --- 參數設定 ---
parser = argparse.ArgumentParser("SDCE")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id (Colab usually 0)')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=200, help='epochs') # 建議 200 或 400
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--save', type=str, default='EXP/', help='location of the data corpus')

# [新增] 核心功能：選擇 Normalization Mode
parser.add_argument('--mode', type=str, default='original', 
                    choices=['original', 'tanh', 'softsign'], 
                    help='Normalization function: original (sigmoid), tanh, or softsign')

args = parser.parse_args()

# --- 環境與路徑設定 ---
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# [修改] 自動產生實驗名稱，不用手動輸入 input()
# 資料夾名稱會變成像: EXP/Train-softsign-20231215-120000
experiment_name = f"Train-{args.mode}"
args.save = os.path.join(args.save, f"{experiment_name}:{time.strftime('%Y%m%d-%H%M%S')}")

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = os.path.join(args.save, 'model_epochs/')
image_path = os.path.join(args.save, 'image_epochs/')
os.makedirs(model_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)

# --- Logging 設定 ---
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor, path):
    """將 Tensor 轉為圖片並儲存"""
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    # 確保數值在 0-255 之間
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'jpeg')

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    # --- [關鍵修改] 初始化模型時傳入 mode ---
    logging.info(f"Initializing Model with mode: {args.mode}")
    try:
        model = Network(mode=args.mode)
    except TypeError:
        logging.error("Error: 您的 Network 類別似乎不支援 'mode' 參數。請檢查 model_dsc.py 是否已修改。")
        sys.exit(1)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)

    # 計算模型參數大小
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f MB", MB)
    
    # --- 資料讀取設定 (已修改為相對路徑) ---
    # 請確保資料夾結構是 ./datasets/UIEB-S/train-sny600
    train_low_data_names = './datasets/UIEB-S/train-sny600'
    
    if not os.path.exists(train_low_data_names):
        logging.error(f"找不到訓練資料集: {train_low_data_names}")
        logging.error("請確認 datasets 資料夾是否已解壓縮在正確位置。")
        sys.exit(1)

    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')

    # 測試集 (Validation)
    test_low_data_names = './datasets/UIEB-S/valid' 
    # 如果 valid 資料夾不存在，這行可能會報錯，請視情況調整
    TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task='test')

    # [移除] 作者原本寫死的路徑 valid_low_data_names，避免報錯
    # validDataset = MemoryFriendlyLoader(...) 

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=True, generator=torch.Generator(device = 'cuda'))

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=False, generator=torch.Generator(device = 'cuda'))

    total_step = 0
    writer = SummaryWriter(log_dir=args.save, flush_secs=30)
    
    logging.info("Start Training...")

    for epoch in range(args.epochs):
        model.train()
        losses = []
        Mylosses = []
        SSIMlosses = []
        Gradientlosses= []

        for batch_idx, (input, label, _) in enumerate(train_queue):
            total_step += 1
            input = Variable(input, requires_grad=False).cuda()
            label = Variable(label, requires_grad=False).cuda()

            optimizer.zero_grad()
            
            # 計算 Loss
            loss, My_Loss, SSIM_Loss, gradientloss = model._loss(input, label)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            losses.append(loss.item())
            Mylosses.append(My_Loss.item())
            SSIMlosses.append(SSIM_Loss.item())
            Gradientlosses.append(gradientloss.item())

            # 減少 log 頻率，每 50 個 batch 印一次，保持版面乾淨
            if batch_idx % 50 == 0:
                logging.info('epoch %03d | step %03d | Loss: %.4f | My: %.4f | SSIM: %.4f | Grad: %.4f', 
                             epoch, batch_idx, loss, My_Loss, SSIM_Loss, gradientloss)

        logging.info('--- End of Epoch %03d | Avg Loss: %f ---', epoch, np.average(losses))
        
        # 儲存權重
        utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))
        
        # 寫入 Tensorboard
        writer.add_scalar('Total Loss', np.average(losses), epoch)
        writer.add_scalar('My Loss', np.average(Mylosses), epoch)
        writer.add_scalar('SSIM Loss', np.average(SSIMlosses), epoch)
        
        # --- 驗證與存圖 (Validation) ---
        # 每 5 個 epoch 存一次測試圖
        if epoch % 5 == 0:
            model.eval() # 切換到評估模式 (有些 layer 行為不同)
            logging.info("Saving test images...")
            with torch.no_grad():
                for i, (input, image_name) in enumerate(test_queue):
                    input = Variable(input).cuda()
                    
                    # 處理檔名
                    if isinstance(image_name, list) or isinstance(image_name, tuple):
                        img_name_str = image_name[0]
                    else:
                        img_name_str = image_name
                    
                    base_name = os.path.basename(img_name_str).split('.')[0]
                    
                    # 推論
                    enh = model(input)
                    
                    # 存檔 (格式: 原名_epoch.jpg)
                    u_name = '%s_ep%d.jpg' % (base_name, epoch)
                    u_path = os.path.join(image_path, u_name)
                    save_images(enh, u_path)
            
            model.train() # 切換回訓練模式

if __name__ == '__main__':
    main()