import argparse
import time
from ast import arg
import os
import torch.nn.functional as F



import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
import torch.multiprocessing as mp
from scipy.signal import convolve2d
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score

from torch.utils.data import Dataset
from adapters import AutoAdapterController, MetaAdapterController, AdapterLayersHyperNetController, \
    AdapterLayersOneHyperNetController, TaskEmbeddingController
from adapters import MetaAdapterConfig
from models import get_model
import pickle
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage import gaussian_filter
import torch
from PIL import Image

SEED = 0


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def apply_filter(channel, srm_filter):
    channel_np = np.asnumpy(channel)
    return np.abs(convolve2d(channel_np, srm_filter, mode='same')).astype(np.uint8)


def apply_filter(channel, srm_filter):
    return np.abs(convolve2d(channel, srm_filter, mode='same')).astype(np.uint8)


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}
stat_from = "clip"
srm_filters = np.array([[
        # Filter 1
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]],
        # Filter 2
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, -1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 3
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 4
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0]],
        # Filter 5
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 6
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 7
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 1, -1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 8
        [[0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, -1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 9
        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, -2, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 10
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, -2, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 11
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 1, -2, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 12
        [[0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, -2, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0]],
        # Filter 13
        [[0, 0, -1, 0, 0],
         [0, 0, 3, 0, 0],
         [0, 0, -3, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 14
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 3, 0],
         [0, 0, -3, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 15
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 1, -3, 3, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 16
        [[0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, -3, 0, 0],
         [0, 0, 0, 3, 0],
         [0, 0, 0, 0, 0]],
        # Filter 17
        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, -3, 0, 0],
         [0, 0, 3, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 18
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, -3, 0, 0],
         [0, 3, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 19
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 3, -3, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 20
        [[0, 0, 0, 0, 0],
         [0, 3, 0, 0, 0],
         [0, 0, -3, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0]],
        # Filter 21
        [[0, 0, 0, 0, 0],
         [0, -1, 2, -1, 0],
         [0, 2, -4, 2, 0],
         [0, -1, 2, -1, 0],
         [0, 0, 0, 0, 0]],
        # Filter 22
        [[0, 0, 0, 0, 0],
         [0, -1, 2, -1, 0],
         [0, 2, -4, 2, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 23
        [[0, 0, 0, 0, 0],
         [0, 0, 2, -1, 0],
         [0, 0, -4, 2, 0],
         [0, 0, 2, -1, 0],
         [0, 0, 0, 0, 0]],
        # Filter 24
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 2, -4, 2, 0],
         [0, -1, 2, -1, 0],
         [0, 0, 0, 0, 0]],
        # Filter 25
        [[0, 0, 0, 0, 0],
         [0, -1, 2, 0, 0],
         [0, 2, -4, 0, 0],
         [0, -1, 2, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 26
        [[-1, 2, -2, 2, -1],
         [2, -6, 8, -6, 2],
         [-2, 8, -12, 8, -2],
         [2, -6, 8, -6, 2],
         [-1, 2, -2, 2, -1]],
        # Filter 27
        [[-1, 2, -2, 2, -1],
         [2, -6, 8, -6, 2],
         [-2, 8, -12, 8, -2],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Filter 28
        [[0, 0, -2, 2, -1],
         [0, 0, 8, -6, 2],
         [0, 0, -12, 8, -2],
         [0, 0, 8, -6, 2],
         [0, 0, -2, 2, -1]],
        # Filter 29
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [-2, 8, -12, 8, -2],
         [2, -6, 8, -6, 2],
         [-1, 2, -2, 2, -1]],
        # Filter 30
        [[-1, 2, -2, 0, 0],
         [2, -6, 8, 0, 0],
         [-2, 8, -12, 0, 0],
         [2, -6, 8, 0, 0],
         [-1, 2, -2, 0, 0]]
    ]
    )

def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]
    if y_pred[0:N // 2].max() <= y_pred[N // 2:N].min():  # perfectly separable case
        return (y_pred[0:N // 2].max() + y_pred[N // 2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)  # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

    return Image.fromarray(img)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc

def getlist(srm_filters_tensor,img,filtered_image_tensor,Orgimg,listAdd):
    for i in range(0, 8):
        srm_filter = srm_filters_tensor[i]
        for channel in range(img.shape[1]):
            filtered_channel = F.conv2d(img[:, channel:channel + 1, :, :], srm_filter.unsqueeze(1),
                                        padding=2)
            filtered_image_tensor[:, channel:channel + 1, :, :] += filtered_channel.abs()
    Orgimg = transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from])(
        Orgimg)
    filtered_image_tensor = transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from])(
        filtered_image_tensor)

    listAdd0 = (model(Orgimg, 5)).sigmoid().flatten().tolist()
    listAdd1 = (model(filtered_image_tensor, 0)).sigmoid().flatten().tolist()

    alpha = 0.8
    for i in range(len(listAdd1)):
        listAdd.append(
            (listAdd0[i] * alpha + listAdd1[i] * (1 - alpha)))
    for i in range(len(listAdd)):
        if listAdd[i] > 0.1 and listAdd[i] < 0.9:
            lower = 0
            bigger = 0
            if listAdd[i] < 0.5:
                lower = 1
            elif listAdd[i] > 0.5:
                bigger = 1
            for j in range(4):
                if j == 0:
                    for k in range(8, 12):
                        srm_filter = srm_filters_tensor[k]
                        filtered_image_tensor = torch.zeros_like(img)
                        for channel in range(img.shape[1]):
                            filtered_channel = F.conv2d(img[i:i + 1, channel:channel + 1, :, :],
                                                        srm_filter.unsqueeze(1),
                                                        padding=2)
                            filtered_image_tensor[i:i + 1, channel:channel + 1, :, :] += filtered_channel.abs()
                    filtered_image_tensor[i] = transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from])(
                        filtered_image_tensor[i])
                    single_image_batch = filtered_image_tensor[i:i + 1]

                    listAdd[i] = (model(single_image_batch, j + 1)).sigmoid().flatten().tolist()[0] * (1 - alpha) + \
                                 listAdd[i] * alpha
                elif j == 1:
                    for k in range(12, 20):
                        srm_filter = srm_filters_tensor[k]
                        filtered_image_tensor = torch.zeros_like(img)
                        for channel in range(img.shape[1]):
                            filtered_channel = F.conv2d(img[i:i + 1, channel:channel + 1, :, :],
                                                        srm_filter.unsqueeze(1),
                                                        padding=2)
                            filtered_image_tensor[i:i + 1, channel:channel + 1, :, :] += filtered_channel.abs()
                    filtered_image_tensor[i] = transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from])(
                        filtered_image_tensor[i])
                    single_image_batch = filtered_image_tensor[i:i + 1]
                    listAdd[i] = (model(single_image_batch, j + 1)).sigmoid().flatten().tolist()[0] * (1 - alpha) + \
                                 listAdd[i] * alpha
                elif j == 2:
                    for k in range(20, 25):
                        srm_filter = srm_filters_tensor[k]
                        filtered_image_tensor = torch.zeros_like(img)
                        for channel in range(img.shape[1]):
                            filtered_channel = F.conv2d(img[i:i + 1, channel:channel + 1, :, :],
                                                        srm_filter.unsqueeze(1),
                                                        padding=2)
                            filtered_image_tensor[i:i + 1, channel:channel + 1, :, :] += filtered_channel.abs()
                    filtered_image_tensor[i] = transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from])(
                        filtered_image_tensor[i])
                    single_image_batch = filtered_image_tensor[i:i + 1]

                    listAdd[i] = (model(single_image_batch, j + 1)).sigmoid().flatten().tolist()[0] * (1 - alpha) + \
                                 listAdd[i] * alpha
                elif j == 3:
                    for k in range(25, 30):
                        srm_filter = srm_filters_tensor[k]
                        filtered_image_tensor = torch.zeros_like(img)
                        for channel in range(img.shape[1]):
                            filtered_channel = F.conv2d(img[i:i + 1, channel:channel + 1, :, :],
                                                        srm_filter.unsqueeze(1),
                                                        padding=2)
                            filtered_image_tensor[i:i + 1, channel:channel + 1, :, :] += filtered_channel.abs()
                    filtered_image_tensor[i] = transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from])(
                        filtered_image_tensor[i])
                    single_image_batch = filtered_image_tensor[i:i + 1]

                    listAdd[i] = (model(single_image_batch, j + 1)).sigmoid().flatten().tolist()[0] * (1 - alpha) + \
                                 listAdd[i] * alpha
                if listAdd[i] < 0.1 and bigger == 1:
                    break
                elif listAdd[i] > 0.9 and lower == 1:
                    break
    return listAdd
def validate(model, loader, find_thres=False):
    batches = list(loader)


    srm_filters_tensor = torch.tensor(srm_filters, dtype=torch.float32).unsqueeze(1).to('cuda:0')
    with torch.no_grad():
        y_true, y_pred, z_pred = [], [], []
        for img, label, path, Orgimg in loader:
            device = torch.device("cuda:0")
            img = img.to(device)
            Orgimg = Orgimg.to(device)
            filtered_image_tensor = torch.zeros_like(img)
            listAdd = []
            listAdd=getlist(srm_filters_tensor,img,filtered_image_tensor,Orgimg,listAdd)
            y_pred.extend(listAdd)
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Get AP

    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)

    print(f"acc: {acc0}, ap: {ap}")


    if not find_thres:
        return ap, r_acc0, f_acc0, acc0

    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class RealFakeDataset(Dataset):
    def __init__(self, real_path,
                 fake_path,
                 data_mode,
                 max_sample,
                 arch,
                 jpeg_quality=None,
                 gaussian_sigma=None):

        assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        # = = = = = = data path = = = = = = = = = #
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l
        self.total_list = real_list + fake_list

        # = = = = = =  label = = = = = = = = = #

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

    def read_path(self, real_path, fake_path, data_mode, max_sample):

        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]

        assert len(real_list) == len(fake_list)

        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):

        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        Orgimg = self.transform(img)
        return Orgimg, label, img_path, Orgimg


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    parser.add_argument('--max_sample', type=int, default=1000,
                        help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str,
                        default='pretrained_weights/HyperDet.pth')
    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--jpeg_quality', type=int, default=None,
                        help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None,
                        help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")

    device = torch.device("cuda:0")
    opt = parser.parse_args()

    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    # model = get_model(opt.arch)
    opt.arch = "CLIP:ViT-L/14"
    adapter_config = MetaAdapterConfig()
    adapter_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    AdapterLayersOneHyperNetController = AdapterLayersOneHyperNetController(adapter_config)
    model = get_model("CLIP:ViT-L/14", AdapterLayersOneHyperNetController).to(torch.float32)

    state_dict = torch.load(opt.ckpt, map_location=device)



    model_state_dict = state_dict['model']
    num = 0

    new_state_dict = {}
    for key in model_state_dict.keys():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = model_state_dict[key]
    model.load_state_dict(new_state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print("Model loaded..")
    model.eval()

    model.to(device)
    print(opt)

    if (opt.real_path == None) or (opt.fake_path == None):
        print("hihihihi")
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [dict(real_path=opt.real_path, fake_path=opt.fake_path, data_mode='ours', key='lesgoo')]
        print(opt.real_path)
    for dataset_path in (dataset_paths):

        set_seed()

        dataset = RealFakeDataset(dataset_path['real_path'],
                                  dataset_path['fake_path'],
                                  dataset_path['data_mode'],
                                  opt.max_sample,
                                  opt.arch,
                                  jpeg_quality=opt.jpeg_quality,
                                  gaussian_sigma=opt.gaussian_sigma,
                                  )
        # print(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, find_thres=True)
        with open(os.path.join(opt.result_folder, 'ap.txt'), 'a') as f:
            f.write(dataset_path['real_path'] + ': ' + str(round(ap * 100, 2)) + '\n')

        with open(os.path.join(opt.result_folder, 'acc0.txt'), 'a') as f:
            f.write(dataset_path['fake_path'] + ': ' + str(round(r_acc0 * 100, 2)) + '  ' + str(
                round(f_acc0 * 100, 2)) + '  ' + str(round(acc0 * 100, 2)) + '\n')
        print("if use best_thres")
        print(f"acc: {acc1}")

