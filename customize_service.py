# -*- coding: utf-8 -*-
from collections import OrderedDict
from hr.seg_hrnet import hrnet18
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from model_service.pytorch_model_service import PTServingBaseService
import cv2
import time
# from metric.metrics_manager import MetricsManager
import log
from io import BytesIO
import base64
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
logger = log.getLogger(__name__)

class ImageClassificationService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.pred_windows = 1024
        self.stride = 512
        self.model = hrnet18(pretrained=False)
        self.use_cuda = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print('Using GPU for inference')
            self.use_cuda = True
            checkpoint = torch.load(self.model_path, map_location="cuda:0")
            self.model = self.model.to(device)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            print('Using CPU for inference')
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])

        self.model.eval()
    def ms_inference(self,model, image, flip=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        size = image.size()
        pred = model(image)

        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=True
        )
        pred = F.softmax(pred, dim=1)
        if flip:
            flip_img = image.cpu().numpy()[:, :,::-1, :]
            flip_output = model(torch.from_numpy(flip_img.copy()).to(device))

            flip_output = F.interpolate(
                input=flip_output, size=size[-2:],
                mode='bilinear', align_corners=True
            )
            flip_output = F.softmax(flip_output, dim=1)
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(
                flip_pred[:, :, ::-1 ,: ].copy()).to(device)
            pred += flip_pred
            pred = pred * 0.5
        return pred  # .exp()

    def multi_scale_aug(self,image,
                        rand_scale=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        long_size = np.int(1024 * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)

        return image

    def multi_scale_inference(self,model, image, scales=[0.75, 1., 1.25], flip=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.cpu().numpy()[0].transpose((1, 2, 0)).copy()  # hwc
        stride_h = np.int(1024 * 1.0)
        stride_w = np.int(1024 * 1.0)
        final_pred = torch.zeros([1, 2,
                                  ori_height, ori_width]).to(device)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image, rand_scale=scale)
            height, width = new_img.shape[:-1]

            if scale <= 2.0:
                if scale==1 or scale==0.75 or scale==1.25:
                    flip=True
                else:
                    flip=False
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img).to(device)
                preds = self.ms_inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             1024) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             1024) / stride_w)) + 1
                preds = torch.zeros([1, 2,
                                     new_h, new_w]).to(device)
                count = torch.zeros([1, 1, new_h, new_w]).to(device)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + 1024, new_h)
                        w1 = min(w0 + 1024, new_w)
                        h0 = max(int(h1 - 1024), 0)
                        w0 = max(int(w1 - 1024), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img).to(device)
                        pred = self.ms_inference(model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=True
            )
            final_pred += preds
        return final_pred/len(scales)
    # read img
    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                # img = self.transforms(img)
                img = np.array(img)
                preprocessed_data[k] = img 
        return preprocessed_data
    # peng zhang yu ce
    def _inference(self, data):
        img = data["input_img"]
        data = img
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.astype(np.float32)
        mean = (0.355403, 0.383969, 0.359276)
        std = (0.206617, 0.202157, 0.210082)
        data /= 255
        data -= mean
        data /= std
        # padding  
        src_h, src_w = data.shape[0], data.shape[1]
        # pad_top = (self.pred_windows - self.stride) // 2
        pad_bottom = 4096-src_h
        # pad_left = (self.pred_windows - self.stride) // 2
        # pad_right = (self.stride - src_w % self.stride) + (
        #             self.pred_windows - self.stride) // 2 if src_w % self.stride else (self.pred_windows - self.stride) // 2
        data_pad = cv2.copyMakeBorder(data, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT,
                                     value=(0.0, 0.0, 0.0))
        data = data_pad.transpose(2, 0, 1) # c,h,w
        c, h, w = data.shape
        label = np.zeros((src_h, src_w))
        h_num = (h-self.pred_windows)//self.stride + 1
        w_num = (w-self.pred_windows)//self.stride + 1
        with torch.no_grad():
            for i in range(h_num):
                for j in range(w_num):
                    h_s, h_e = i * self.stride, i * self.stride + self.pred_windows
                    w_s, w_e = j * self.stride, j * self.stride + self.pred_windows
                    img = data[:, h_s:h_e, w_s:w_e]
                    img = img[np.newaxis, :, :, :].astype(np.float32)
                    img = torch.from_numpy(img)
                    img = img.to(device)
                    # out_l = self.model(img)
                    out_l = self.multi_scale_inference(self.model, img, scales=[0.5,0.75,1,1.25,1.5])
                    # out_l = torch.softmax(out_l, dim=1)
                    out_l = out_l.cpu().data.numpy()
                    # out_l[out_l[0,0,:,:]>=0.4] = 0
                    pred = np.ones([out_l.shape[2], out_l.shape[3]], dtype=np.uint8)
                    pred[out_l[0, 0, :, :] >= 0.3] = 0
                    out_l = pred
                    label_h_s, label_h_e = (self.pred_windows - self.stride) // 2 + i * self.stride, min(
                        (i + 1) * self.stride + (self.pred_windows - self.stride) // 2, src_h)
                    label_w_s, label_w_e = (self.pred_windows - self.stride) // 2 + j * self.stride, min(
                        (j + 1) * self.stride + (self.pred_windows - self.stride) // 2, src_w)
                    out_h_s, out_h_e = (self.pred_windows - self.stride) // 2, (self.pred_windows - self.stride) // 2 + (
                            label_h_e - label_h_s)
                    out_w_s, out_w_e = (self.pred_windows - self.stride) // 2, (self.pred_windows - self.stride) // 2 + (
                            label_w_e - label_w_s)
                    if i == 0:
                        label_h_s = 0
                        label_h_e = self.stride + (self.pred_windows - self.stride) // 2
                        out_h_s = 0
                        out_h_e = self.stride + (self.pred_windows - self.stride) // 2
                    if j == 0:
                        label_w_s = 0
                        label_w_e = self.stride + (self.pred_windows - self.stride) // 2
                        out_w_s = 0
                        out_w_e = self.stride + (self.pred_windows - self.stride) // 2
                    if i == h_num - 1:
                        label_h_e = src_h
                        out_h_e = (self.pred_windows - self.stride) // 2 + (
                                label_h_e - label_h_s)
                    if j == w_num - 1:
                        label_w_e = src_w
                        out_w_e = (self.pred_windows - self.stride) // 2 + (
                                label_w_e - label_w_s)

                    label[label_h_s:label_h_e, label_w_s:label_w_e] = out_l[out_h_s: out_h_e,
                                                                      out_w_s: out_w_e].astype(
                        np.int8)
        # _label = label.astype(np.int8).tolist()
        _label = label.astype(np.int8).tolist()
        _len, __len = len(_label), len(_label[0])
        o_stack = []
        for _ in _label:
            out_s = {"s":[], "e":[]}
            j = 0
            while j < __len:
                if _[j] == 0:
                    out_s["s"].append(str(j))
                    while j < __len and _[j] == 0: j += 1
                    out_s["e"].append(str(j))
                j += 1
            o_stack.append(out_s)
        result = {"result": o_stack}
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')
        # if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
        #     MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)
        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000
        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)
        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        # if self.model_name + '_LatencyInference' in MetricsManager.metrics:
        #     MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)
        # Update overall latency metric
        # if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
        #     MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)
        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
        return data