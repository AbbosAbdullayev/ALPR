import cv2
import argparse
import os
import platform
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from utils import AttnLabelConverter
from model import Model

from dataset import NormalizePAD
from PIL import Image,ImageDraw,ImageFont
import math
import os
import sys
from time import time

from implements import *
import torch

      
global thresh
thresh = 0.5
        #image=self.image

        # Image convert to Tensor
def ConvertToTensor(s_size, src):
            imgH = s_size[0]
            imgW = s_size[1]

            input_channel = 3 if src.mode == 'RGB' else 1

            transform= NormalizePAD((input_channel, imgH, imgW))

            w, h = src.size
            ratio= w / float(h)

            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = math.ceil(imgH * ratio)

            # tensor reshape
            resized_image = src.resize((resized_w, imgH), Image.BICUBIC)

            # img to Tensor
            tmp           = transform(resized_image)

            img_tensor    = torch.cat([tmp.unsqueeze(0)], 0)
            # rgb tensor convert to grayscale tensor
            img_tensor    = rgb_to_grayscale(img_tensor)

            return img_tensor

        # OCR Recognition
def Recognition(img):
            #img=self.img
            # static w,h
           # print(type(img))
            imgH=32
            imgW=100
            s_size = [imgH,imgW]

            # result Text
            text   = []

            if img:
                # convert image to tensor
                #print('going.....')
                src= ConvertToTensor(s_size, img)
                #device= device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                batch_size = src.size(0)
                image= src.to(device)

                # For max length prediction
                length_for_pred   = torch.IntTensor([25] * batch_size).to(device)
                text_for_pred     = torch.LongTensor(batch_size, 26).fill_(0).to(device)

                preds= model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index    = preds.max(2)
                preds_str         = converter.decode(preds_index, length_for_pred)

                preds_prob        = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)

                pred_EOS          = preds_str[0].find('[s]')

                # prune after "end of sentence" token ([s])
                pred = preds_str[0][:pred_EOS]
                #print('prediction=',pred)
                pred_max_prob= preds_max_prob[0][:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                # confidence score type is tensor
                confidence_score  = pred_max_prob.cumprod(dim=0)[-1]

                if confidence_score >= thresh:
                    text.append(pred)
                else:
                    text.append('Missing OCR')
                
                text.append(confidence_score) # text = [predict text, confidence score]

            else:
                text.append('Missing Plate') # text  = [Missing Plate] 

            return text


        # Get Coordinate of Number Plate
def GetCoordinate(cor):
            #cor=self.cor
            pts  = []
            x_coor   = cor[0][0]
            y_coor   = cor[0][1]

            for i in range(4):
                pts.append([int(x_coor[i]),int(y_coor[i])])

            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1,1,2))
            return pts
def load_model(path):
            #path=self.path
            try:
                path = os.path.splitext(path)[0]
                with open('%s.json' % path, 'r') as json_file:
                    model_json = json_file.read()
                model = model_from_json(model_json, custom_objects={})
                model.load_weights('%s.h5' % path)
                print("Loading model successfully...")
                return model
            except Exception as e:
                print(e)
def get_plate(img, Dmax=608, Dmin = 256):
            #vehicle = preprocess_image(image_path)
            #img=self.img
            ratio = float(max(img.shape[:2])) / min(img.shape[:2])
            side = int(ratio * Dmin)
            bound_dim = min(side, Dmax)
            _ , LpImg, _, cor = detect_lp(wpod_net,img, bound_dim, lp_threshold=0.5)
            return LpImg, cor 
def Detect_plate(img):
                #img=self.img
                Lpimg=None
                dst=None
                cor=None
                try:
                    Lpimg,cor=get_plate(img)
                    print('LP detected')
                except Exception as e:
                    print('LP detection failed')
                    print(e)
                if Lpimg:
                    dst=Image.fromarray((Lpimg[0]*255).astype(np.uint8))
                return dst,cor            
        
'''gpu configure''' 
wpod_net_path = "weights/wpod-net.json"
wpod_net = load_model(wpod_net_path)
cudnn.benchmark = True
cudnn.deterministic = True
#num_gpu = torch.cuda.device_count()
device= device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('going')
character='0123456789abcdefghijklmnopqrstuvwxyz가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자배하허호국합육해공울산대인천광전울산경기강원충북남제'
'''model configuration '''               
converter = AttnLabelConverter(character)
#print('going--')
     # opt= CreateParser()
num_class = len(converter.character)  
#print('number of classes=',num_class)
model=Model().to(device)
# print(model)
# print('going+++')
model=torch.nn.DataParallel(model).to(device)
# print('going')

model.load_state_dict(torch.load('weights/v1.6-best_accuracy.pth',map_location=device))
model.eval()
cap = cv2.VideoCapture('video.mp4')
save_path='output.mp4'
output = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 18, (1280, 720))
frameRate  = int(cap.get(cv2.CAP_PROP_FPS))
with torch.no_grad():
    while True :                 
       t = time()
       retval,frame=cap.read()
       if not retval:
        break
       img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
       img1 = img / 255
       img,cor=Detect_plate(img1)             
       if img:      
            img = img.resize((435,100)) 
                #print('here I am')        
            t1= time()
            result = Recognition(img)
            print('OCR Recognition Time : ', time() - t1)
            print('Total Process Time : ',time() - t)
                #print('result==',result[0])
            dst=Image.fromarray((img1*255).astype(np.uint8))      
            font=ImageFont.truetype('fonts/gulim.ttc',size=30)
            draw=ImageDraw.Draw(dst)
            draw.text((30,280),result[0],(255,0,0),font=font)  
            dst=np.array(dst)
            dst= cv2.cvtColor(dst,cv2.COLOR_RGB2BGR)
            if cor:
                pts=GetCoordinate(cor)
                   # print('cor=',pts)
                   # print('coordinates=',cor[0])
                    #pts=([255,233],[440,334])
                cv2.polylines(dst,[pts],True,color=(255,0,0),thickness=2) 
                # dst=dst[:,:,::-1]
                del pts
            del cor,draw,font
            output.write(dst)
            cv2.imshow('frame',dst)
            if cv2.waitKey(frameRate)==27:
                break
    cv2.destroyAllWindows()
    output.release()
    cap.release()      