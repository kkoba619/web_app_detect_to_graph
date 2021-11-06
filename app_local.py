# 参考: https://www.python.ambitious-engineer.com/archives/1630
# 参考: https://note.com/kamakiriphysics/n/n2aec5611af2a
# 参考:  https://qiita.com/Gen6/items/2979b84797c702c858b1

import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, g, flash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import glob
import shutil
import argparse
import pathlib

import numpy as np
from numpy import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly
import plotly.express as px
import plotly.offline as offline 

from PIL import Image
import cv2
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True,force_reload=True)
import torchvision

# https://stackoverflow.com/questions/68140388/an-error-cache-may-be-out-of-date-try-force-reload-true-comes-up-even-thou
import torch.backends.cudnn as cudnn

from pathlib import Path

# graphファイル削除用
def remove_glob(pathname, recursive=True):
    for p in glob.glob(pathname, recursive=recursive):
        if os.path.isfile(p):
            os.remove(p)


# mp4から画像を抽出
def save_frame_sec(video_path, sec, result_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.set(cv2.CAP_PROP_POS_FRAMES, round(fps * sec))

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(result_path, frame)


# 物体検出
def dtc_grph_label(img_ad,img_dtct,dtct_lbl,i):
    img = [img_ad]
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path='static/yolov5s.pt',force_reload=True)
    results = model(img)

    # plotデータの整理
    detect = results.pandas().xyxy[0]
    detect['x'] = (detect.xmin + detect.xmax)/2
    detect['y'] = (detect.ymin + detect.ymax)/2
    detect['size'] =  np.sqrt((detect.xmax - detect.xmin)*(detect.ymax - detect.ymin))
    detect['frame'] = i

    #グラフ作成
    fig = plt.figure(figsize=(8, 8))
    # fig = plt.figure()

    sns.scatterplot(data=detect, x='x', y='y', hue='name',size = detect['size']*100,alpha = 0.5,sizes=(100,500))
    plt.xlim(0,np.array(img).shape[2])
    plt.ylim(np.array(img).shape[1],0)

    #画像の読み込み https://qiita.com/zaburo/items/5637b424c655b136527a
    im = Image.open(img_ad)

    #画像をarrayに変換
    im_list = np.asarray(im)
    #貼り付け
    plt.imshow(im_list, alpha=1.0)
    #表示
    plt.axis("off") #https://qiita.com/tsukada_cs/items/8d31a25cd7c860690270
    plt.imshow(im, alpha=0.6)

    if np.array(img).shape[2] > np.array(img).shape[1]:
        plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0, fontsize=8)
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)

    plt.savefig(img_dtct+'/'+img_ad.split('.')[-2].split('/')[-1]+'_detect.png')
    detect.to_csv(dtct_lbl+'/'+img_ad.split('.')[-2].split('/')[-1]+'_label.csv')


app = Flask(__name__)

# ファイル容量を制限する
# https://tanuhack.com/flask-client2server/
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  #5MB




SAVE_DIR = "graph"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)


@app.route('/graph/<path:filepath>')
def send_js(filepath):
    return send_from_directory(SAVE_DIR, filepath)


@app.route("/", methods=["GET","POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")

    if request.method == "POST":

        image = request.files['image']  
        if image:

            remove_glob('./upload/**')
            app.logger.info('file_name={}'.format(image.filename))
            app.logger.info('content_type={} content_length={}, mimetype={}, mimetype_params={}'.format(
                image.content_type, image.content_length, image.mimetype, image.mimetype_params))
            #imagefile_en = image.filename.encode('utf-8')
            image.save("./upload/"+image.filename)
            
            video_path = "./upload/"+image.filename
            video_2_jpg_path = './images/frame'
            img_dtct = './images/detect'
            dtct_lbl = './images/labels'
            
            remove_glob(video_2_jpg_path+'/**')
            remove_glob(img_dtct+'/**')

            # ファイルの情報抽出
            cap = cv2.VideoCapture(video_path)
            video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            video_fps = cap.get(cv2.CAP_PROP_FPS) 
            video_len_sec = video_frame_count / video_fps
            print('sec:',video_len_sec) 
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            print('width:',width)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print('height:',height)


            # 処理開始前に不要データ削除
            remove_glob(video_2_jpg_path+'/**')
            remove_glob(img_dtct+'/**')
            remove_glob(dtct_lbl+'/**')

            # framem→jpg→png/csv
            stp = 0.5 #stp[sec]に一枚画像取得
            nomax ='{0:04d}'.format(int(len(np.arange(0,video_len_sec//1+stp,stp)))-1)
            for i,sec in enumerate(np.arange(0,video_len_sec//1+stp,stp)): #再生時間(秒)切り上げ(c//1+1で切り上げ)
                no = '{0:04d}'.format(i)
                save_frame_sec(video_path, sec, video_2_jpg_path+'/'+no+'.jpg')
                dtc_grph_label(video_2_jpg_path+'/'+no+'.jpg',img_dtct,dtct_lbl,i)
                print(no,'/',nomax)
                remove_glob(video_2_jpg_path+'/**')
                
            # gifの元情報pngファイル確認
            files = sorted(glob.glob(img_dtct+'/*.png'))
            images = list(map(lambda file: Image.open(file), files))

            # 古いgifファイル削除
            remove_glob('./graph/**')

            #gifファイル作成
            filepath = "./graph/" + datetime.now().strftime("%Y%m%d%H%M%S_") + "out.gif"
            print(filepath)
            images[0].save(filepath, save_all=True, append_images=images[1:], duration=400, loop=0)


            # labelファイル抽出・統合
            df = pd.DataFrame()

            for file_path in pathlib.Path(dtct_lbl).glob('*.csv'):
                f_path = pathlib.Path(file_path)
                file_name = f_path.name
                df_tmp = pd.read_csv(dtct_lbl+'/'+file_name)
                df = pd.concat([df, df_tmp], axis=0)

            # plotlyでグラフ作成
            #fig = px.scatter(df, x="x", y="y",size = "size",size_max =30,color = 'name',animation_frame="frame",range_x=[0,width], range_y=[0,height])

            fig = px.scatter(df.sort_values(by = 'frame'), x="x", y="y",size = "size",size_max =30,color = 'name',animation_frame="frame")
            fig.update_xaxes(
                range=[0,width],  # sets the range of xaxis
            )
            fig.update_yaxes(
                range=[0,height],  # sets the range of xaxis
                scaleanchor = "x",
                scaleratio = 1
            )
            fig.update_yaxes(autorange="reversed")
            filepath_pltly = "./graph/" + datetime.now().strftime("%Y%m%d%H%M%S_") + "out.html"
            offline.plot(fig,filename=filepath_pltly,auto_open=False)

            return render_template("index.html", image = image ,filepath=filepath,filepath_pltly=filepath_pltly)

        else: # エラー処理
            return render_template("index.html", err_message_1="ファイルを選択してください！")
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True,  host='192.168.2.138', port=5000) # ポートの変更
    # port = int(os.environ.get("PORT", 5000))
    # app.run(debug=True, host="0.0.0.0", port=port)