import sys
import re
import requests
import json
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import threading

from tkinter import *
from socket import *
from time import ctime
from pylab import mpl
from pandas.io.formats import console, format as fmt
from pandas import get_option
from pandas.io.formats.printing import pprint_thing
from scipy.stats import norm, skew
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
PORT = [11111,22222,33333,44444]
count=range(len(PORT))
model_dir='tf/model/'
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
cloud1=[]
cloud2=[]
cloud3=[]
cloud4=[]

class DataAnalyze:
    def __init__(self,file,port):
        self.file=file
        self.df=pd.read_csv(file)
        self.port=port
    #用热力图来描述数值型特征之间的相关性
    def correlation(self):
        df = self.df
        corrmat = df.corr()#两两之间关系的数值表示。值越高，关系越密切
        sns.heatmap(df.corr(), cmap="YlGnBu")
        plt.title('数值型数据关系热力图')
        plt.legend()
        if self.port==11111:
         plt.savefig("heatmap1.jpg")
        elif self.port==22222:
         plt.savefig("heatmap2.jpg")
        elif self.port==33333:
         plt.savefig("heatmap3.jpg")
        elif self.port==44444:
         plt.savefig("heatmap4.jpg")
    #用对角线图描述数值型特征的之间的分布（网格图中包含柱状分布图，散点分布图和密度图）
    def dig_dist(self):
        df = self.df
        g = sns.PairGrid(df)  # 主对角线是数据集中每一个数值型数据列的直方图，
        g.map_diag(sns.distplot)  # 指定对角线绘图类型
        g.map_upper(plt.scatter, edgecolor="white")  # 两两关系分布
        g.map_lower(sns.kdeplot)  # 绘制核密度分布图。
        plt.title('数值型数据分布图')
        plt.legend()
        if self.port==11111:
         plt.savefig("distplot1.jpg")
        elif self.port==22222:
         plt.savefig("distplot2.jpg")
        elif self.port==33333:
         plt.savefig("distplot3.jpg")
        elif self.port==44444:
         plt.savefig("distplot4.jpg")
    #返回缺省值的统计情况（饼状图），并且补全缺省值。最后返回缺省的统计数据以及缺省填充后的新的全体数据
    def default_value(self):
        df = self.df
        #统计数量，比率
        count = df.isnull().sum().sort_values(ascending=False)#统计出每个属性的缺省数量，再根据缺省数量倒排序
        ratio = count / len(df)#每个属性的缺省占自己总数的比率
        nulldata = pd.concat([count, ratio], axis=1, keys=['count', 'ratio'])
        #饼状图
        explode = [0]
        explode = explode * len(nulldata.index)
        explode[0] = 0.1
        plt.figure()
        plt.pie(x=count, labels=nulldata.index, autopct='%1.1f%%', shadow=True, startangle=90,
                explode=explode, pctdistance=0.8, textprops={'fontsize': 16, 'color': 'w'})#饼状图画出每个属性的缺省在整体缺省数据的占比
        plt.title('属性缺省占比图')
        plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), borderaxespad=0.3)
        if self.port==11111:
         plt.savefig("default_value_pie1.jpg")
        elif self.port==22222:
         plt.savefig("default_value_pie2.jpg")
        elif self.port==33333:
         plt.savefig("default_value_pie3.jpg")
        elif self.port==44444:
         plt.savefig("default_value_pie4.jpg")
        # 填充缺省，字符串类型的属性用None填充,数值型用众数填充。简单填充，可能会产生更多的误差
        for index in nulldata.index:
            if type(index) is not object:
                df[index].fillna("None", inplace=True)
            else:
                df[index].fillna(df[index].mode()[0], inplace=True)
        return nulldata,df
    #将偏值较大的log归正: 用柱状图将数值型属性的偏值表示出，最终会返回它们的偏值和进行log1p后新的数据
    def Skew(self):
        df=self.df
        skew_value=np.abs(df.skew()).sort_values(ascending=False)
        skew_value=skew_value[skew_value>0.5]
        #用柱状图描述各个属性的偏值
        plt.figure()
        sns.barplot(skew_value.index, skew_value, palette="BuPu_r", label="偏值")
        plt.title('数值型属性的偏值')
        plt.xlabel('属性')
        plt.ylabel('偏值skew')
        plt.legend()
        if self.port==11111:
         plt.savefig("skew_hist1.jpg")
        elif self.port==22222:
         plt.savefig("skew_hist2.jpg")
        elif self.port==33333:
         plt.savefig("skew_hist3.jpg")
        elif self.port==44444:
         plt.savefig("skew_hist4.jpg")
        #对于偏值大于0.15的定量进行正态化
        X_numeric = df.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= 0.5].index
        df[skewness_features] = np.log1p(df[skewness_features])
        self.df=df
        return skew_value,df
    #定义生成词云的方法
    def generate_wordcloud(self,text,name, mask='Images/man_mask.png'):
        # 设置显示方式
        d = path.dirname(__file__)
        alice_mask = np.array(Image.open(path.join(d, mask)))
        font_path = path.join(d, "font//msyh.ttf")
        stopwords = set(STOPWORDS)
        stopwords.add('|')
        wc = WordCloud(background_color="white",  # 设置背景颜色
                       max_words=2000,  # 词云显示的最大词数
                       mask=alice_mask,  # 设置背景图片
                       stopwords=stopwords,  # 设置停用词
                       font_path=font_path,  # 兼容中文字体，不然中文会显示乱码
                       )
        # 生成词云
        wc.generate(text)
        # # 生成的词云图像保存到本地
        if self.port==11111:
         wc.to_file(name+"词云1.jpg")
        elif self.port==22222:
         wc.to_file(name+"词云2.jpg")
        elif self.port==33333:
         wc.to_file(name+"词云3.jpg")
        elif self.port==44444:
         wc.to_file(name+"词云4.jpg")
    #调用生成词云的方法，给每个非数值型特征生成词云保存在本地
    def generate_img(self):
        list=[]
        df=self.df
        for col in df.columns:
            if df[col].dtype == object:
                list.append(col)
                text = '|'.join(df[col].tolist())
                self.generate_wordcloud(text, col)
        return list

class NodeLookup(object):
  def __init__(self,label_lookup_path=None,uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)
    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string
    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]
    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name
    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

def create_graph():
  with tf.gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def ser(PORT,text):
  HOST = '127.0.0.1'
  BUFSIZ = 1024
  ADDR = (HOST, PORT)
  tcpSerSock = socket(AF_INET, SOCK_STREAM)
  tcpSerSock.bind(ADDR)
  tcpSerSock.listen(5)
  while True:
   print('port ',PORT,' is waiting for connection...')
   tcpCliSock, addr = tcpSerSock.accept()
   print('port ',PORT,' connected from:', addr)
   data = tcpCliSock.recv(BUFSIZ).decode()
   if data=='复读机mode':
    text.insert(INSERT,'复读开始\n')
    tcpCliSock.send(data.encode())
    data = tcpCliSock.recv(BUFSIZ).decode()
    text.insert(INSERT,'接收到port')
    text.insert(INSERT,PORT)
    text.insert(INSERT,'端口发送的数据')
    text.insert(INSERT,data)
    text.insert(INSERT,'\n')
    if re.search('你好|您好|hello|hi',data) is not None:
     tcpCliSock.send('您好！'.encode())
    elif re.search('现在|时间|time',data) is not None:
     tcpCliSock.send(('[%s]' %ctime()).encode())
    elif re.search('天气|weather',data) is not None:
     tcpCliSock.send('今天是晴天呢！'.encode())
    elif re.search('再见|see you|goodbye',data) is not None:
     tcpCliSock.send('再见！'.encode())
    elif re.search('你叫什么|你是谁|name',data) is not None:
     tcpCliSock.send('我叫小黑！'.encode())
    else:tcpCliSock.send(data.encode())

   elif data=='加密文件转发mode1':
    tcpCliSock.send(data.encode())
    data = tcpCliSock.recv(BUFSIZ)
    text.insert(INSERT,'接收到来自')
    text.insert(INSERT,PORT)
    text.insert(INSERT,'端口发送到客户端1的数据:\n')
    text.insert(INSERT,data)
    text.insert(INSERT,'\n')
    infile=open("cli1.txt","wb")
    infile.write(data)
    infile.close()
    tcpCliSock.send("接收加密文件完成".encode())

   elif data=='加密文件转发mode2':
    tcpCliSock.send(data.encode())
    data = tcpCliSock.recv(BUFSIZ)
    text.insert(INSERT,'接收到来自')
    text.insert(INSERT,PORT)
    text.insert(INSERT,'端口发送到客户端2的数据:\n')
    text.insert(INSERT,data)
    text.insert(INSERT,'\n')
    infile=open("cli2.txt","wb")
    infile.write(data)
    infile.close()
    tcpCliSock.send("接收加密文件完成".encode())

   elif data=='加密文件转发mode3':
    tcpCliSock.send(data.encode())
    data = tcpCliSock.recv(BUFSIZ)
    text.insert(INSERT,'接收到来自')
    text.insert(INSERT,PORT)
    text.insert(INSERT,'端口发送到客户端3的数据:\n')
    text.insert(INSERT,data)
    text.insert(INSERT,'\n')
    infile=open("cli3.txt","wb")
    infile.write(data)
    infile.close()
    tcpCliSock.send("接收加密文件完成".encode())

   elif data=='加密文件转发mode4':
    tcpCliSock.send(data.encode())
    data = tcpCliSock.recv(BUFSIZ)
    text.insert(INSERT,'接收到来自')
    text.insert(INSERT,PORT)
    text.insert(INSERT,'端口发送到客户端4的数据:\n')
    text.insert(INSERT,data)
    text.insert(INSERT,'\n')
    infile=open("cli4.txt","wb")
    infile.write(data)
    infile.close()
    tcpCliSock.send("接收加密文件完成".encode())

   elif data=='接收文件mode':
    text.insert(INSERT,'转发文件开始\n')
    if PORT==11111:
     outfile=open("cli1.txt","rb")
     data=outfile.read()
     text.insert(INSERT,'将加密文件转发给客户端1\n')
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='success':
      text.insert(INSERT,'转发到客户端1完成\n')
    elif PORT==22222:
     outfile=open("cli2.txt","rb")
     data=outfile.read()
     text.insert(INSERT,'将加密文件转发给客户端2\n')
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='success':
      text.insert(INSERT,'转发到客户端2完成\n')
    elif PORT==33333:
     outfile=open("cli3.txt","rb")
     data=outfile.read()
     text.insert(INSERT,'将加密文件转发给客户端3\n')
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='success':
      text.insert(INSERT,'转发到客户端3完成\n')
    elif PORT==44444:
     outfile=open("cli4.txt","rb")
     data=outfile.read()
     text.insert(INSERT,'将加密文件转发给客户端4\n')
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='success':
      text.insert(INSERT,'转发到客户端4完成\n')

   elif data=='发送图片mode':
    if PORT==11111:
     text.insert(INSERT,'接收到客户端1发送的图片\n')
     tcpCliSock.send('OK'.encode())
     size= tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send('OK'.encode())
     size=int(size)
     data = tcpCliSock.recv(size)
     myfile = open('cli1.jpg', 'wb')
     myfile.write(data)
     myfile.close()
     tcpCliSock.send('OK'.encode())
    elif PORT==22222:
     text.insert(INSERT,'接收到客户端2发送的图片\n')
     tcpCliSock.send('OK'.encode())
     size= tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send('OK'.encode())
     size=int(size)
     data = tcpCliSock.recv(size)
     myfile = open('cli2.jpg', 'wb')
     myfile.write(data)
     myfile.close()
     tcpCliSock.send('OK'.encode())
    elif PORT==33333:
     text.insert(INSERT,'接收到客户端3发送的图片\n')
     tcpCliSock.send('OK'.encode())
     size= tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send('OK'.encode())
     size=int(size)
     data = tcpCliSock.recv(size)
     myfile = open('cli3.jpg', 'wb')
     myfile.write(data)
     myfile.close()
     tcpCliSock.send('OK'.encode())
    elif PORT==44444:
     text.insert(INSERT,'接收到客户端4发送的图片\n')
     tcpCliSock.send('OK'.encode())
     size= tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send('OK'.encode())
     size=int(size)
     data = tcpCliSock.recv(size)
     myfile = open('cli4.jpg', 'wb')
     myfile.write(data)
     myfile.close()
     tcpCliSock.send('OK'.encode())

   elif data=='人脸检测mode':
    if PORT==11111:
     text.insert(INSERT,'对客户端1发送的图片进行人脸检测\n')
     files = {'image_file':open('cli1.jpg', 'rb')}
     payload = {'api_key': 'RGNQvj8QqV9lrjQvUpg2HSB2fL-8iUi5',
           'api_secret': 'QzcTneMMUEdr79SjCC24LJXACesikPyt',
           'return_landmark': 0,
           'return_attributes':'gender,age,glass,beauty'}
     r = requests.post(url,files=files,data=payload)
     data=json.loads(r.text)
     print (r.text)
     if data["faces"]:
      gender=data['faces'][0]['attributes']['gender']['value']
      age=data['faces'][0]['attributes']['age']['value']
      score=(data['faces'][0]['attributes']['beauty']['female_score']+data['faces'][0]['attributes']['beauty']['male_score'])/2
      score='%.2f'%score
      width= data['faces'][0]['face_rectangle']['width']
      top= data['faces'][0]['face_rectangle']['top']
      height= data['faces'][0]['face_rectangle']['height']
      left= data['faces'][0]['face_rectangle']['left']
      data=",".join([gender,str(age),score,str(width),str(top),str(height),str(left)])
      tcpCliSock.send(data.encode())
     else: tcpCliSock.send('No face'.encode())
    elif PORT==22222:
     text.insert(INSERT,'对客户端2发送的图片进行人脸检测\n')
     files = {'image_file':open('cli2.jpg', 'rb')}
     payload = {'api_key': 'RGNQvj8QqV9lrjQvUpg2HSB2fL-8iUi5',
           'api_secret': 'QzcTneMMUEdr79SjCC24LJXACesikPyt',
           'return_landmark': 0,
           'return_attributes':'gender,age,glass,beauty'}
     r = requests.post(url,files=files,data=payload)
     data=json.loads(r.text)
     print (r.text)
     if data["faces"]:
      gender=data['faces'][0]['attributes']['gender']['value']
      age=data['faces'][0]['attributes']['age']['value']
      score=(data['faces'][0]['attributes']['beauty']['female_score']+data['faces'][0]['attributes']['beauty']['male_score'])/2
      score='%.2f'%score
      width= data['faces'][0]['face_rectangle']['width']
      top= data['faces'][0]['face_rectangle']['top']
      height= data['faces'][0]['face_rectangle']['height']
      left= data['faces'][0]['face_rectangle']['left']
      data=",".join([gender,str(age),score,str(width),str(top),str(height),str(left)])
      tcpCliSock.send(data.encode())
     else: tcpCliSock.send('No face'.encode())
    elif PORT==33333:
     text.insert(INSERT,'对客户端3发送的图片进行人脸检测\n')
     files = {'image_file':open('cli3.jpg', 'rb')}
     payload = {'api_key': 'RGNQvj8QqV9lrjQvUpg2HSB2fL-8iUi5',
           'api_secret': 'QzcTneMMUEdr79SjCC24LJXACesikPyt',
           'return_landmark': 0,
           'return_attributes':'gender,age,glass,beauty'}
     r = requests.post(url,files=files,data=payload)
     data=json.loads(r.text)
     print (r.text)
     if data["faces"]:
      gender=data['faces'][0]['attributes']['gender']['value']
      age=data['faces'][0]['attributes']['age']['value']
      score=(data['faces'][0]['attributes']['beauty']['female_score']+data['faces'][0]['attributes']['beauty']['male_score'])/2
      score='%.2f'%score
      width= data['faces'][0]['face_rectangle']['width']
      top= data['faces'][0]['face_rectangle']['top']
      height= data['faces'][0]['face_rectangle']['height']
      left= data['faces'][0]['face_rectangle']['left']
      data=",".join([gender,str(age),score,str(width),str(top),str(height),str(left)])
      tcpCliSock.send(data.encode())
     else: tcpCliSock.send('No face'.encode())
    elif PORT==44444:
     text.insert(INSERT,'对客户端4发送的图片进行人脸检测\n')
     files = {'image_file':open('cli4.jpg', 'rb')}
     payload = {'api_key': 'RGNQvj8QqV9lrjQvUpg2HSB2fL-8iUi5',
           'api_secret': 'QzcTneMMUEdr79SjCC24LJXACesikPyt',
           'return_landmark': 0,
           'return_attributes':'gender,age,glass,beauty'}
     r = requests.post(url,files=files,data=payload)
     data=json.loads(r.text)
     print (r.text)
     if data["faces"]:
      gender=data['faces'][0]['attributes']['gender']['value']
      age=data['faces'][0]['attributes']['age']['value']
      score=(data['faces'][0]['attributes']['beauty']['female_score']+data['faces'][0]['attributes']['beauty']['male_score'])/2
      score='%.2f'%score
      width= data['faces'][0]['face_rectangle']['width']
      top= data['faces'][0]['face_rectangle']['top']
      height= data['faces'][0]['face_rectangle']['height']
      left= data['faces'][0]['face_rectangle']['left']
      data=",".join([gender,str(age),score,str(width),str(top),str(height),str(left)])
      tcpCliSock.send(data.encode())
     else: tcpCliSock.send('No face'.encode())

   elif data=='目标识别mode':
    if PORT==11111:
     text.insert(INSERT,'对客户端1发送的图片进行目标识别\n')
     image_data = tf.gfile.FastGFile('cli1.jpg','rb').read()
     create_graph()
     sess=tf.Session()
#Inception-v3模型的最后一层softmax的输出
     softmax_tensor= sess.graph.get_tensor_by_name('softmax:0')
#输入图像数据，得到softmax概率值（一个shape=(1,1008)的向量）
     predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
     predictions = np.squeeze(predictions)
     node_lookup = NodeLookup()
#取出前5个概率最大的值（top-5)
     top_5 = predictions.argsort()[-5:][::-1]
     list=[]
     for node_id in top_5:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      score ='%.5f'%score
      list.append(human_string+'!'+score+'!')
     sess.close()
     data=''.join(list)[:-1]
     tcpCliSock.send(data.encode())
    elif PORT==22222:
     text.insert(INSERT,'对客户端2发送的图片进行目标识别\n')
    elif PORT==33333:
     text.insert(INSERT,'对客户端3发送的图片进行目标识别\n')
     myfile.write(data)
     tcpCliSock.send('OK'.encode())
    elif PORT==44444:
     text.insert(INSERT,'对客户端4发送的图片进行目标识别\n')

   elif data=='发送csv文件mode':
    if PORT==11111:
     text.insert(INSERT,'接收到客户端1发送的csv文件\n')
     tcpCliSock.send('OK'.encode())
     size= tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send('OK'.encode())
     size=int(size)
     data = tcpCliSock.recv(size)
     myfile = open('cli1.csv', 'wb')
     myfile.write(data)
     myfile.close()
     tcpCliSock.send('OK'.encode())
    elif PORT==22222:
     text.insert(INSERT,'接收到客户端2发送的csv文件\n')
     tcpCliSock.send('OK'.encode())
     size= tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send('OK'.encode())
     size=int(size)
     data = tcpCliSock.recv(size)
     myfile = open('cli2.csv', 'wb')
     myfile.write(data)
     myfile.close()
     tcpCliSock.send('OK'.encode())
    elif PORT==33333:
     text.insert(INSERT,'接收到客户端3发送的csv文件\n')
     tcpCliSock.send('OK'.encode())
     size= tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send('OK'.encode())
     size=int(size)
     data = tcpCliSock.recv(size)
     myfile = open('cli3.csv', 'wb')
     myfile.write(data)
     myfile.close()
     tcpCliSock.send('OK'.encode())
    elif PORT==44444:
     text.insert(INSERT,'接收到客户端4发送的csv文件\n')
     tcpCliSock.send('OK'.encode())
     size= tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send('OK'.encode())
     size=int(size)
     data = tcpCliSock.recv(size)
     myfile = open('cli4.csv', 'wb')
     myfile.write(data)
     myfile.close()
     tcpCliSock.send('OK'.encode())

   elif data=='数据分析mode':
    if PORT==11111:
     text.insert(INSERT,'对客户端1发送的csv文件数据分析\n')
     data=DataAnalyze('cli1.csv',PORT)
     data.correlation()
     data.dig_dist()
     null,df=data.default_value()
     skew,df=data.Skew()
     global cloud1
     cloud1.clear()
     cloud1=data.generate_img()
     tcpCliSock.send('OK'.encode())
    elif PORT==22222:
     text.insert(INSERT,'对客户端2发送的csv文件数据分析\n')
     data=DataAnalyze('cli2.csv',PORT)
     data.correlation()
     data.dig_dist()
     null,df=data.default_value()
     skew,df=data.Skew()
     global cloud2
     cloud2.clear()
     cloud2=data.generate_img()
     tcpCliSock.send('OK'.encode())
    elif PORT==33333:
     text.insert(INSERT,'对客户端3发送的csv文件数据分析\n')
     data=DataAnalyze('cli3.csv',PORT)
     data.correlation()
     data.dig_dist()
     null,df=data.default_value()
     skew,df=data.Skew()
     global cloud3
     cloud3.clear()
     cloud3=data.generate_img()
     tcpCliSock.send('OK'.encode())
    elif PORT==44444:
     text.insert(INSERT,'对客户端4发送的csv文件数据分析\n')
     data=DataAnalyze('cli4.csv',PORT)
     data.correlation()
     data.dig_dist()
     null,df=data.default_value()
     skew,df=data.Skew()
     global cloud4
     cloud4.clear()
     cloud4=data.generate_img()
     tcpCliSock.send('OK'.encode())

   elif data=='热力图mode':
    if PORT==11111:
     text.insert(INSERT,'准备向客户端1发送热力图\n')
     myfile = open('heatmap1.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端1发送热力图完成\n')
    elif PORT==22222:
     text.insert(INSERT,'准备向客户端2发送热力图\n')
     myfile = open('heatmap2.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端2发送热力图完成\n')
    elif PORT==33333:
     text.insert(INSERT,'准备向客户端3发送热力图\n')
     myfile = open('heatmap3.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端3发送热力图完成\n')
    elif PORT==44444:
     text.insert(INSERT,'准备向客户端4发送热力图\n')
     myfile = open('heatmap4.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端4发送热力图完成\n')

   elif data=='柱状图mode':
    if PORT==11111:
     text.insert(INSERT,'准备向客户端1发送柱状图\n')
     myfile = open('skew_hist1.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端1发送柱状图完成\n')
    elif PORT==22222:
     text.insert(INSERT,'准备向客户端2发送柱状图\n')
     myfile = open('skew_hist2.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端2发送柱状图完成\n')
    elif PORT==33333:
     text.insert(INSERT,'准备向客户端3发送柱状图\n')
     myfile = open('skew_hist3.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端3发送柱状图完成\n')
    elif PORT==44444:
     text.insert(INSERT,'准备向客户端4发送柱状图\n')
     myfile = open('skew_hist4.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端4发送柱状图完成\n')

   elif data=='饼状图mode':
    if PORT==11111:
     text.insert(INSERT,'准备向客户端1发送饼状图\n')
     myfile = open('default_value_pie1.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端1发送饼状图完成\n')
    elif PORT==22222:
     text.insert(INSERT,'准备向客户端2发送饼状图\n')
     myfile = open('default_value_pie2.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端2发送饼状图完成\n')
    elif PORT==33333:
     text.insert(INSERT,'准备向客户端3发送饼状图\n')
     myfile = open('default_value_pie3.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端3发送饼状图完成\n')
    elif PORT==44444:
     text.insert(INSERT,'准备向客户端4发送饼状图\n')
     myfile = open('default_value_pie4.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端4发送饼状图完成\n')

   elif data=='散点图mode':
    if PORT==11111:
     text.insert(INSERT,'准备向客户端1发送散点图\n')
     myfile = open('distplot1.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端1发送散点图完成\n')
    elif PORT==22222:
     text.insert(INSERT,'准备向客户端2发送散点图\n')
     myfile = open('distplot2.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端2发送散点图完成\n')
    elif PORT==33333:
     text.insert(INSERT,'准备向客户端3发送散点图\n')
     myfile = open('distplot3.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端3发送散点图完成\n')
    elif PORT==44444:
     text.insert(INSERT,'准备向客户端4发送散点图\n')
     myfile = open('distplot4.jpg', 'rb')
     data = myfile.read()
     size = str(len(data))
     tcpCliSock.send(size.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     tcpCliSock.send(data)
     if tcpCliSock.recv(BUFSIZ).decode()=='OK':
      text.insert(INSERT,'向客户端4发送散点图完成\n')

   elif data=='词云mode':
    if PORT==11111:
     text.insert(INSERT,'准备向客户端1发送词云图\n')
     count=str(len(cloud1))
     tcpCliSock.send(count.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     for i in range(len(cloud1)):
      myfile = open(cloud1[i]+'词云1.jpg', 'rb')
      data = myfile.read()
      size=str(len(data))
      tcpCliSock.send(size.encode())
      rec = tcpCliSock.recv(BUFSIZ).decode()
      tcpCliSock.send(data)
      tcpCliSock.recv(BUFSIZ).decode()
      myfile.close()
    elif PORT==22222:
     text.insert(INSERT,'准备向客户端2发送词云图\n')
     count=str(len(cloud2))
     tcpCliSock.send(count.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     for i in range(len(cloud2)):
      myfile = open(cloud2[i]+'词云2.jpg', 'rb')
      data = myfile.read()
      size=str(len(data))
      tcpCliSock.send(size.encode())
      rec = tcpCliSock.recv(BUFSIZ).decode()
      tcpCliSock.send(data)
      tcpCliSock.recv(BUFSIZ).decode()
      myfile.close()
    elif PORT==33333:
     text.insert(INSERT,'准备向客户端3发送词云图\n')
     count=str(len(cloud3))
     tcpCliSock.send(count.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     for i in range(len(cloud3)):
      myfile = open(cloud3[i]+'词云3.jpg', 'rb')
      data = myfile.read()
      size=str(len(data))
      tcpCliSock.send(size.encode())
      rec = tcpCliSock.recv(BUFSIZ).decode()
      tcpCliSock.send(data)
      tcpCliSock.recv(BUFSIZ).decode()
      myfile.close()
    elif PORT==44444:
     text.insert(INSERT,'准备向客户端4发送词云图\n')
     count=str(len(cloud4))
     tcpCliSock.send(count.encode())
     rec = tcpCliSock.recv(BUFSIZ).decode()
     for i in range(len(cloud4)):
      myfile = open(cloud4[i]+'词云4.jpg', 'rb')
      data = myfile.read()
      size=str(len(data))
      tcpCliSock.send(size.encode())
      rec = tcpCliSock.recv(BUFSIZ).decode()
      tcpCliSock.send(data)
      tcpCliSock.recv(BUFSIZ).decode()
      myfile.close()

   tcpCliSock.close()

class Ser:
 def __init__(self):
  window=Tk()
  window.title("Serve")
  self.text=Text(window,width=100,height=20)
  self.text.pack()
  frame1=Frame(window)
  frame1.pack()
  Button(frame1,text="创建服务端",command=self.processInit).grid(row=1,column=1)
  Button(frame1,text="清空记录",command=self.processClear).grid(row=1,column=2)
  window.mainloop()

 def processInit(self):
  self.text.insert(INSERT,'Init\n')
  self.threads=[]
  for i in count:
   t=threading.Thread(target=ser,args=(PORT[i],self.text))
   self.threads.append(t) 
  for i in count:
   self.threads[i].start()

 def processClear(self):
  self.text.delete('1.0','end')

Ser()