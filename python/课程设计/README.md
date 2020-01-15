设计内容：

基于多线程socket通信设计了一个简单的C/S架构，多个客户端选择功能模块并向服务端发送数据，再由服务端进行计算后将处理结果返回给对应的客户端。整个过程是可视化的，通过Tkinter来绘制客户端与服务端的界面。

一、Demo运行

1.代码获取
可从Github或者Python课设群里下载到py文件和其他文件夹，配置好tensorflow-gpu环境以及其他需要的库后，运行ser+.py文件，就可以生成服务端。而cli1.py、cli2.py、cli3.py与cli4.py分别对应四个客户端，如果在同一个局域网内可以将socket通信绑定的IP地址更改为同一局域网IP地址。代码中默认的IP地址为127.0.0.1，这是本机环回地址。四个客户端绑定在同一个IP地址下，通过不同的端口号可以实现进程间的通信。因此，在服务端开启后，四个socket线程分别监听四个端口，这样四个客户端可以同时操作，相互之间不会干扰。
2.操作与命令
服务端在终端输入命令：
activate tensorflow-gpu(激活tensorflow环境)
python ser+.py (运行服务端程序)

服务端界面
客户端1在终端输入命令：python cli1.py (运行客户端1)
客户端2、3、4也是同样运行对应的客户端程序。

客户端界面
二、设计背景
计算机已经越来越成为人们生活必不可少的一部分，不管是日常文字处理还是专业的图像和数据处理，计算机都发挥着越来越重要的作用。但是由于客户机的算力有限，因此通常会将一些大型的项目放在服务器上面运行，然后由服务器返回处理结果。客户机仅仅充当了一个数据的“搬运工”而并非“工作者”。
这就是当前软件开发中使用最多的“C/S架构”：客户端因特定的请求而联系服务器，并发送必要的数据，然后等待服务器的回应，最后完成请求或给出故障的原因。服务器无限地运行下去，并不断地处理请求。

从办公的OFFICE，WPS，WINRAR到杀毒软件如金山，瑞金再到我们的娱乐软件，如播放器，QQ，微信等，无处不见C/S架构。

三、设计目标
1.基于多线程socket构建了一个简单的C/S架构，多线程socket用来监听多个端口。
2.基于Tkinter来绘制服务端与客户端的界面，客户端通过点击按钮向服务端发出请求。
3.每次请求都会建立一个TCP连接，实现客户端与服务端之间通信（传输文件、图片等）。
4.每个功能都在服务端实现，再通过socket通信将处理结果返回给客户端。
5.实现下面五个功能：
    
C/S通信     文件加密传输  人脸检测       目标识别          数据分析
四、技术路线
环境配置：tensorflow-gpu 1.14.0     cuda 10.0
          Python3.7                 GPU:GTX 1060


五、关键原理与实现
1.多线程socket通信原理（基于TCP协议）
socket接口是TCP/IP网络的API，该接口定义了许多函数或例程。通过socket模块，可以用它们来开发TCP/IP网络上的应用程序。
服务端整个流程：
创建并初始化TCP 套接字，使用 SOCK_STREAM 作为套接字类型。

1）将地址（主机名、端口号对）绑定到TCP套接字上。

2）设置并启动 TCP 监听器来监听端口通信。

3）被动接受 TCP 客户端连接，一直等待直到连接到达（阻塞）。

4）连接建立，交换数据。

5）通信结束，关闭TCP连接。

客户端整个流程：
1）创建并初始化TCP 套接字，使用 SOCK_STREAM 作为套接字类型。

2）主动发起 TCP 服务器连接。

3）连接建立，交换数据。

4）通信结束，关闭TCP连接。


多线程通信的实现：
1）导入threading模块，在服务端创建一个线程列表。

2）创建四个线程，每个线程的target均调用ser（）函数来实现，但是监听的端口号不同。

3）执行四个线程。

2.基于Tkinter的图形界面编程
Tkinter是Python的默认GUI库，它结合了GUI开发的灵活性与集成脚本语言的简洁性，可以让新手快速开发和实现很多与商业软件品质相当的GUI 应用。而本课设中使用到的Tkinter构件有很多，各自的作用也并不相同。
Frame：作为包含其他控件的纯容器。
Button：按钮；用来实现相应的功能，服务端与客户端的连接通信都是通过按钮触发的。
Label：文本标签；用来显示较短的文本。
Text：多行文本框；用来显示服务端与客户端的操作信息。
Canvas：画布；用来显示目标识别和人脸识别功能中服务端返回的信息。
Messagebox：弹出提示框；用来提示一些非法操作信息与返回信息。

文件不存在         文件格式错误      人脸检测返回信息   数据分析返回信息
3.各个功能模块的原理
3.1简单人机交互
服务端整个流程：
1）与客户端建立TCP连接后，接收客户端发送的数据。
2）若接收到客户端数据“复读机mode”，开启人机交互模块。
3）对客户端发送的数据进行正则表达式关键字匹配。
4）匹配成功返回对应的信息，匹配失败则返回客户端发送的信息（复读机的由来）。


客户端整个流程：
1）与服务端建立TCP连接，发送数据“复读机mode”。
2）将用户输入的信息写入文本框内，并发送给服务端。
3）接收服务端返回的信息并写入文本框内，关闭TCP连接。

3.2文件加密转发
服务端整个流程：
1）与客户端建立TCP连接后，接收客户端发送的数据。
2）若接收到客户端数据为“加密文件转发mode2”，服务端会准备接收加密文件。
（ps：mode后面的数字代表发送文件的客户端期望将文件转发至哪一个客户端，mode2代表该客户端期望将加密文件转发给客户端2。）
3)接收客户端发送的字节流，并将字节流写入本地文件“cli2.txt”。

4）若接收到客户端2数据“接收文件mode”，服务端会准备转发加密文件。
5）将“cli2.txt”文件转发给客户端2。

发送文件客户端整个流程：
1）浏览需要发送的文件，确认后对文件进行加密，加密成功后写入“jiami.txt”文件。
加密过程如下：
逐个字符的读取文件，并对每个字符做以下操作，直到读取到文件末尾。
a)使用ord()把字符转为十进制数
b)判断十进制数是否大于250（循环移位，位移大小为5）
i.小于250的进行加5
ii.大于250的减去250

2）与服务端建立TCP连接，并将加密后的“jiami.txt”文件发送给服务端。
3）发送完毕后关闭TCP连接，由服务端将文件转发给其他客户端。

接收文件客户端整个流程：
1）点击“接收文件”按钮，与服务端建立TCP连接，请求接收转发来的文件。
2）从服务端接收到文件后关闭TCP连接，对文件进行解密，获取原文件。
解密过程如下：
a)按字节读取加密后的文件，使用ord转为十进制数（加密过程的逆）
i.若十进制数小于5，则加250
ii.若十进制数大于5，则减5
b)把解密后的文件保存在本地，并显示在text域中


3.3基于旷视API的人脸检测
服务端整个流程：
1）与客户端建立TCP连接后，接收客户端发送的数据。
2）若接收到客户端数据为“发送图片mode”，服务端会准备接收图片。
3）首先会接收到客户端发送的图片大小，然后根据图片大小接收客户端发送的图片。
4）将客户端发送的图片保存到本地。

5）若接收到客户端数据为“人脸检测mode”，则开启人脸检测。
6）将客户端之前发送的图片上传到旷视的云平台上，并接收返回的人脸检测结果。

原理：通过request.post()函数向Face++云平台上传和接收数据，上传数据为客户端传过来的图片，接收的数据是云平台的检测结果。下面是旷视API文档的一部分。

旷视云平台返回的数据为json格式，通过json.loads函数进行解码，可以得到faces数组。其中包括face_token、face_rectangle和attributes数组。face_token是人脸的标识号，face_rectangle中是人脸在图片中的位置，attributes数组里面包含了人脸的一些相关信息，比如性别、年龄、是否戴眼镜、颜值评分等元素。

（上图仅为attributes数组中的部分元素）
7）将返回的人脸检测结果发送给客户端。
原理：需要返回的数据分为两种情况
a．存在人脸，则返回性别、年龄、颜值评分、人脸的位置
（通过join函数将各个参数用逗号隔开，拼接成一个字符串）
b．不存在人脸，返回信息“No face”

客户端整个流程：
1）浏览需要发送的图片，确认后建立TCP连接并向服务端发送图片。
图片发送详细过程：
a．二进制读取需要发送的图片，并用size记录字节流大小。
b．将size转成字符串类型发送给服务端，告知其将要发送字节流的大小。
c．得到服务端的回复后客户端将字节流发送给服务端，发送图片完成。
2）发送图片完毕，关闭TCP连接

3）点击“检测结果”按钮，与服务端建立TCP连接，获取人脸检测结果。
4）接收数据后关闭TCP连接，对服务端返回的数据进行处理并展示。
处理过程原理：
a．若不存在人脸，利用messagebox弹出提示框“No face in the picture！”
b．若存在人脸
i．将服务端返回的字符串分割到列表中，得到各项信息。
ii．基于返回的人脸位置参数，使用opencv中的cv2.rectangle（）框出人脸。
iii．将性别、年龄与颜值打分信息写入新建窗口的画布中。


3.4数据分析和结果反馈
服务端整个流程：
1）与客户端建立TCP连接后，接收客户端发送的数据。
2）若接收到客户端数据为“发送csv文件mode”，服务端会准备接收csv文件。
3）首先会接收到客户端发送的文件大小，再根据文件大小接收客户端发送的csv文件。
4）将客户端发送的csv文件保存到本地。

5）若接收到客户端数据为“数据分析mode”，则准备进行数据分析。
6）将客户端之前发送的csv文件进行数据分析，数据分析结束后会通知客户端。


数据分析模块详解[DataAnalyze类的解析]：
DataAnalyze类的属性：
self.file: csv文件指针
self.df: pd.read_csv()函数将csv文件读取到DataFrame变量df中
self.port: 当前通信的端口号
DataAnalyze类的方法：
correlation( ):直接借助DataFrame的corr函数来获得任意两个属性的关系数据，利用sns.heatmap(df.corr(), cmap="YlGnBu")做出热力图。
dig_dist( ):生成散点图，相关代码如下：

default_value( ): 首先使用df.isnull().sum()统计每种属性的缺省的数量以及缺省数占总数的比率，并将这两个特征组合在一起构成一个新的DataFrame数据返回。然后根据所有缺省之中每种特征的占比做出饼状图来显示。
Skew( ): 计算每种数值型特征的偏值，将其倒排序后返回，并利用柱状图将其表示出来。
generate_wordcloud( ):生成词云图，通过调节mask参数可以改变生成词云的形状
generate_img( ):将生成的词云图命名并保存在本地

7)按客户端的要求返回相应的数据分析结果。
(根据客户端发送的信息不同返回不同的统计图)

客户端整个流程：
1）浏览需要发送的csv文件，确认后建立TCP连接并向服务端发送文件。
（发送csv文件的流程与发送图片类似，都需要先确认文件的大小，再发送文件）
2）按下“数据分析”按钮，之后根据服务端返回的信息判断数据是否分析成功。

3）数据分析成功后，按下不同的按钮会请求服务端返回相应的统计图。（以热力图为例）


3.5基于深度学习的目标识别
服务端整个流程：
1）与客户端建立TCP连接后，接收客户端发送的数据。
2）若接收到客户端数据为“发送图片mode”，服务端会接收客户端发送的图片。
3）若接收到客户端数据为“目标识别mode”，服务端会对发送的图片进行目标识别。
4）将目标识别的结果返回给客户端。
目标识别过程详解：
a．使用tf.gfile.FastGFile().read()读取客户端发送的文件
b. 创建图（网络结构），本课设中直接导入了训练好的Googlnet模型。

c．通过调用NodeLookup类来将softmax概率值映射到标签上。
下面是本课设基于tensorflow的TFlearn模块构建的Googlenet模型框架：
#网络结构
conv1_7_7 = conv_2d(images, 64, 3, strides=1, activation='relu', name='conv1_7_7_s2')
    pool1_3_3 = max_pool_2d(conv1_7_7, 2, strides=2)
pool1_3_3 = local_response_normalization(pool1_3_3)
conv2_3_3_reduce=conv_2d(pool1_3_3,96,3,strides=1,activation='relu')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3,  strides=1,activation='relu')
    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=2, strides=2)
# 3a （Inception结构）
    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu',)
    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu')
    inception_3a_3_3=conv_2d(inception_3a_3_3_reduce,1283,activation='relu')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu')
    inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, 5, activation='relu')
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1)
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, 1, activation='relu')
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)
# 3b
    inception_3b_1_1 = conv_2d(inception_3a_output, 128, 1, activation='relu')
    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, 1, activation='relu')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, 3, activation='relu')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, 1, activation='relu')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, 5, name='inception_3b_5_5')
    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1)
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, 1, activation='relu')
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat', axis=3, name='inception_3b_output')
    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2,)
# 4a
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, 3,  activation='relu')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, 5,  activation='relu')
    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1)
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, 1, activation='relu')
    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')
# 4b
    inception_4b_1_1 = conv_2d(inception_4a_output, 160, 1, activation='relu')
    inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, 1, activation='relu')
    inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, 3, activation='relu')
    inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, 1, activation='relu')
    inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, 5,  activation='relu')
    inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1)
    inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, 1, activation='relu')
    inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')
# 4c
    inception_4c_1_1 = conv_2d(inception_4b_output, 128, 1, activation='relu')
    inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, 1, activation='relu')
    inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256, 3, activation='relu')
    inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, 1, activation='relu')
    inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64, 5, activation='relu')
    inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
    inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, 1, activation='relu')
    inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3, name='inception_4c_output')
# 4d
    inception_4d_1_1 = conv_2d(inception_4c_output, 112, 1, activation='relu')
    inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, 1, activation='relu')
    inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, 3, activation='relu')
    inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, 1, activation='relu')
    inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, 5,  activation='relu')
    inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1)
    inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, 1, activation='relu')
    inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')
# 4e
    inception_4e_1_1 = conv_2d(inception_4d_output, 256, 1, activation='relu')
    inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, 1, activation='relu')
    inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, 3, activation='relu')
    inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, 1, activation='relu')
    inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128, 5, activation='relu')
    inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1)
    inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, 1, activation='relu')
    inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=3, mode='concat')
    pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2)
# 5a
    inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu')
    inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, 3, activation='relu')
    inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, 5,  activation='relu')
    inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1)
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, 1, activation='relu')
    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3, mode='concat')
# 5b
    inception_5b_1_1 = conv_2d(inception_5a_output, 384, 1, activation='relu')
    inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, 1, activation='relu')
    inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384, 3, activation='relu')
    inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, 1, activation='relu')
    inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, 5, activation='relu')
    inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1)
    inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, 1, activation='relu')
    inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')
pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
# 全连接层
    flatten = tflearn.flatten(pool5_7_7)
    fc1 = fully_connected(dropout(flatten, keep_prob),1024,activation='tanh',scope='fc1')

客户端整个流程：
1）浏览需要进行目标识别的图片，确认后建立TCP连接并向服务端发送图片。
2）发送图片完毕，关闭TCP连接。
3）点击“识别结果”按钮，与服务端建立TCP连接，获取目标识别结果。
4）接收数据后关闭TCP连接，对服务端返回的数据进行处理并展示。
处理过程详解：
a. 将服务端返回的字符串分割到列表中，得到目标识别的结果。
b. 将目标识别的结果写入新建窗口的画布中。


六、结果展示
测试功能一：简单人机交互
  
服务端界面接收到的数据                      客户端界面 
     
人机交互窗口
测试功能二：文件加密传输
 
客户端1发送文件窗口                    客户端2接收文件窗口
测试功能三：人脸检测

客户端传入图片

     服务端返回人脸检测结果                      颜值分析结果

测试功能四：数据分析
（1）客户端向服务端传入一个drinks.csv文件
（2）服务端对其进行数据分析，并可以按要求向客户端返回对应的统计图

热力图                                柱状图
 
散点图                            词云图

测试功能五：目标识别
 
客户端上传图片                 服务端返回的目标识别结果
 
客户端上传图片                 服务端返回的目标识别结果
 
客户端上传图片                 服务端返回的目标识别结果

七、存在的问题
1、GUI界面不够美观。
2、受限于服务器端的性能，在进行目标识别时可能需耗费较长时间。
3、受限于解决socket多线程处理的复杂性，与主流通信软件相比缺少一定的功能（例如视频通话等），但现在完成的结构易于添加相应的功能模块，只需在客户端定义添加相应的类即可。

八、总结与展望
总结：
本课设基于多线程socket通信做了一个简单的C/S架构，在这个架构上，我们每个人都能够将自己设计的模块应用上去，并且做到了可视化操作。无论是简单的人机交互、文件加密传输，还是后面调用旷视的API接口实现人脸检测、数据分析与基于深度学习的目标识别，这些功能都是在网络通信的基础上实现的。
展望：
通过这次课设，我们对于多线程网络编程、GUI编程、数据分析与深度学习方面都有了更加深刻的理解，使用一些课堂上学过的模块也越来越得心应手。但由于时间有限，本课设尽管实现了很多功能，但是系统的健壮性方面并没有进行过多的测试。比如说一些非法操作并没有考虑到，如果要进一步优化程序，可以从这个方面入手。


参考文献：
1．《Python核心编程（第三版）》
2.旷视Face++平台：
 https://www.faceplusplus.com.cn/face-detection/
3.socket通信原理介绍：
 https://blog.csdn.net/dlutbrucezhang/article/details/8577810
4.目标识别模块：
https://blog.csdn.net/sinat_27382047/article/details/80534234
5.数据处理模块：
（1）数据预处理
https://segmentfault.com/a/1190000019706696?utm_medium=referral&utm_source=tuicool
https://blog.csdn.net/qq_29893385/article/details/100081419
（2）seaborn介绍
https://blog.csdn.net/geekmubai/article/details/87008603
（3）词云分析
https://blog.csdn.net/ArthurLok/article/details/83817305
