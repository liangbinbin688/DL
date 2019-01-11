import os
import tensorflow as tf
from PIL import Image
import logging
import matplotlib.pyplot as plt
import numpy as np
cwd='pic/train'
IMAGE_NUM=4800
BATCH_SIZE=200
"""
cwd='pic/validation/'
classs={'glacier','rock','urban','water','wetland','wood'}
writer= tf.python_io.TFRecordWriter("pic/validation.tfrecords")

logging.INFO('开始了！')
for index,name in enumerate(classs):
    class_path=cwd+name+'/'
    for img_name in os.listdir(class_path):
        img_path=class_path+img_name
        img=Image.open(img_path)
        img=img.resize((128,128))
        img_raw=img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) #example对象对label和image数据进行封装
    writer.write(example.SerializeToString())
writer.close()
"""
"""
img=Image.open(cwd+'glacier'+'/'+'40957_91343_18.jpg')
#img.resize(128*128)
img=img.resize((128,128))
print(img)
"""
def read_and_decode(filename,batch_size):
    filename_queue=tf.train.string_input_producer([filename])
    reader=tf.TFRecordReader()
    file_, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来
    img=tf.decode_raw(features['img_raw'],tf.uint8)
    img=tf.reshape(img,[128,128,3])
    img=tf.cast(img,tf.float32)*(1./255)-0.5
    label=tf.cast(features['label'],tf.int32)
    X,Y=tf.train.batch([img, label], batch_size=batch_size, num_threads=4, capacity=batch_size*8)

    return X,Y

"""
filename_queue = tf.train.string_input_producer(["pic/train.tfrecords"],shuffle=True) #读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                   })  #取出包含image和label的feature对象
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [128, 128, 3])
label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(30):
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
        print(example, l)
    coord.request_stop()
    coord.join(threads)
"""

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    """
        for i in range(int(IMAGE_NUM/BATCH_SIZE)):
        print("第%d个batch"% (i+1))
        X,Y=read_and_decode('pic/train.tfrecords',BATCH_SIZE)
        print(Y)
    """
    X, Y = read_and_decode('pic/train.tfrecords', BATCH_SIZE)
#将label进行one_hot操作
    label=tf.one_hot(Y,6)
    print(label)








