# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import imageio
import facenet
from scipy import misc


image_size = 160 #don't need equal to real image size, but this value should not small than this
modeldir = '/Users/chenxiaoxuan/Downloads/20180402-114759' #change to your model dir
savdir = '/Users/chenxiaoxuan/Documents/python/face'

print('建立facenet embedding模型')
tf.Graph().as_default()
sess = tf.InteractiveSession()
#init = tf.global_variables_initializer()
#sess.run(init)
#with sess:

facenet.load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

print('facenet embedding模型建立完毕')



from facenet import detect_face

margin=32
with tf.Graph().as_default():
    sess2 = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    with sess2.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess2, None)
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
def prewhiten(img):
    return np.multiply(np.subtract(img,127.5),1/128.0)
def image_character(image_n):
    img = imageio.imread(image_n, pilmode='RGB')
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        return np.add(np.zeros(512),2)
    det = np.squeeze(bounding_boxes[0,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = [prewhiten(aligned)]
    
    #emb_array1 = np.zeros((1, embedding_size))
    return sess.run(embeddings, feed_dict={images_placeholder: prewhitened, phase_train_placeholder: False })[0]


characters=np.zeros(512)+2
names=['No_Face']
with open(savdir+"/names.txt",'a') as f:
    f.write("0")
def add_face(vec,name):
    global characters,names
    names.append(name)
    characters=np.column_stack((characters,vec))
    np.save(savdir+"/chara.npy",characters)
    with open(savdir+"/names.txt",'a') as f:
        f.write(" "+name)
#add_face(characters,"0")
def get_name(vec):
    tmp=np.sum(np.square(np.subtract(characters,vec[:,None])),axis=0)
    index=tmp.argmin()
    if tmp[index]<0.9:
        return names[index]
    else:
        return "N"



#from urllib.parse import urlparse
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import base64
# Instantiate the Node

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('./index.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    
    tmp = request.files['avatar']
    vec=image_character(tmp)
    response = {
        'message':base64.b64encode(vec.tostring()),
        'name':get_name(vec)
    }
    return jsonify(response), 200

@app.route('/add_data', methods=['POST'])
def add_data():
    add_face(np.fromstring(base64.b64decode(request.form['message']),dtype=np.float32),request.form['name'])
    response = {
        'message':'ha'
    }
    return jsonify(response), 200



if __name__ == '__main__':
    #from argparse import ArgumentParser
    #with open('./data.txt','r') as cg:
    #    nodes=cg.read().split('|')
    #parser = ArgumentParser()
    #parser.add_argument('-p', '--port', default=80, type=int, help='port to listen on')
    #args = parser.parse_args()
    port=80
    #port = args.port
    app.run(host='0.0.0.0', port=port)
