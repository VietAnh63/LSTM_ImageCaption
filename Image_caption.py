# Thêm thư viện
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
#% matplotlib inline
import string
import os
import os.path
from os import path
from PIL import Image
import glob
from pickle import dump, load
from time import time
import time
from keras.preprocessing import sequence
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

#####TIỀN XỬ LÝ####

# Đọc file chứa caption
# Đọc file các caption
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

filename = "Flickr8k/Flickr8k_text/Flickr8k.token.txt"
doc = load_doc(filename)

# Lưu caption dưới dạng key value: id_image : ['caption 1', 'caption 2', 'caption 3',' caption 4', 'caption 5']
def load_descriptions(doc):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # Nếu dòng nào có độ dài = 1 ko xét
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        # Phương thức join dùng để nối chuỗi trong mảng
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping

descriptions = load_descriptions(doc)

# Hàm này loại bỏ các ký tự đặc biệt
# Preprocessing text
def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    # Hàm maketrans có 3 tham số: ts1 k/tự thay thế, ts2 k/tự cần được thay thế, ts3 k/tự loại bỏ
    # Hàm string.punctuation là lấy các ký tự đặc biệt trong câu
    # Ham punctuation chi duoc thuc thi khi co translate
    table = str.maketrans('', '', string.punctuation)
    # .item() dùng với dict, trả về toàn bộ key và value
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            # Tach tung tieng va dua ve mang
            desc = desc.split()
            # convert to lower case
            # Dua chu viet hoa ve chu thuong "trong mang"
            desc = [word.lower() for word in desc]
            # Loai bo cac ky tu dac biet trong mang
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            # remove tokens with numbers in them
            # isalpha kiểm tra các ký tự thuộc bảng chữ cái
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] =  ' '.join(desc)

# clean descriptions
clean_descriptions(descriptions)

# Lưu description xuống file
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

save_descriptions(descriptions, 'descriptions.txt')

# Lấy id ảnh tương ứng với dữ liệu train, test, dev
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

# load training dataset (6K)
filename = 'Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)

# Folder chứa dữ ảnh
images = 'Flickr8k/Flicker8k_Dataset/'
# Lấy lấy các ảnh jpg trong thư mục
img = glob.glob(images + '*.jpg')

# File chứa các id ảnh để train
train_images_file = 'Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
# Read the train image names in a set
# Hàm strip() xóa khoảng trắng hai đầu
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

# Create a list of all the training images with their full path names
train_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in train_images: # Check if the image belongs to training set
        train_img.append(i) # Add it to the list of train images

# File chứa các id ảnh để test
test_images_file = 'Flickr8k/Flickr8k_text/Flickr_8k.testImages.txt'
# Read the validation image names in a set# Read the test image names in a set
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# Create a list of all the test images with their full path names
test_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in test_images: # Check if the image belongs to test set
        test_img.append(i) # Add it to the list of test images

# Thêm 'startseq', 'endseq' cho chuỗi
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions

# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)

# Load ảnh, resize về khích thước mà Inception v3 yêu cầu.
# Đầu vào của nó là 1 cái ảnh có chiều là 229,229,3 
# Vì là 1 cái ảnh nên phải expand_dims để có chiều là (1,229,229,3)
def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

##### XÂY DỰNG MÔ HÌNH #####
# Load the inception v3 model
model = InceptionV3(weights='imagenet')

# Tạo model mới, bỏ layer cuối từ inception v3
model_new = Model(model.input, model.layers[-2].output)

def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

# Gọi hàm encode với các ảnh trong traning set sau đó encode và lưu vào file pkl
if(path.exists('Flickr8k/encoded_train_images.pkl') == False):
    start = time()
    encoding_train = {}
    for img in train_img:
        encoding_train[img[len(images):]] = encode(img)
    print("Time taken in seconds =", time()-start)
    # Lưu image embedding lại
    with open("Flickr8k/encoded_train_images.pkl", "wb") as encoded_pickle:
        dump(encoding_train, encoded_pickle)

# Gọi hàm encode với các ảnh trong test set sau đó encode và lưu vào file pkl
if(path.exists('Flickr8k/encoded_test_images.pkl') == False):
    start = time()
    encoding_test = {}
    for img in test_img:
        encoding_test[img[len(images):]] = encode(img)
    print("Time taken in seconds =", time()-start)
    #Save the bottleneck test features to disk
    with open("Flickr8k/encoded_test_images.pkl", "wb") as encoded_pickle:
        dump(encoding_test, encoded_pickle)

train_features = load(open("Flickr8k/encoded_train_images.pkl", "rb"))
print('Photos: train=%d' % len(train_features))

# Tạo list các training caption
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)


# Chỉ lấy các từ xuất hiện trên 10 lần
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

# Tạo một dict thứ tự các từ có trong vocab
ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

# Thêm một từ nữa để padding cho những câu không đủ độ dài với câu dài nhất
vocab_size = len(ixtoword) + 1 # Thêm 1 cho từ dùng để padding
vocab_size

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# Lấy caption có độ dài nhất
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# data generator cho việc train theo từng batch model.fit_generator()
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0

# Load Glove model
# File glove.6B dùng để vector hóa mỗi từ bằng một vector có độ dài = 200
glove_dir = 'glove.6B'
embeddings_index = {} # empty dictionary
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# Tạo ra ma trận vector các từ trong vocab
embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoix.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

# Build model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# Layer 2 dùng GLOVE Model nên set weight thẳng và không cần train
# Do dùng glove.6b để tạo một ma trận mã hóa với các từ nên ở layer này ko cần train
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False


name = "Image_Caption-{}".format(int(time.time()))
# Training model 
tensor_board = TensorBoard(log_dir='Graph/{}'.format(name))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
callbacks_list = [tensor_board]
model.optimizer.lr = 0.0001
epochs = 20
number_pics_per_bath = 2
steps = len(train_descriptions)//number_pics_per_bath


generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_bath)
model.fit_generator(generator, epochs = epochs, steps_per_epoch=steps, verbose=1, callbacks = callbacks_list)

# Lưu vào file h5
model.save_weights('./model_31.h5')