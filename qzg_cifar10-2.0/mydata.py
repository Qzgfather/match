import numpy as np
from PIL import Image
f = open('./train.txt', 'r')

# for i in f.readlines():
#     train_images.append(i[:-1].split('\t')[0])
#     train_label.append(int(i[:-1].split('\t')[1]))
#
# f2 = open('../test.txt', 'r')
#
# test_images = []
# test_label = []
# for i in f2.readlines():
#     test_images.append(i[:-1].split('\t')[0])
#     test_label.append(int(i[:-1].split('\t')[1]))
data = []


def pro(path):
    global data
    images = path.split('\t')[0]
    labels = int(path.split('\t')[1][:-1])
    data.append(labels)
    images = np.array(Image.open(images))
    return images


train_images = [x for x in f.readlines()]
train_images = np.array(list(map(pro, train_images)))
train_labels = np.array(data)
f.close()
# data清零
data = []

f2 = open('./test.txt', 'r')
test_images = [x for x in f2.readlines()]
test_images = np.array(list(map(pro, test_images)))

test_labels = data
f2.close()


train = {'data': train_images, 'targets': train_labels}
test = {'data': test_images, 'targets': test_labels}

data = {
    'train': train,
    'valid': test
}


def get_data():
    return data
