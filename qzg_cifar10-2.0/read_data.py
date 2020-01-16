import tensorflow as tf
import os


class_list = os.listdir('./train')
print(class_list)

root = './train/'
classname = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'ship', 'truck']

def child_path(father_path):
    return os.listdir(father_path)



file = open('train.txt', 'a')


for index, path in enumerate(class_list):
    for x in child_path(root + path):
        file.write(root + path + '/' + x + '\t' + str(index) + '\n')




