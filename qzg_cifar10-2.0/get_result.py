'''
验证模块
对模型进行实例化，然后去验证输入图片的类别，直接调用get_result函数传入路径耐心等待即可，一定要耐心，因为我没时间优化了现在已经2020年1月1日01:33:03了。
'''
from torchvision import transforms
from torch_backend import *
from PIL import Image

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'ship', 'truck']
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式
    def get_result(path):
      img = Image.open(path)  # 读取要预测的图片
      img = np.array(img)
      trans = transforms.Compose(
          [
              transforms.ToTensor(),
              transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
          ])
      img = trans(img)
      img = img.half().to(device).unsqueeze(0)
      output = model({'input': img})
      a = output['logits'].max(1)[1].cpu().numpy()[0]
      return classes[a]

    # get_result("./train/airplane/batch_1_num_29.jpg")

    f = open('result.csv', 'a')
    f2 = open('./test.txt', 'r')
    data = [x.split('\t')[0] for x in f2.readlines()]
    data = list(map(get_result, data))
    f.write(str(data))
    f.close()
    f2.close()
    '''
    上面的导出我直接把列表写入csv了，我在这里再整理一下，形成最终的结果，为啥直接写入列表？因为懒
    '''
    file = open('result.csv', 'r')
    f2 = open('./test.txt', 'r')
    f3 = open('f_result.csv', 'a')  # 最终结果
    test_file = [x.split('\t')[0].split('/')[-1] for x in f2.readlines()]  # 双重split最为致命
    result = file.read()
    result = result.replace('[', '').replace(']', '').replace('\'', '').split(',')
    print(len(test_file))
    print(len(result))
    print('完美')

    for i in range(len(test_file)):
        f3.write(test_file[i] + ',' + result[i] + '\n')

    print('恭喜最终结果生成完成！！！')