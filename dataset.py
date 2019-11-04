import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
from sklearn.utils import shuffle

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

class Label_Extraction():
    '''将Staging和Grading tokenize
    Low 1 High 0
    NMIBC 1 MIBC 0
    再各自one-hot,前两个对应Grading, 后两个对应Staging
    [0, 1, 0, 1] ---> [Low, NMIBC]
    [1, 0, 1, 0] ---> [High, MIBC] '''
    def __init__(self, label_path):
        self.cancer_data = pd.read_csv(label_path)
        # self.imgs = os.listdir('data1')
        # self.imgs = sorted(self.imgs, key=lambda x: x.split('.')[0])
        # self.csv_imgs_name = self.cancer_data['Imagename']
        # for i in range(len(self.imgs)):
        #    if self.csv_imgs_name[i] != self.imgs[i].replace('.jpg', ''):
        #         print(self.imgs[i], self.csv_imgs_name[i])
        self.label_df = self.cancer_data[['Grading', 'Staging']]
        # # self.label_df = self.cancer_data[['Gradingnumber', 'Staingnumber']]
        # print(self.label_df)
        self.label_df = MultiColumnLabelEncoder(columns=['Grading', 'Staging']).fit_transform(self.label_df)
        self.y = np.array(self.label_df)
        '''self.one_hot_y = np.zeros((self.y.shape[0], 4))
        for i in range(self.y.shape[0]):
            # 如果Grading为High，则one-hot的为[1, 0]
            if self.y[i][0] == 0:
                self.one_hot_y[i][0] = 1
            # Grading为Low [0, 1]
            elif self.y[i][0] == 1:
                self.one_hot_y[i][1] = 1
            # Staging为MIBC [1, 0]
            if self.y[i][1] == 0:
                self.one_hot_y[i][2] = 1
            # NMIBC [0, 1]
            elif self.y[i][1] == 1:
                self.one_hot_y[i][3] = 1'''
            # y即两者串联起来 [Low, NMIBC] ---> [0, 1, 0, 1]

    #def __getitem__(self, index):
    #    return self.one_hot_y[index]

    def get_data(self):
        #return self.one_hot_y
        return self.y

class Label_Extraction_for_celoss():
    '''将Staging和Grading tokenize
    Low 1 High 0
    NMIBC 1 MIBC 0
    再各自one-hot,前两个对应Grading, 后两个对应Staging
    [0, 1, 0, 1] ---> [Low, NMIBC]
    [1, 0, 1, 0] ---> [High, MIBC]

    High MIBC --> 0
    High NMIBC--> 1
    Low MIBC --> 2
    Low NMIBC --> 3'''
    def __init__(self, label_path):
        self.cancer_data = pd.read_csv(label_path)
        self.label_df = self.cancer_data[['Gradingnumber', 'Staingnumber']]
        # self.label_df = MultiColumnLabelEncoder(columns=['Grading', 'Staging']).fit_transform(self.cancer_data)
        self.y = np.array(self.label_df)
        self.ce_y = np.zeros(self.y.shape[0])
        for i in range(self.y.shape[0]):
            # 如果Grading为High，Staging 为 MIBC
            if self.y[i][0] == 0 and self.y[i][1] == 0:
                self.ce_y[i] = 0
            # Grading 为 High，Staging 为NMIBC
            elif self.y[i][0] == 0 and self.y[i][1] == 1:
                self.ce_y[i] = 1
            # Low, MIBC
            elif self.y[i][0] == 1 and self.y[i][1] == 0:
                self.ce_y[i] = 2
            # Low NMIBC
            else:
                self.ce_y[i] = 3

class Label_Extraction_for_celoss():
    '''将Staging和Grading tokenize
    Low 1 High 0
    NMIBC 1 MIBC 0
    再各自one-hot,前两个对应Grading, 后两个对应Staging
    [0, 1, 0, 1] ---> [Low, NMIBC]
    [1, 0, 1, 0] ---> [High, MIBC]

    High MIBC --> 0
    High NMIBC--> 1
    Low MIBC --> 2
    Low NMIBC --> 3'''
    def __init__(self, label_path, rf=False):
        self.cancer_data = pd.read_csv(label_path)
        self.cancer_data = self.cancer_data[['ImageName', 'Grading', 'Staging']]
        self.label_df = MultiColumnLabelEncoder(columns=['Grading', 'Staging']).fit_transform(self.cancer_data)
        self.y = np.array(self.label_df[['Grading', 'Staging']])
        self.ce_y = np.zeros(self.y.shape[0])
        for i in range(self.y.shape[0]):
            # 如果Grading为High，Staging 为 MIBC
            if self.y[i][0] == 0 and self.y[i][1] == 0:
                self.ce_y[i] = 0
            # Grading 为 High，Staging 为NMIBC
            elif self.y[i][0] == 0 and self.y[i][1] == 1:
                self.ce_y[i] = 1
            # Low, MIBC
            elif self.y[i][0] == 1 and self.y[i][1] == 0:
                self.ce_y[i] = 2
            # Low NMIBC
            else:
                self.ce_y[i] = 3




    def __getitem__(self, index):
        return self.ce_y[index]

    def get_data(self):
        return self.ce_y

class Cancer_Pic(data.Dataset):

    def __init__(self, root_path, labels, transforms=None, train=True, test=False):
        '''划分数据集'''
        self.root_path = root_path
        self.train = train
        self.test = test
        self.labels = labels
        # 包含所有image名称的列表
        self.imgs = os.listdir(root_path)
        self.imgs = sorted(self.imgs, key=lambda x: x.split('.')[0])
        self.imgs_num = len(self.imgs)
        # self.imgs1, self.labels1 = self.imgs[:int(0.25 * self.imgs_num)], self.labels[:int(0.25 * self.imgs_num)]
        # self.imgs2, self.labels2 = self.imgs[int(0.25 * self.imgs_num):int(0.5 * self.imgs_num)], self.labels[int(0.25 * self.imgs_num):int(0.5 * self.imgs_num)]
        # self.imgs3, self.labels3 = self.imgs[int(0.5 * self.imgs_num):int(0.75 * self.imgs_num)], self.labels[int(0.5 * self.imgs_num):int(0.75 * self.imgs_num)]
        # self.imgs4, self.labels4 = self.imgs[int(0.75 * self.imgs_num):], self.labels[int(0.75 * self.imgs_num):]

        # ratio_train = int(0.7 * 0.25 * self.imgs_num)
        # ratio_test = int(0.85 * 0.25 * self.imgs_num)
        ratio_train = int(0.3 * self.imgs_num)
        ratio_test = int(0.15 * self.imgs_num)
        random_seed = int(1000 * time.time()) % 19491001
        if self.test:
            # self.imgs = self.imgs1[ratio_test:] + self.imgs2[ratio_test:] + self.imgs3[ratio_test:] + self.imgs4[ratio_test]
            self.imgs = self.imgs[ratio_test:ratio_train]
            self.labels = self.labels[ratio_test:ratio_train]

            # self.labels = np.array(list(self.labels1[ratio_test:]) + list(self.labels) + list(self.labels3[:ratio_train]) + list(self.labels4[:ratio_train]))
            self.imgs_num = len(self.imgs)
        elif train:
            # self.imgs = self.imgs1[:ratio_train] + self.imgs2[:ratio_train] + self.imgs3[:ratio_train] + self.imgs4[:ratio_train]
            # self.labels = np.array(list(self.labels1[:ratio_train]) + list(self.labels2[:ratio_train]) + list(self.labels3[:ratio_train]) + list(self.labels4[:ratio_train]))
            self.imgs = self.imgs[ratio_train:]
            self.labels = self.labels[ratio_train:]
            self.random_shuffle(random_seed)
            self.imgs_num = len(self.imgs)

        else:
            # self.imgs = self.imgs1[ratio_train:ratio_test] + self.imgs2[ratio_train:ratio_test] + self.imgs3[ratio_train:ratio_test] + self.imgs4[ratio_train:ratio_test]
            # self.labels = np.array(list(self.labels1[ratio_train:ratio_test]) + list(self.labels2[ratio_train:ratio_test]) + list(self.labels3[ratio_train:ratio_test]) + list(self.labels4[ratio_train:ratio_test]))
            self.imgs = self.imgs[:ratio_test]
            self.labels = self.labels[:ratio_test]
            self.imgs_num = len(self.imgs)

        if transforms is None:
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor()
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    # T.RandomHorizontalFlip(),
                    T.ToTensor()
                ])

    def random_shuffle(self, random_seed):

        self.imgs, self.labels = shuffle(self.imgs, self.labels, random_state=random_seed)
        # state = np.random.get_state()
        #self.imgs = np.random.permutation(self.imgs)
        #self.labels = np.random.permutation(self.labels)


    def __getitem__(self, index):
        """
        返回一张图片的数据
        """
        # np.random.seed(index)
        # idx = np.random.randint(0, len(self.imgs))
        img_path = self.imgs[index]
        # if self.test:
        label = self.labels[index]
        data = Image.open(os.path.join(self.root_path, img_path))
        data = self.transforms(data)
        return data, label

    def __len__(self):
        """
        返回数据集中所有图片的个数
        """
        return len(self.imgs)

#
# if __name__ == '__main__':
#     label_extra = Label_Extraction('label1.csv')
#     labels = label_extra.get_data()
#     train_dataset = Cancer_Pic('data1', labels, train=True)
#     trainloader = data.DataLoader(train_dataset, shuffle=True, num_workers=4)
#
#     for ii, (data, label) in enumerate(trainloader):
#         train()
    # data1, label1, img_name = cancer_pic.__getitem__(-1)
    # print(np.array(data1).shape)
    # print(label1)
    # print(img_name)