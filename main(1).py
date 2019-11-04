import math
from config import opt
from data.dataset import Label_Extraction, Cancer_Pic, Label_Extraction_for_celoss
from torch.utils.data import DataLoader
from models import AlexNet, ResNet18, new_alex
import torch.optim as optim
import torch
import torchvision
import os
import time
import torch.nn as nn
# vgg = torchvision.models.vgg16()



# for k, v in model.named_parameters():
#     if k != 'fc.weight' or k != 'fc.bias':
#         v.requires_grad = False

def train(**kwargs):
    opt._parse(kwargs)
    '''model = torchvision.models.resnet18(pretrained=True)
    onechannel_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    onechannel_conv1.state_dict()['weight'] = (model.conv1.state_dict()['weight'][:, 0, :, :] + \
                                                 model.conv1.state_dict()['weight'][:, 1, :, :] + \
                                                 model.conv1.state_dict()['weight'][:, 2, :, :]) / 3
    model.conv1 = onechannel_conv1
    four_output_fc = nn.Linear(in_features=512, out_features=2, bias=True)
    model.fc = four_output_fc'''

    # model = new_alex()
    # model = AlexNet()
    model = ResNet18()

    if opt.use_gpu:
        model.cuda()

    label_extra = Label_Extraction('label1.csv')
    labels = label_extra.get_data()
    train_data = Cancer_Pic('data1', labels, train=True)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # label_extra = Label_Extraction('label1.csv')
    labels = label_extra.get_data()
    val_data = Cancer_Pic('data1', labels, train=False)

    # label_extra = Label_Extraction('label1.csv')
    labels = label_extra.get_data()
    test_data = Cancer_Pic('data1', labels, test=True)
    i = 0
    # for imgname, label in test_data:
    #    if i % 100 != 0:
    #        i += 1
    #        continue
    #    i += 1
    #    print(imgname, label)

    # raise ValueError
    # label_extra = Label_Extraction('label1.csv')
    lr = opt.lr
    weight_decay = opt.weight_decay
    # optimizer = model.get_optimizer(lr, weight_decay)
    # optimizer = optim.Adam(params=[p for p in model.layer1.parameters()] + [p for p in model.layer3.parameters()] + [p for p in model.layer4.parameters()] + [p for p in model.fc.parameters()],
    #                       lr=lr, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
    for epoch in range(opt.max_epoch):
        for batch_idx, (data, label) in enumerate(train_dataloader):
            # if batch_idx % 2 == 0:
            #     continue
            optimizer.zero_grad()
            output = model(data.cuda())
            # sftmax = nn.Softmax(dim=1)
            # grading = sftmax(output[:, :2])
            # staging = sftmax(output[:, 2:])
            label = label.float().cuda()
            # label = label.long().cuda()
            # 同时更新两列 softmax
            sigmoid  = nn.Sigmoid()
            output = sigmoid(output)
            '''output[:, 0], output[:, 1] = torch.exp(output[:, 0]) / (torch.exp(output[:, 0]) + torch.exp(output[:, 1])), \
                                         torch.exp(output[:, 1]) / (torch.exp(output[:, 0]) + torch.exp(output[:, 1]))
            output[:, 2], output[:, 3] = torch.exp(output[:, 2]) / (torch.exp(output[:, 2]) + torch.exp(output[:, 3])), \
                                         torch.exp(output[:, 3]) / (torch.exp(output[:, 2]) + torch.exp(output[:, 3]))'''
            # output = F.softmax(output, dim=1)
            # print('output', output)
            # print('label', label)
            output = output.cuda()

            # criterion = nn.CrossEntropyLoss()
            # loss = criterion(output, label)
            loss = torch.sum(- label[:, 0] * torch.log(output[:, 0] + 1e-10) - (1 - label[:, 0]) * torch.log(1 - output[:, 0] + 1e-10) \
                             - label[:, 1] * torch.log(output[:, 1] + 1e-10) - (1 - label[:, 1]) * torch.log(1 - output[:, 1] + 1e-10))
                           #  - label[:, 2] * torch.log(output[:, 2]) - (1 - label[:, 2]) * torch.log(1 - output[:, 2]) \
                            # - label[:, 3] * torch.log(output[:, 3]) - (1 - label[:, 3]) * torch.log(1 - output[:, 3]) )
            if math.isnan(loss.item()) :
                print('label', label)
                print('output', output)

            loss.backward()
            optimizer.step()
            if batch_idx % 30 == 0:
                print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                           100. * batch_idx / len(train_dataloader), loss.item()))
                #print(label, output)
            if batch_idx % 1500 == 0 and batch_idx!=0 :
                # val_celoss(model, train_data)
                val(model, val_data)
                val(model, test_data)
            # == 0 and batch_idx!=0:
                # val_celoss(model, val_data)
        #val(model, val_data)

        # if epoch >= 40:
        # val_celoss(model)
        prefix = './checkpoints/' + 'resnet' + '_'
        name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(model.state_dict(), name)

    # model.save()
    return model

def val_celoss(model, data_to_val):
    # val_data = Cancer_Pic('data1', labels, train=False)
    val_dataloader = DataLoader(data_to_val, opt.batch_size, num_workers=opt.num_workers)
    model.eval()
    correct = 0
    total = 0

    for ii, (data, label) in enumerate(val_dataloader):
        score = model(data.cuda())
        # print('score', score)
        predicted = torch.argmax(score, dim=1)

        # print('predicted', predicted.int())
        # print('label', label.int())
        total += label.size(0)
        correct += (predicted.int() == label.int().cuda()).sum().item()
    print('Test Accuracy of the model is {} %'.format(100 * correct / total))

def val_train_celoss(model):
    train_data = Cancer_Pic('data1', labels, train=True)
    train_dataloader = DataLoader(train_data, opt.batch_size, num_workers=opt.num_workers)
    model.eval()
    correct = 0
    total = 0

    for ii, (data, label) in enumerate(train_dataloader):
        score = model(data.cuda())
        # print('score', score)
        predicted = torch.argmax(score, dim=1)

        # print('predicted', predicted.int())
        # print('label', label.int())
        total += label.size(0)
        correct += (predicted.int() == label.int().cuda()).sum().item()
    print('Train Accuracy of the model is {} %'.format(100 * correct / total))

def val(model, data_to_val):
    # val_data = Cancer_Pic('data1', labels, train=False)
    val_dataloader = DataLoader(data_to_val, opt.batch_size, num_workers=opt.num_workers)
    model.eval()
    correct_Grading = 0
    correct_Staging = 0
    total = 0

    for ii, (data, label) in enumerate(val_dataloader):
        if ii % 2 == 0:
            continue
        label.cuda()
        with torch.no_grad():
            output = model(data.cuda())
        # print('score', score)
        # 同时更新两列
        '''output[:, 0], output[:, 1] = torch.exp(output[:, 0]) / (torch.exp(output[:, 0]) + torch.exp(output[:, 1])), \
                                     torch.exp(output[:, 1]) / (torch.exp(output[:, 0]) + torch.exp(output[:, 1]))
        output[:, 2], output[:, 3] = torch.exp(output[:, 2]) / (torch.exp(output[:, 2]) + torch.exp(output[:, 3])), \
                                     torch.exp(output[:, 3]) / (torch.exp(output[:, 2]) + torch.exp(output[:, 3]))'''
        sigmoid = nn.Sigmoid()
        output = sigmoid(output)
        predicted = (output >= 0.5)
        # print('predicted', predicted.int())
        # print('label', label.int())
        total += label.size(0)
        # print((predicted[:, 0] == label[:, 0]))
        # print((predicted[:, 0] == label[:, 0])[:, 0])
        # 前两列为Grading后两列为Staging，因为加了softmax，所以一行的两列之和为1，只需要判断第一列和第三列就能得出准确率
        correct_Grading += (predicted[:, 0].int() == label[:, 0].int().cuda()).sum().item()
        correct_Staging += (predicted[:, 1].int() == label[:, 1].int().cuda()).sum().item()
    print('Test Accuracy of the model on Grading is {} %'.format(100 * correct_Grading / total))
    print('Test Accuracy of the model on Staging is {} %'.format(100 * correct_Staging / total))

def val_our_model():
    model = ResNet18()
    model.load_state_dict(torch.load('./checkpoints/' + 'resnet' + '_' + '0607_14_30_57.pth'))
    model.eval()
    model.cuda()
    label_extra = Label_Extraction('label1.csv')
    labels = label_extra.get_data()
    print('Running model on the validation data:')
    val_data = Cancer_Pic('data1', labels, train=False)
    val(model, val_data)

    labels = label_extra.get_data()
    print('Running model on the test data:')

    test_data = Cancer_Pic('data1', labels, test=True)
    val(model, test_data)

if __name__ == '__main__':
    model = train()
   # val_our_model()
