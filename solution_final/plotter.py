#!/usr/bin/env python3

import matplotlib.pyplot as plt

with open('results.dat') as f:
    lines = f.read().split('\n')

header = lines[0].split(', ')
data = {}
for item in header:
    data[item] = []

for line in lines[1:]:
    if not line:
        continue
    elements = [elem for elem in line.split(', ')]
    #elements[1] = int(elements[1])
    elements[3] = float(elements[3])
    for item, element in zip(header, elements):
        data[item].append(element)

def getEquals(dat, key, value):
    toRet = {}
    for item in header:
        toRet[item] = []
    for i, element in enumerate(dat[key]):
        if element == value:
            for item in header:
                toRet[item].append(dat[item][i])
    return toRet

plt.figure()
dat = getEquals(data, 'lbp', '4')
datValidate = getEquals(dat, 'test', 'validate')
datTest = getEquals(dat, 'test', 'test')
plt.bar(datValidate['kernel'], datValidate['iou'], label='Validate IoU')
plt.bar(datTest['kernel'], datTest['iou'], label='Test IoU')
plt.legend(loc='upper right')
plt.xlabel('Kernel')
plt.ylabel('IoU')
plt.title('Validation and Test IoU by Kernel')
plt.savefig('plots/kernelIoU.png')
#plt.show()

plt.figure()
dat = getEquals(data, 'kernel', 'rbf')
datValidate = getEquals(dat, 'test', 'validate')
datTest = getEquals(dat, 'test', 'test')
plt.bar(datValidate['lbp'], datValidate['iou'], label='Validate IoU')
plt.bar(datTest['lbp'], datTest['iou'], label='Test IoU')
plt.legend(loc='center right')
plt.xlabel('LBP setting')
plt.ylabel('IoU')
plt.title('Validation and Test IoU by LBP setting')
plt.savefig('plots/lbpIoU.png')
