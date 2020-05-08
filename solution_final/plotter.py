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
    for i in range(4, len(elements)):
        elements[i] = float(elements[i])
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

def cvtMap(dat, key, mapping):
    for i, val in enumerate(dat[key]):
        if val in mapping:
            dat[key][i] = mapping[val]

plt.figure()
dat = getEquals(data, 'lbp', '0')
dat = getEquals(dat, 'objects-clutter', 'all-all')
datValidate = getEquals(dat, 'test', 'validate')
datTest = getEquals(dat, 'test', 'test')
plt.bar(datValidate['kernel'], datValidate['iou'], label='Validate IoU')
plt.bar(datTest['kernel'], datTest['iou'], label='Test IoU')
plt.legend(loc='upper left')
plt.xlabel('Kernel')
plt.ylabel('IoU')
plt.title('Validation and Test IoU by Kernel')
plt.savefig('plots/kernelIoU.png')
#plt.show()

plt.figure()
dat = getEquals(data, 'kernel', 'rbf')
dat = getEquals(dat, 'objects-clutter', 'all-all')
cvtMap(dat, 'lbp', {'0': '-(24, 8)', '1': '-(16, 4)', '2': '-(12, 2)', '3': '-(8, 1)', '4': '(full)'})
datValidate = getEquals(dat, 'test', 'validate')
datTest = getEquals(dat, 'test', 'test')
plt.bar(datValidate['lbp'], datValidate['iou'], label='Validate IoU')
plt.bar(datTest['lbp'], datTest['iou'], label='Test IoU')
plt.legend(loc='center right')
plt.xlabel('LBP setting')
plt.ylabel('IoU')
plt.title('Validation and Test IoU by LBP setting')
plt.savefig('plots/lbpIoU.png')

plt.figure()
dat = getEquals(data, 'kernel', 'rbf')
dat = getEquals(dat, 'lbp', '0')
datValidate = getEquals(dat, 'test', 'validate')
datTest = getEquals(dat, 'test', 'test')
plt.bar(datValidate['objects-clutter'], datValidate['iou'], label='Validate IoU')
plt.bar(datTest['objects-clutter'], datTest['iou'], label='Test IoU')
plt.legend(loc='upper left')
plt.xlabel('Train Source - Inference Source')
plt.ylabel('IoU')
plt.title('Validation and Test IoU by Train and Inference Image Source')
plt.savefig('plots/sourceIoU.png')
