import numpy as np
import csv
import os

numbers = [6,9,3,5,8]



def createData(path,train_path,test_path,learn_count,test_count):
    training_data_list = []
    test_data_list = []
    count = np.zeros(10)
    with open(path, 'r') as file:
        for line in file:
            if int(line[0]) in numbers:
                numb = int(line[0])
                if count[numb] < learn_count:
                    training_data_list.append(line)
                elif count[numb] >= learn_count and count[numb]<(learn_count+test_count):
                    test_data_list.append(line)
                count[numb]+=1
    
    createFileData(train_path,training_data_list)
    createFileData(test_path,test_data_list)

    return True


def createFileData(path, data):
    with open(path, mode='w', newline='') as file:
        for line in data:
            file.write(line)


def readData(path):
    data = []
    if not(os.path.exists(path)):
        return []
    with open(path,'r') as file:
        for line in file:
            data.append(line.split(","))
    return data