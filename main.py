from show import showData, showGraphError,  showGraphAccess, showConfusionMatrix
from data import createData, readData
from Perceptron import learnP;
from MLP import learnMLP;
from data import numbers
import random


train_data, test_data = 0,0
global_path = "./"
error,numb,chance = [],0,0


type = -1
do = -1
while(type != 1 and type != 2):
    print("1 - Perceptron, 2 - Multilayer Perceptron")
    type = int(input())



if type == 1:
    train_data = readData(global_path+"trainP.csv")
    test_data = readData(global_path+"testP.csv")
    print(len(train_data),len(test_data),"Traning Perceptorn")
    error,numb,chance = learnP(train_data, len(train_data))
else:
    train_data = readData(global_path+"trainMLP.csv")
    test_data = readData(global_path+"testMLP.csv")
    print(len(train_data),len(test_data), "Traning MLP")
    error,numb,chance = learnMLP(train_data, len(train_data))



while (do!=0):
    print("\n 1 - Show Train Data,\n 2 - Show Test Data,\n 3 - Graph Error,\n 4 - Graph Access,\n 5 - Test Error\n 6 - Test Access,\n 7 - What is it number,\n 8 - Confusion Matrix,\n 9 - Create learn and test data for .csv\n 0 - exit")
    do = int(input())
    if do == 1:
        showData(train_data)
    elif do == 2:
        showData(test_data)
    elif do == 3:
        if type == 1:
            showGraphError(error, 'График обучения персептрона')
        else:
            showGraphError(error, 'График обучения многослойного персептрона')
    elif do == 4:
        if type == 1:
            showGraphAccess(error, 'График обучения персептрона')
        else:
            showGraphAccess(error, 'График обучения многослойного персептрона')
    elif do == 5:
        if type == 1:
            error,numb,chance = learnP(test_data,len(test_data))
            showGraphError(error, 'График обучения персептрона')
        else:
            error,numb,chance = learnMLP(test_data,len(test_data))
            showGraphError(error, 'График обучения многослойного персептрона')
    elif do == 6:
        if type == 1:
            error,numb,chance = learnP(test_data,len(test_data))
            showGraphAccess(error, 'График обучения персептрона')
        else:
            error,numb,chance = learnMLP(test_data,len(test_data))
            showGraphAccess(error, 'График обучения многослойного персептрона')
    elif do == 7:

        numb = -1
        while(not(numb in numbers)):
            print("Get number for ",numbers,"from random test variant")
            numb = int(input())

        data = []
        for test in test_data:
            if(int(test[0])==numb):
                data.append(test)
        data = [data[random.randint(0,len(data))]]
        
        if(len(data)):
            if type == 1:
                error,numb,chance = learnP(data,1)
                print("IS: ", numb, "; chance: ", chance)
            else:
                error,numb,chance = learnMLP(data,1)
                print("IS: ", numb, "; chance: ", chance)
            showData(data,1,1)
    elif do == 8:
        true = []
        predicat = []
        if type == 1:
            for test in test_data:
                error,numb,chance = learnP([test],1)
                true.append(int(test[0]))
                predicat.append(numb)
                
        else:
            for test in test_data:
                error,numb,chance = learnMLP([test],1)
                true.append(int(test[0]))
                predicat.append(numb)

        showConfusionMatrix(true, predicat)
    elif do == 9:
        if type == 1:
            createData(global_path+'mnist_test.csv',global_path+'trainP.csv',global_path+'testP.csv',20,10)
        else:
            createData(global_path+'mnist_test.csv',global_path+'trainMLP.csv',global_path+'testMLP.csv',200,100)