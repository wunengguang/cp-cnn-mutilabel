from conformpredict import MLCP
from configparser import ConfigParser
import os
from  write_to_file  import wirte
from write_to_file import wirteauroc
from drawsingle import drawfig
from singlemetric import Mutimetric




def dealsingle(output,r,test_y,testregression,a_y,a_hat):
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    class_names = cp["DEFAULT"].get("class_names").split(",")

    #
    signficace = [i/100.0 for i in range(0,98)]


    varname=[test_y,testregression,a_y,a_hat]
    filepath=[]
    filename=['test_y.txt','testregression.txt','a_y.txt','a_hat.txt']
    for i in filename:
        path=os.path.join(output,i)
        filepath.append(path)

    for i in range(len(filepath)):

        wirte(varname[i],filepath[i])


    comformist_log_path1 = os.path.join(output, "test01.log")


    print(f"** write log to {comformist_log_path1} **")

    wirteauroc(comformist_log_path1,test_y,testregression)

    numInstance = len(a_hat[0])  # 计算出总共有多少个测试样本

    comformist = MLCP(numInstance, output)  # 对mlcp实例化
    lastpredict, accuary, truearray, nosurerate, nonearray, cnnaccuary, \
    cnntruearray, cnnnosurerate, cnnnonearray = comformist.prediction(a_y, a_hat, test_y, testregression, signficace, r)
    print('第一轮lastpredict输出值r{}'.format(r))
    comformist_log_path = os.path.join(output, "test02.log")
    print(f"** write log to {comformist_log_path} **")
    wirteauroc(comformist_log_path, test_y, lastpredict)


    drawfig(output,accuary,signficace,truearray,nosurerate,nonearray,class_names,cnnaccuary,cnntruearray, cnnnosurerate, cnnnonearray)



    Mutimetric(lastpredict,testregression,test_y,output)