from sklearn.metrics import roc_auc_score
from write_to_file import wirteauroc
from conformpredict import MLCP
from configparser import ConfigParser
import os
from  write_to_file  import wirte
from draw import drawfig
from mutilabelmetric import Mutimetric
from PTMLCP import TMLCP
from PT import TPMLCP
import copy
from ThreadWithReturnValue import *

def dealmain(output1,r1,test_y1,testregression1,a_y1,a_hat1,signficace):
    # ----------进行配置文件配置---------
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    class_names = cp["DEFAULT"].get("class_names").split(",")

    #---------进行深度copy--------------
    output=copy.deepcopy(output1)
    r=copy.deepcopy(r1)
    test_y=copy.deepcopy(test_y1)
    testregression=copy.deepcopy(testregression1)
    a_y=copy.deepcopy(a_y1)
    a_hat=copy.deepcopy(a_hat1)

    #------------输出这些数值------------
    varname=[test_y,testregression,a_y,a_hat]#把需要打印的类型放入数据，

    filename=['test_y.txt','testregression.txt','a_y.txt','a_hat.txt']#打印文件的文件名

    for i in range(len(filename)):
        dirpath=os.path.join(output,filename[i])#整合文件路径
        wirte(varname[i], dirpath)#写入文件中

    #-------cnn模型的auroc评估-------------
    comformist_log_path1 = os.path.join(output, "test01.log")
    print(f"** write log to {comformist_log_path1} **")
    wirteauroc(comformist_log_path1,test_y,testregression)


    numInstance = len(a_hat[0])#计算出总共有多少个测试样本
    comformist = MLCP(numInstance,output)#对mlcp实例化
    tcomformist = TMLCP(numInstance, output)
    ptcmformist = TPMLCP(numInstance, output)
    thread1 = ThreadWithReturnValue(target=comformist.prediction,args=(a_y,a_hat,test_y,testregression,signficace,r))
    thread2 = ThreadWithReturnValue(target=tcomformist.prediction,args=(a_y,a_hat,test_y,testregression,signficace,r))
    thread3 = ThreadWithReturnValue(target=ptcmformist.prediction1,args=(a_y,a_hat,test_y,testregression,signficace,r))
    thread1.start()
    thread2.start()
    thread3.start()

    lastpredict,accuary, truearray,nosurerate,nonearray,cnnaccuary,cnntruearray, cnnnosurerate, cnnnonearray = thread1.join()
    tlastpredict, taccuary, ttruearray, tnosurerate, tnonearray=thread2.join()
    ptlastpredict, ptaccuary, pttruearray, ptnosurerate, ptnonearray=thread3.join()
    # lastpredict, accuary, truearray, nosurerate, nonearray, cnnaccuary,\
    # cnntruearray, cnnnosurerate, cnnnonearray=comformist.prediction(a_y,a_hat,test_y,testregression,signficace,r)



    comformist_log_path = os.path.join(output, "test02.log")
    print(f"** write log to {comformist_log_path} **")
    wirteauroc(comformist_log_path, test_y, lastpredict)
    tcomformist_log_path = os.path.join(output, "test03.log")
    wirteauroc(tcomformist_log_path, test_y, tlastpredict)
    ptcomformist_log_path = os.path.join(output, "test04.log")
    wirteauroc(ptcomformist_log_path, test_y, ptlastpredict)

    # 画出可置信的图像
    drawfig(output, accuary, signficace, truearray, nosurerate, nonearray, class_names, cnnaccuary, cnntruearray,
            cnnnosurerate, cnnnonearray
            , taccuary, ttruearray, tnosurerate, tnonearray, ptaccuary, pttruearray, ptnosurerate, ptnonearray)
    # 多标记评估
    Mutimetric(lastpredict, tlastpredict, ptlastpredict, testregression, test_y, output)



    # tlastpredict,taccuary, ttruearray,tnosurerate,tnonearray = tcomformist.prediction(a_y,a_hat,test_y,testregression,signficace,r)







    # ptcmformist = TPMLCP(numInstance,output)
    # ptlastpredict,ptaccuary, pttruearray,ptnosurerate,ptnonearray = ptcmformist.prediction(a_y,a_hat,
    #                                                                                         test_y,testregression,signficace,r)






