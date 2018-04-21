import numpy as np
import copy
import os
from conformpredict import MLCP


class TPMLCP(MLCP):
   def __init__(self,numInstance,path,numclasses=14,count=0,anum=0):
       '''
       :param numInstance: number
       :param numclasses:类别

       '''
       MLCP.__init__(self,numInstance,path,numclasses=14,count=0,anum=0)


   def conformist(self,test_y,regression,r):
        '''
        计算每一个的奇异值
        :param regression:是个数据 list
        :return a: list 表示的是奇异值
        '''

        a =copy.deepcopy(regression)
        a1socre=[]
        a0socre=[]
        for j in range(self.numclasses):
            a1=[]
            a0=[]
            for i in range(len(regression[0])):

                if test_y[j][i]==1:
                     if regression[j][i]>0.5:
                        a[j][i] = 1 / (regression[j][i]+r[j])
                        a1.append(a[j][i])
                     else:
                        a[j][i] = 1 / (1 - regression[j][i]+r[j])
                        a1.append(a[j][i])
                else:
                     if regression[j][i]<0.5:
                        a[j][i] = 1 / (1-regression[j][i]+r[j])
                         # a[j][i] = 1 / (regression[j][i]+r[j])
                        a0.append(a[j][i])
                     else:
                         a[j][i] = 1 / (regression[j][i] + r[j])
                        #a[j][i] = 1 / (1-regression[j][i]+r[j])
                         a0.append(a[j][i])
            a1socre.append(np.array(a1))
            a0socre.append(np.array(a0))


        return a1socre,a0socre

    #-------表示y为１的概率的奇异值-----------进行奇异映射
   def evconformist(self,test_y,regression,r):
       '''
       :param regression:
       :param r:
       :return:
       '''

       a = copy.deepcopy(regression)

       for j in range(self.numclasses):

           for i in range(len(regression[0])):


                if test_y==1:
                    if regression[j][i] > 0.5:
                        a[j][i] = 1 / (regression[j][i] + r[j])
                    else:
                        a[j][i] = 1 / (1 - regression[j][i] + r[j])

                else:
                    if regression[j][i] < 0.5:
                        a[j][i] = 1 / (1 - regression[j][i] + r[j])

                    else:
                        a[j][i] = 1 / (regression[j][i] + r[j])

       return a


   def prediction1(self, a_y,a_regression,test_y, testregression,signficace, r):
       '''
    　　进行预测
       :param testregression: list
       :param devregression: list
       :param r: 网络敏感数
       :return: lastpredict :list
       '''
       lastpredictpath = os.path.join(self.path, "Ptlastvalue")
       testnum = len(testregression[0])
       devnum = len(a_regression[0])
       onearray = np.ones((testnum,1))
       zerosarray =np.zeros((testnum,1))
       # -----把所有可能类别进行遍历－－－－－－
       test_Y_zero = []
       test_y_one = []
       test_other_regression = []
       for i in range(self.numclasses):
           test_y_one.append(onearray)
           test_Y_zero.append(zerosarray)
           test_other_regression.append(onearray-testregression[i])

       #-----计算出所有的奇异值--------------


       print('pt的值｛r｝'.format(r))

       a1socre,a0socre = self.conformist(a_y,a_regression,r)#得奇异值
       y_one_ascore = self.evconformist(test_y_one,testregression,r)
       y_zero_ascore = self.evconformist(test_Y_zero,testregression,r)#为０时候的奇异值

       #-----初始化最大值最终预测------------
       initlastpredict=copy.deepcopy(testregression)

       #-----计算出p值,并进行预测---------------------

       lastpredict,p1tvalue,p0tvalue = super(TPMLCP,self).pvalue(a1socre,a0socre,initlastpredict,y_one_ascore,y_zero_ascore,
                                                                 testregression,test_other_regression)

       accuary,truearray ,nosurerate,nonearray= super(TPMLCP,self).signficance(p1tvalue,p0tvalue,test_y,signficace)

       #-------对cp-mcnn点预测的值进行写入-----------
       with open(lastpredictpath, 'w') as flie:

            for i in range(self.numclasses):  # 有多少列
                flie.write('第%d类:' % i)
                for j in range(len(lastpredict[i])):  # 遍历每一列中的每个元素
                    flie.write(str(lastpredict[i][j]))
                flie.write("\n")

       return lastpredict,accuary, truearray,nosurerate,nonearray






