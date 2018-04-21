import numpy as np
import copy
import os


class TMLCP(object):
   def __init__(self,numInstance,path,numclasses=14,count=0,anum=0):
       '''

       :param numInstance: number
       :param numclasses:类别
       '''
       self.numInstance = numInstance
       self.numclasses = numclasses
       self.path = path
       self.count=count
       self.anum=anum

   def aconformist(self,regression,r):
        '''
        计算每一个的奇异值
        :param regression:是个数据 list
        :return a: list 表示的是奇异值
        '''

        a =copy.deepcopy(regression)

        for j in range(self.numclasses):

            for i in range(len(regression[0])):

                a[j][i] = (1 - regression[j][i]) / (regression[j][i] + r[j])

        return a

   def pvalue(self,a,initlastpredict,devconforvalue,y_zero_score,devregression,dev_other_regression):

       pvaluepath = os.path.join(self.path, "MLCPpvalue.log")
       p1tvalue=copy.deepcopy(devregression)
       p0tvalue=copy.deepcopy(dev_other_regression)

       with open(pvaluepath, 'w')as f:
           for i in range(self.numclasses): #有多少列
               f.write('第%d类:' % i)
               for j in range(len(devconforvalue[i])):#遍历每一列中的每个元素
                   ctr1 = 0
                   other0 = 0
                   for l in range(len(a[i])):
                        if a[i][l] >=devconforvalue[i][j]:
                           ctr1 += 1

                   for l in range(len(a[i])) :
                        if a[i][l] >=y_zero_score[i][j]:
                           other0 += 1

                   p1tvalue[i][j] = (ctr1 / (len(a[i]) + 1))
                   p0tvalue[i][j] =(other0/(len(a[i]) + 1))

                   f.write(f"{p1tvalue[i][j]}:{p0tvalue[i][j]}")

                   if p1tvalue[i][j]>p0tvalue[i][j]:

                           initlastpredict[i][j]=1
                   else:
                           initlastpredict[i][j] = 0

               f.write('\n')



       return initlastpredict,p1tvalue,p0tvalue


   def signficance(self,p1tvalue,p0tvalue,test_y,signficace):
       '''

       :param p1tvalue:表是为一的一致性list内部有
       :param p0tvalue:表示为０的一的一致性程度 list
       :param testregression:list
       :param signficace:
       :return:onearray :array
       '''

       onesarray = np.ones((self.numclasses, len(signficace)))
       truearray =np.ones((self.numclasses, len(signficace)))
       favoriteratearray=np.ones((self.numclasses,len(signficace)))
       nonearray=np.ones((self.numclasses,len(signficace)))
       a = 0
       for z in signficace:
           signpredict = []
           truerate=[]
           favoriterate=[]
           ennone=[]
           for i in range(self.numclasses):
               hangpredic = []

               count=0
               en=0
               favorite=0

               for j in range(len(p1tvalue[i])):
                   everypredic = []
                   if p1tvalue[i][j]>z:
                       everypredic.append(1)
                   if p0tvalue[i][j]>z:
                       everypredic.append(0)

                   if test_y[i][j] in everypredic:
                       hangpredic.append(True)

                   if len(everypredic)==1:
                       count += 1
                       if test_y[i][j]==everypredic[0]:
                           favorite +=1

                   elif len(everypredic)==0:
                       en += 1

               nonerate=en/len(p1tvalue[i])
               accuaryrate=count/len(p1tvalue[i])
               accuary=len(hangpredic)/len(p1tvalue[i])
               favoritrate = favorite/len(p1tvalue[i])
               signpredict.append(accuary)
               truerate.append(accuaryrate)
               favoriterate.append(favoritrate)
               ennone.append(nonerate)
           onesarray[:,a] = np.array(signpredict)
           truearray[:,a] =np.array(truerate)
           favoriteratearray[:,a]=np.array(favoriterate)
           nonearray[:,a]=np.array(ennone)
           a  += 1

       return onesarray,truearray,favoriteratearray,nonearray


   def prediction(self, a_y,a_regression,test_y, testregression,signficace, r):
       '''
    　　进行预测
       :param testregression: list
       :param devregression: list
       :param r: 网络敏感数
       :return: lastpredict :list
       '''
       lastpredictpath = os.path.join(self.path, "MLCPlastvalue")
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
       print('PTMLCP的值｛r｝'.format(r))

       a= self.aconformist(a_regression,r)#得奇异值
       y_one_ascore = self.aconformist(testregression,r)
       y_zero_ascore = self.aconformist(test_other_regression,r)#为０时候的奇异值


           #-----初始化最大值最终预测------------
       initlastpredict=copy.deepcopy(testregression)

       #-----计算出p值,并进行预测---------------------

       lastpredict,p1tvalue,p0tvalue = self.pvalue(a,initlastpredict,y_one_ascore,
                                 y_zero_ascore,testregression,test_other_regression)

       accuary,truearray ,nosurerate,nonearray= self.signficance(p1tvalue,p0tvalue,test_y,signficace)

       #-----


       with open(lastpredictpath, 'w') as flie:

            for i in range(self.numclasses):  # 有多少列
                flie.write('第%d类:' % i)
                for j in range(len(lastpredict[i])):  # 遍历每一列中的每个元素
                    flie.write(str(lastpredict[i][j]))
                flie.write("\n")

       return lastpredict,accuary, truearray,nosurerate,nonearray






