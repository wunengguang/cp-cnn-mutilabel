import numpy as np
import copy
import os


class MLCP(object):
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

   def aconformist(self,test_y,regression,r):
        '''
        计算每一个的奇异值
        :param regression:是个数据 list
        :return a: list 表示的是奇异值
        '''
        a1socrepath = os.path.join(self.path, "a1score.log")
        a0socrepath= os.path.join(self.path, "a0score.log")
        path=[a1socrepath,a0socrepath]
        # test_a_path = os.path.join(self.path, "testa.log")
        # othertest_a_path = os.path.join(self.path, "othertesta.log")
        # path = [ascorepath ,test_a_path, othertest_a_path]


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

                        a[j][i] = 1 / (regression[j][i]+r[j])
                        a0.append(a[j][i])
                    else:

                         a[j][i] = 1 / (1-regression[j][i]+r[j])
                         a0.append(a[j][i])
            a1socre.append(np.array(a1))
            a0socre.append(np.array(a0))

        asocre=[a1socre,a0socre]
        with open(path[self.anum], 'w')as f:
            for j in range(self.numclasses):
                f.write('第%d类:' % j)
                for i in range(len(asocre[self.anum][j])):
                    f.write(str(asocre[self.anum][j][i]))
                f.write("\n")
        self.anum += 1

        return a1socre,a0socre

    #表示y为１的概率的奇异值
   def devconformist(self,regression,r):
       '''
       :param regression:
       :param r:
       :return:
       '''
       # test_a_path = os.path.join(self.path, "testa.log")
       # othertest_a_path = os.path.join(self.path, "othertesta.log")
       # path=[test_a_path,othertest_a_path]
       a = copy.deepcopy(regression)

       for j in range(self.numclasses):

           for i in range(len(regression[0])):

                a[j][i] = 1 / (regression[j][i] +r[j])

       return a

   #0概率的回归值奇异值


   #－－－－－－－计算出p值－－－－－－－－－－
   def pvalue(self,a1score,a0score,initlastpredict,devconforvalue,y_zero_score,devregression,dev_other_regression):

       pvaluepath = os.path.join(self.path, "pvalue.log")
       p1tvalue=copy.deepcopy(devregression)
       p0tvalue=copy.deepcopy(dev_other_regression)

       with open(pvaluepath, 'w')as f:
           for i in range(self.numclasses): #有多少列
               f.write('第%d类:' % i)
               for j in range(len(devconforvalue[i])):#遍历每一列中的每个元素
                   ctr1 = 0
                   other0 = 0

                   for l in range(len(a1score[i])):
                        if a1score[i][l] >=devconforvalue[i][j]:
                               ctr1 += 1

                   for l in range(len(a0score[i])) :
                        if a0score[i][l] >=y_zero_score[i][j]:
                               other0 += 1


                   p1tvalue[i][j] = (ctr1 / (len(a1score[i]) + 1))
                   p0tvalue[i][j] =(other0/(len(a0score[i]) + 1))
                   # f.write(f"{p1tvalue[i][j]}: {p0tvalue[i][j]}")
                   f.write(f"{p1tvalue[i][j]}:{p0tvalue[i][j]}")

                   if p1tvalue[i][j]>p0tvalue[i][j]:
                       initlastpredict[i][j]=1
                       # if devregression[i][j]>0.5:
                       #     initlastpredict[i][j]=devregression[i][j]
                       # else:
                       #     initlastpredict[i][j] = dev_other_regression[i][j]
                   else:
                       initlastpredict[i][j]=0
                       # if devregression[i][j]<0.5:
                       #     initlastpredict[i][j] = devregression[i][j]
                       # else:
                       #      initlastpredict[i][j]= dev_other_regression[i][j]

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
               # notsurerate=(feinull-count)/len(p1tvalue[i])
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


   def prediction(self, a_y,a_regression,test_y, testregression1,signficace, r):
       '''
    　　进行预测
       :param testregression: list
       :param devregression: list
       :param r: 网络敏感数
       :return: lastpredict :list
       '''
       lastpredictpath = os.path.join(self.path, "lastvalue")
       testregression=copy.deepcopy(testregression1)
       testnum = len(testregression[0])
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
       print(r)
       print('conformal输出值r{}'.format(r))
       a1score,a0score= self.aconformist(a_y,a_regression,r)#得奇异值
       y_one_ascore = self.devconformist(testregression,r)
       y_zero_ascore = self.devconformist(test_other_regression,r)#为０时候的奇异值

       #-----初始化最大值最终预测------------
       initlastpredict=copy.deepcopy(testregression)

       #-----计算出p值,并进行预测---------------------

       lastpredict,p1tvalue,p0tvalue = self.pvalue(a1score,a0score,initlastpredict,y_one_ascore,
                                 y_zero_ascore,testregression,test_other_regression)

       accuary,truearray ,nosurerate,nonearray= self.signficance(p1tvalue,p0tvalue,test_y,signficace)

       #-------mcnn  ------------------
       cnnaccuary, cnntruearray, cnnnosurerate, none1array =self.signficance(testregression,test_other_regression,test_y,signficace)


       with open(lastpredictpath, 'w') as flie:

            for i in range(self.numclasses):  # 有多少列
                flie.write('第%d类:' % i)
                for j in range(len(lastpredict[i])):  # 遍历每一列中的每个元素
                    flie.write(str(lastpredict[i][j]))
                flie.write("\n")


       return lastpredict,accuary, truearray,nosurerate,nonearray,cnnaccuary, cnntruearray, cnnnosurerate, none1array






