import nonconformist
import os
import test
import dev

#参数设置
rlist=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
    ,0.5,0.5,0.5]]#设置不同的r值的参数
paramnum =0
signficace = [i / 100.0 for i in range(0, 98)]
#----------得出测试集的回归值 - -----
#--------得出验证集    的回归值 - ------－

test_y,testregression = test.testpredict()#得到测试集的label 和回归值

a_y,a_hat = dev.modelpredict()#得到dev集合的测试集和回归值


for r in rlist:#遍历每一个参数，并且为每个实验室的数据保存在不同的文件中
    outpath_dir='/media/thomas/文档/shiyan03/'+str(paramnum)

    if not os.path.isdir(outpath_dir):#表示如果文件不存在，
        os.makedirs(outpath_dir)#新建一个文件夹
    nonconformist.dealmain(outpath_dir,r,test_y,testregression,a_y,a_hat,signficace)#计算奇异值，测试不用的奇异函数在

    paramnum = paramnum+1