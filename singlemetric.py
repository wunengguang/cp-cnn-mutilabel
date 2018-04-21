from sklearn.metrics import hamming_loss,accuracy_score,coverage_error,\
    label_ranking_loss


from sklearn.metrics import zero_one_loss,f1_score,accuracy_score,average_precision_score,precision_recall_curve
import numpy as np
import copy
from sklearn.metrics import label_ranking_average_precision_score


def Mutimetric(lastpredict,testregression1,test_y1,output):

    testregression = copy.deepcopy(testregression1)
    test_y = copy.deepcopy(test_y1)
    for i in range(len(lastpredict)):
        lastpredict[i] = np.round(lastpredict[i])
        testregression[i] = np.round(testregression[i])

    #把数据进行合并，并且和平到第一列
    for i in range(len(lastpredict)):
        lastpredict[0] = np.c_[lastpredict[0], lastpredict[i]]
        test_y[0] = np.c_[test_y[0], test_y[i]]
        testregression[0] = np.c_[testregression[0], testregression[i]]
        if i==13:
            break

    lastpredict[0] = lastpredict[0][:, 1:15]
    test_y[0] = test_y[0][:, 1:15]
    testregression[0] = testregression[0][:,1:15]

    try:
        print(lastpredict[0].shape)
        #汉明损失
        shuzhi = hamming_loss(test_y[0],lastpredict[0])
        score = hamming_loss(test_y[0],testregression[0])
        # 子集准确率

        last_setaccuray = accuracy_score(test_y[0],lastpredict[0])
        test_setaccuray = accuracy_score( test_y[0],testregression[0])
        last_zero_one_loss = zero_one_loss(test_y[0],lastpredict[0])
        test_zero_one_loss =zero_one_loss(test_y[0],testregression[0])


        # f1_score
        lastf1_score = f1_score(test_y[0],lastpredict[0], average='macro')
        testf1_score = f1_score(test_y[0],testregression[0], average='macro')
        last_coverage_error = coverage_error(test_y[0],lastpredict[0])
        test_converage_error = coverage_error(test_y[0],testregression[0])


        #计算平均准确率
        # precision=dict()
        # recall=dict()
        # lastaccuracy_precision=dict()
        # for i in range(14):
        #     precision[i],recall[i]=precision_recall_curve(test_y[0][:,i],lastpredict[0][:,i])
        #     lastaccuracy_precision[i]=average_precision_score(test_y[0][:,i],lastpredict[0][:,i])
        #
        # precision["micro"],recall["micro"], _=precision_recall_curve(test_y[0].ravel(),lastpredict[0].ravel())
        #
        # lastaccuracy_precision["micro"]=average_precision_score(test_y[0],lastpredict[0],average='micro')
        #
        # print('average precision score,micro-averaged over all classes:{0:0.4f}'.format(lastaccuracy_precision['micro']))

        lastaverage = label_ranking_average_precision_score(test_y[0],lastpredict[0])
        testaverage = label_ranking_average_precision_score( test_y[0],testregression[0])
        predictlabel_ranking_loss = label_ranking_loss( test_y[0],lastpredict[0])
        test_ranking_loss = label_ranking_loss( test_y[0],testregression[0])
    except ValueError:
        print("metircerr:")
    finally:
        path =output+'/'+'meltmetric'
        with open(path,'w')as f:
            f.write(
                f"haminglosspredict:{shuzhi}haminglosstest:{score}\n")
            f.write(
                f"lastzero_one_loss:{last_zero_one_loss}testzero_one_loss:{test_zero_one_loss}\n")
            f.write(
                f"predictf1score:{lastf1_score}testf1_score:{testf1_score}\n")
            f.write(f"last_setaccuray:{last_setaccuray} test_setaccuray:{test_setaccuray}\n" )
            f.write(
                f"lastaverage:{lastaverage}testf1_score:{testaverage}\n")
            # f.write(str(lastaccuracy_precision)+"\n")
            f.write(
                f"last_coverage_error:{last_coverage_error}test_converage_error:{test_converage_error}\n")
            f.write(
                f"predictlabel_ranking_loss:{predictlabel_ranking_loss} test_ranking_loss :{test_ranking_loss}\n")
