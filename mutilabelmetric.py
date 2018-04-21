from sklearn.metrics import hamming_loss,accuracy_score,f1_score,average_precision_score,coverage_error,label_ranking_loss

import numpy as np
import copy

def Mutimetric(lastpredict,tlastpredict,ptlastpredict,testregression1,test_y1,output):

    testregression=copy.deepcopy(testregression1)
    test_y=copy.deepcopy(test_y1)
    for i in range(len(lastpredict)):
        lastpredict[i]=np.round(lastpredict[i])
        testregression[i]=np.round(testregression[i])
        tlastpredict[i]= np.round(tlastpredict[i])
        ptlastpredict[i] =np.round(ptlastpredict[i])


    for i in range(len(lastpredict)):
        lastpredict[0] = np.c_[lastpredict[0],lastpredict[i]]
        test_y[0] = np.c_[test_y[0],test_y[i]]
        testregression[0]=np.c_[testregression[0],testregression[i]]
        tlastpredict[0]=np.c_[tlastpredict[0],tlastpredict[i]]
        ptlastpredict[0]=np.c_[ptlastpredict[0],ptlastpredict[i]]

        if i==13:
            break

    lastpredict[0]=lastpredict[0][:,1:15]
    test_y[0]=test_y[0][:,1:15]
    testregression[0]=testregression[0][:,1:15]
    tlastpredict[0]=tlastpredict[0][:,1:15]
    ptlastpredict[0]=ptlastpredict[0][:,1:15]
    try:
        print(lastpredict[0].shape)
        #汉明损失
        shuzhi = hamming_loss(lastpredict[0],test_y[0])
        score = hamming_loss(testregression[0],test_y[0])
        thamingloss  =hamming_loss(tlastpredict[0],test_y[0])
        pthamingloss =hamming_loss(ptlastpredict[0],test_y[0])

        #子集准确率
        last_setaccuray = accuracy_score(lastpredict[0],test_y[0])
        test_setaccuray = accuracy_score(testregression[0],test_y[0])
        taccuracy_score = accuracy_score(tlastpredict[0], test_y[0])
        ptaccuracy_score =accuracy_score(ptlastpredict[0], test_y[0])




        # f1_score
        lastf1_score = f1_score(lastpredict[0],test_y[0],average='macro')
        testf1_score = f1_score(testregression[0],test_y[0],average='macro')
        tf1_score = f1_score(tlastpredict[0], test_y[0],average='macro')
        ptf1_score = f1_score(ptlastpredict[0], test_y[0],average='macro')

        last_coverage_error  = coverage_error(lastpredict[0],test_y[0])
        test_converage_error  =coverage_error(testregression[0],test_y[0])
        tcoverage_error = coverage_error(tlastpredict[0], test_y[0])
        ptcoverage_error = coverage_error(ptlastpredict[0],test_y[0])

        #平均准确率
        lastaverage = average_precision_score(lastpredict[0],test_y[0])
        testaverage =average_precision_score(testregression[0],test_y[0])
        taverage_precision_score = average_precision_score(tlastpredict[0], test_y[0])
        ptaverage_precision_score = average_precision_score(ptlastpredict[0], test_y[0])

        micrcolast =average_precision_score(lastpredict[0],test_y[0],average='micro')
        microtest = average_precision_score(testregression[0],test_y[0],average='micro')
        predictlabel_ranking_loss  = label_ranking_loss(lastpredict[0],test_y[0])
        test_ranking_loss =label_ranking_loss(testregression[0],test_y[0])
        tlabel_ranking_loss = label_ranking_loss(tlastpredict[0], test_y[0])
        ptlabel_ranking_loss = label_ranking_loss(ptlastpredict[0], test_y[0])
        # hinge_loss(lastpredict[0],test_y[0])
    except ValueError:
        print("metircerr:")
    finally:
        path =output+'/'+'meltmetric'
        with open(path,'w')as f:
            f.write(f"haminglosspredict:{shuzhi}haminglosstest:{score}thamingloss:{thamingloss}pthamingloss:{pthamingloss}\n")
            f.write(f"predictsetaccuary:{last_setaccuray}testsetaccuary:{test_setaccuray}taccuracy_score{taccuracy_score}ptaccuracy_score{ptaccuracy_score}")
            f.write(f"predictf1score:{lastf1_score}testf1_score:{testf1_score}tf1_score:{tf1_score}ptf1_score:{ptf1_score}\n")
            f.write(f"predictmarco:{lastaverage}testmarco:{testaverage}taverage_precision_score：{taverage_precision_score}ptaverage_precision_score:{ptaverage_precision_score}\n")
            f.write(f"predictmicro:{micrcolast}\ntestmicro:{microtest}\n")
            f.write(f"last_coverage_error:{last_coverage_error}test_converage_error:{test_converage_error}tcoverage_error:{tcoverage_error}ptcoverage_error:{ptcoverage_error}\n")
            f.write(f"predictlabel_ranking_loss:{predictlabel_ranking_loss} test_ranking_loss :{test_ranking_loss} tlabel_ranking_loss:{tlabel_ranking_loss}ptlabel_ranking_loss:{ptlabel_ranking_loss}\n")











