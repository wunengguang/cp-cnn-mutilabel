import numpy as np
import test
import dev
import os
from singlealg import dealsingle



test_y,testregression = test.testpredict()

a_y,a_hat = dev.modelpredict()

rlist=[0,0.1,0.5]

i=0
for r in rlist:
    path='/media/thomas/文档/shiyan11/'+str(i)
    os.makedirs(path)
    dealsingle(path,r,test_y,testregression,a_y,a_hat)
    i=i+1