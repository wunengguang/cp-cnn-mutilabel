from sklearn.metrics import roc_auc_score
import numpy as np
from configparser import ConfigParser

config_file = "./config.ini"
cp = ConfigParser()
cp.read(config_file)
class_names = cp["DEFAULT"].get("class_names").split(",")

def wirte(y,path):
    with open(path,"w")as f:
        for i in range(len(y)):
            for j in range(len(y[i])):
                f.write(f'{y[i][j]} ')

            f.write('\n')


def wirteauroc(path,test_y,lastpredict):
    aurocs1 = []
    with open(path, "w") as f:
         for i in range(len(class_names)):
            try:
                score = roc_auc_score(test_y[i], lastpredict[i])
                aurocs1.append(score)
            except ValueError:
                score = 0
            f.write(f"{class_names[i]}: {score}\n")
         mean_auroc = np.mean(aurocs1)
         f.write("-------------------------\n")
         f.write(f"mean auroc: {mean_auroc}\n")