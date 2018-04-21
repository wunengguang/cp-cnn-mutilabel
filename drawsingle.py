import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np





def drawfig(outputpath,accaury,signficace,truearray,nosurerate,nonerate,class_names,cnnaccuary,cnntruearray, cnnnosurerate, cnnnonearray):
    signficace = np.array(signficace)
    onearray = np.ones((len(signficace), 1))
    t = onearray - signficace

    for i in range(accaury.shape[0]):
        plt.figure()
        plt.plot(t[0], t[0], 'b--', label='Baseline Calibration', linewidth=1)
        plt.plot(t[0], cnnaccuary[i, :], label='MCNN', color='brown', linewidth=1, linestyle='--', marker='')
        plt.plot(t[0], accaury[i, :], label='CP-MCNN', color='darkblue', linewidth=1, linestyle='--', marker='+')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(str(class_names[i]))
        plt.legend(loc='best')
        pigfure = outputpath + '/' + str(class_names[i]) + 'calibration'
        plt.savefig(pigfure)
        plt.close('all')



        plt.figure()
        plt.plot(t[0], nonerate[i, :], label='CP-MCNN empty prediction', color='darkblue', linewidth=1, linestyle='-',
                 marker='.')

        plt.plot(t[0], cnnnonearray[i, :], label='MCNN empty prediction', color='brown', linewidth=1, linestyle='--',
                 marker='+')
        plt.xlabel('Confidence')
        plt.ylabel('Rate')
        plt.title(str(class_names[i]))
        pigfure = outputpath + '/' + str(class_names[i]) + 'empty'
        # plt.axis([0,1,0,1])
        plt.legend(loc='best')
        # plt.grid()
        plt.savefig(pigfure)
        plt.close('all')

        plt.figure()
        plt.plot(t[0], nosurerate[i, :], label='CP-MCNN favorite prediction', color='darkblue', linewidth=1,
                 linestyle='-', marker='.')

        plt.plot(t[0], cnnnosurerate[i, :], label='MCNN favorite prediction', color='brown', linewidth=1,
                 linestyle='--',
                 marker='+')
        plt.xlabel('Confidence')
        plt.ylabel('Rate')
        plt.title(str(class_names[i]))
        pigfure = outputpath + '/' + str(class_names[i]) + 'favorite'

        plt.legend(loc='best')

        plt.savefig(pigfure)
        plt.close('all')

        plt.figure()
        plt.plot(t[0], truearray[i, :], label='CP-MCNN certain prediction', color='darkblue', linewidth=1,
                 linestyle='--',
                 marker='')
        plt.plot(t[0], cnntruearray[i, :], label='MCNN certain prediction', color='brown', linewidth=1, linestyle='--',
                 marker='+')
        plt.xlabel('Confidence')
        plt.ylabel('Rate')
        plt.title(str(class_names[i]))
        pigfure = outputpath + '/' + str(class_names[i]) + 'cetrain'
        # plt.axis([0,1,0,1])
        plt.legend(loc='best')
        # plt.grid()
        plt.savefig(pigfure)
        plt.close('all')