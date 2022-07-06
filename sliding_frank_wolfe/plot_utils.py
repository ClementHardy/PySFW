import openturns as ot
import numpy as np


def load_spectra(path):
    etude = ot.Study(path)
    etude.load()
    print('Chargement de tous les spectres')
    procSampleAll = ot.ProcessSample()
    etude.fillObject('procSampleAll', procSampleAll)
    print('Nbre total de spectres = ', procSampleAll.getSize())
    data = []

    for i in range(procSampleAll.getSize()):
        data.append(np.array(procSampleAll[i].getValues()).transpose()[0])

    data = np.array(data).transpose()
    times = np.flip(np.array(procSampleAll.getMesh().getVertices()), axis=1)
    temp_nb = len(times)
    times = times.reshape(temp_nb)
    return times, data
