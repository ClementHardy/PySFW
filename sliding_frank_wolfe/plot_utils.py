import openturns as ot
import numpy as np


def load_spectra(path):
    '''
    This function extracts from a database of spectra in xml format two tables corresponding respectively to
    the signals and the discretization points.

    Parameters
    ----------
    path : string,
    path of the database of spectra. The format of the spectra must be xml.

    Returns
    -------
    data : array, shape (p,n)
        array of n signals distretized on p points.

    times : array, shape(p,)
        array of size p corresponding to the points over which the signals are discretized.
    '''
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
