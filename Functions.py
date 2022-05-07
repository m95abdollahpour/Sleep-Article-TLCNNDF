import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy
import zlib
import pyedflib
import pyeeg
import math
import gc
from pywt import wavedec
from scipy import stats
from entropy import *  
import sys
import sklearn
from scipy.signal import butter, lfilter, freqz
from keras.preprocessing import image 
import scipy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from keras import backend as K
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)






def edfxRead(directory):

    
    """
    Read hypnogram files of Sleep EDF dataset (*edf)
    set integer values for sleep stages and remove unwanted epochs

    
    """
    f = open(directory)
    file = f.read()
    f.close()
    
    f = pyedflib.EdfReader(directory)
    h =  np.array(f.readAnnotations())
    del f
    labels = np.ones((len(h[2,:])))
    n = 0
    while (n < len(h[2,:])):
    
        
        if (h[2,n][12] == '?'):
            labels[n] = 6
        elif (h[2,n][12] == 'W'):
            labels[n] = 0
        elif (h[2,n][12] == 'e'):   # movement time
            labels[n] = 6
        elif (h[2,n][12] == 'R'):
            labels[n] = 5
        elif (h[2,n][12] == '4'):
            labels[n] = 3
        else:
            labels[n] = int(h[2,n][12])

        n =  n + 1
    
    pos = np.float32(np.array(h[0,:]))/30
    hlength = int(pos[len(pos)-1])
    n = 0
    Labels = np.ravel(30* np.ones((1, hlength)))

    
    while (n<len(pos) - 1):
        
        a = int(pos[n])
        b = int(pos[n+1])

        if (labels[n] == 0):
            Labels[a:b] = 0
        elif (labels[n] == 1):
            Labels[a:b] = 1
        elif (labels[n] == 2):
            Labels[a:b] = 2
        elif (labels[n] == 3):
            Labels[a:b] = 3
        elif (labels[n] == 4):
            Labels[a:b] = 4
        elif (labels[n] == 5):
            Labels[a:b] = 5
        elif (labels[n] == 6):
            Labels[a:b] = 6
    
        n = n + 1
        
    return Labels






def feature_extraction(signal, featurel_type,sub_epoch,fs):

    
    
    """
    Extract features from signals
    
    Parameters
    -----------
    signal: 1D array (input signal)
    
    fs: int (sampling frequency of input signal)
    
    sub_epoch: integer (divide epoch by this number then extract features from sub epochs)
    
    featurel_type: 'Wavelet1', 'PSD', 'nonlinear_stats_features'
    
    Returns
    ---------
    feature vectors
    
    """
    if featurel_type=='Wavelet1':
        epoch_count=int(len(signal)/(fs*30))
        f_num = 22
        f_vectors=np.ones((f_num*sub_epoch,epoch_count))
        m=0
        t=0
        while(m<epoch_count):
            i=0
            s=0
            while(i<sub_epoch):
                            t1=int(np.floor((fs*30)/sub_epoch))
                            EEG1=signal[t:t+t1]

                            cA, cD, cD1, cD2, cD3 = wavedec(EEG1, 'db4', level = 4)

                            b2 = kolmogorov(cD)
                            b3 = kolmogorov(cD1)
                            b33 = kolmogorov(cD2)
                            b4 = kolmogorov(cD3)
                            [b11, b12] = pyeeg.hjorth(cD)
                            [b13, b14] = pyeeg.hjorth(cD1)
                            [b17, b16] = pyeeg.hjorth(cD2)
                            [b19, b18] = pyeeg.hjorth(cD3)
                            b28 = scipy.stats.kurtosis(cD)
                            b29 = scipy.stats.kurtosis(cD1)
                            b30 = scipy.stats.kurtosis(cD2)
                            b31 = scipy.stats.kurtosis(cD3)
                            b46 = np.std(cA)  
                            b47 = np.std(cD)
                            b48 = np.std(cD1)
                            b49 = np.std(cD2)
                            b50 = np.std(cD3)
                            b55 = np.mean(abs(cA))
                            b56 = np.mean(abs(cD))
                            b57 = np.mean(abs(cD1))
                            b58 = np.mean(abs(cD2))
                            b59 = np.mean(abs(cD3))

                            B=[b46,b55,
                               b2,b11,b28,b47,b56,
                               b3,b13,b29,b48,b57,
                               b33,b17,b30,b49,b58,
                               b4,b19,b31 ,b50,b59]

                            f_vectors[s : s + f_num, m] = B
                            s = s  + f_num
                            t=t+t1
                            i=i+1
                    
            m=m+1


    if featurel_type=='PSD':  
        epoch_count=int(len(signal)/(fs*30))
        f_num = 34
        f_vectors=np.ones((f_num*sub_epoch,epoch_count))
        m=0
        t=0
        while(m<epoch_count):
            i=0
            s=0
            while(i<sub_epoch):
                            t1=int(np.floor((fs*30)/sub_epoch))
                            EEG1=signal[t:t+t1]
                            f ,p = scipy.signal.periodogram(EEG1, fs,  window = None, scaling='density')
                            p = p + 0.000001
                            p = 20*np.log10(p)
                            p = scipy.signal.decimate(p, 15)  
                            p = scipy.signal.decimate(p, 3)
                            f_vectors[s : s + f_num, m] = p
                            s = s  + f_num
                            t=t+t1
                            i=i+1

            m=m+1


    if featurel_type=='nonlinear_stats_features':
        epoch_count=int(len(signal)/(fs*30))
        f_num=5
        f_vectors=np.ones((f_num*sub_epoch,epoch_count))
        m=0
        t=0
        while(m<epoch_count):
            i=0
            s=0
            while(i<sub_epoch):
                            t1=int((fs*30)/sub_epoch)
                            EEG1=signal[t:t+t1]
                            b9 = kolmogorov(EEG1)
                            [b15, b16] = pyeeg.hjorth(EEG1) 
                            b12 = scipy.stats.kurtosis(EEG1)
                            b13 = np.mean(abs(EEG1))
                            B=[b9, b13, b12, b15, b16]
                            f_vectors[s : s + f_num, m] = B
                            s = s  + f_num
                            t = t + t1
                            i = i + 1
               
            m=m+1
    return f_vectors
      



#normalizing feature vector (mapping into [0.5,1.5] range)
def normal(feature_vector):
    
    """
    normalizing feature values into specific range (feature wise)
    
    """
    
    
    t=0
    while (t<len(feature_vector)):
        feature_vector[t,:]=np.interp(feature_vector[t,:] ,(feature_vector[t,:].min(),feature_vector[t,:].max()),(0.25,1.75))
        t=t+1
    return feature_vector



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y





#kolmogorov complexityyyyyyyy
def kolmogorov(data):
   l = float(len(data))
   compr = zlib.compress(data)
   c = float(len(compr))
   return c/l

   
 




#winsorising
def cut_amp(data,amp,direction='up'):
    
    if (direction == 'up'):
        i=0
        while (i<len(data)):
            
            if (data[i]>amp):
                data[i]=amp
                i=i+1
            elif (data[i]<-amp):
                data[i]=-amp
                i=i+1
            else:
                i=i+1
    elif (direction == 'down'):
        i=0
        while (i<len(data)):
            
            if (data[i]<amp):
                data[i]=amp
                i=i+1
            else:
                i=i+1
    return data
    
    


#location of each node on circular graph
def poss(angle,scale=1):
    
    ang = math.radians(angle)    #convert to radians!
    position = ( scale* math.cos(ang),  scale*math.sin(ang))
    return position





def noded_graph(series, node_count):
    g = nx.Graph()
    j=0
    while (j < node_count):
        
        g.add_node(series[j], pos = [series[j]*poss(j*360/node_count)[0],series[j]*poss(j*360/node_count)[1]])
        j=j+1
    return g





def hvisibility_graph1(series):

    
    """
    adds edges or connections between nodes with repsect to horizontal visibility algorithm
    
    Parameters
    -----------
    series: 1D array (input feature vector or time series)
    
    
    Returns
    ---------
    networkx horizontal visibility graph with node values and node locations to plot the graph
    """
    
    node_count = len(series)
    
    g = nx.Graph()
    j=0
    while (j < node_count):
        
        g.add_node(series[j],pos = [series[j]*poss(j*360/node_count)[0],series[j]*poss(j*360/node_count)[1]])
        j=j+1

    
    tseries = []
    n = 0
    for magnitude in series:
        tseries.append( (n, magnitude ) )
        n += 1
        

    for n in range(0,len(tseries)-1):
        (ta, ya) = tseries[n]
        (tb, yb) = tseries[n+1]
        g.add_edge(tseries[ta][1], tseries[tb][1])


    for a,b in combinations(tseries, 2):
       
        (ta, ya) = a
        (tb, yb) = b

        connect = True
        

        for tc, yc in tseries:

            if (ta<tc<tb):

                if yc>ya or yc>yb:
                    connect = False
                    
        if connect:
            g.add_edge(tseries[ta][1], tseries[tb][1])

    return g







