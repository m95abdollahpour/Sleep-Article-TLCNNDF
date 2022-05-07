import pyedflib
import numpy as np
import github_functions
from scipy import stats


directory =   "E:"

data = pyedflib.EdfReader(directory+"\\SC4001E0-PSG.edf")
eeg1=np.float32(data.readSignal(1))
eog1=np.float32(data.readSignal(2))
del data
h1 = edfxRead(directory+"\\SC4001EC-Hypnogram.edf")



data = pyedflib.EdfReader(directory+"\\SC4021E0-PSG.edf")
eeg2=np.float32(data.readSignal(1))
eog2=np.float32(data.readSignal(2))
del data
h2 = edfxRead(directory+"\\SC4021EH-Hypnogram.edf")


data = pyedflib.EdfReader(directory+"\\SC4031E0-PSG.edf")
eeg3=np.float32(data.readSignal(1))
eog3=np.float32(data.readSignal(2))
del data
h3 = edfxRead(directory+"\\SC4031EC-Hypnogram.edf")





data = pyedflib.EdfReader(directory+"\\SC4061E0-PSG.edf")
eeg4=np.float32(data.readSignal(1))
eog4=np.float32(data.readSignal(2))
del data
h4 = edfxRead(directory+"\\SC4061EC-Hypnogram.edf")



data = pyedflib.EdfReader(directory+"\\SC4071E0-PSG.edf")
eeg5=np.float32(data.readSignal(1))
eog5=np.float32(data.readSignal(2))
del data
h5 = edfxRead(directory+"\\SC4071EC-Hypnogram.edf")



data = pyedflib.EdfReader(directory+"\\SC4081E0-PSG.edf")
eeg6=np.float32(data.readSignal(1))
eog6=np.float32(data.readSignal(2))
del data
h6 = edfxRead(directory+"\\SC4081EC-Hypnogram.edf")





data = pyedflib.EdfReader(directory+"\\SC4141E0-PSG.edf")
eeg7=np.float32(data.readSignal(1))
eog7=np.float32(data.readSignal(2))
del data
h7 = edfxRead(directory+"\\SC4141EU-Hypnogram.edf")




data = pyedflib.EdfReader(  directory+"\\SC4151E0-PSG.edf")
eeg8=np.float32(data.readSignal(1))
eog8=np.float32(data.readSignal(2))
del data
h8 = edfxRead(directory+"\\SC4151EC-Hypnogram.edf")






data = pyedflib.EdfReader(directory+"\\SC4181E0-PSG.edf")
eeg9=np.float32(data.readSignal(1))
eog9=np.float32(data.readSignal(2))
del data
h9 = edfxRead(directory+"\\SC4181EC-Hypnogram.edf")







data = pyedflib.EdfReader(directory+"\\SC4211E0-PSG.edf")
eeg10=np.float32(data.readSignal(1))
eog10=np.float32(data.readSignal(2))
del data
h10 = edfxRead(directory+"\\SC4211EC-Hypnogram.edf")





data = pyedflib.EdfReader(directory+"\\ST7022J0-PSG.edf")
eeg11=np.float32(data.readSignal(1))
eog11=np.float32(data.readSignal(2))
del data

h11 = edfxRead(directory+"\\ST7022JM-Hypnogram.edf")
eeg11[eeg11 < -400] = -400
eeg11[eeg11 > 400] = 400
eeg11 = eeg11 - np.mean(eeg11)
eog11[eog11 < -600] = -600
eog11[eog11 > 600] = 600
eog11 = eog11 - np.mean(eog11)



data = pyedflib.EdfReader(directory+"\\ST7061J0-PSG.edf")
eeg12=np.float32(data.readSignal(1))
eog12=np.float32(data.readSignal(2))
del data
h12 = edfxRead(directory+"\\ST7061JR-Hypnogram.edf")
eeg12[eeg12 < -400] = -400
eeg12[eeg12 > 400] = 400
eeg12 = eeg12 - np.mean(eeg12)
eog12[eog12 < -600] = -600
eog12[eog12 > 600] = 600
eog12 = eog12 - np.mean(eog12)


data = pyedflib.EdfReader(directory+"\\ST7121J0-PSG.edf")
eeg13=np.float32(data.readSignal(1))
eog13=np.float32(data.readSignal(2))
del data
h13= edfxRead(directory+"\\ST7121JE-Hypnogram.edf")
eeg13[eeg13 < -400] = -400
eeg13[eeg13 > 400] = 400
eeg13 = eeg13 - np.mean(eeg13)
eog13[eog13 < -600] = -600
eog13[eog13 > 600] = 600
eog13 = eog13 - np.mean(eog13)



data = pyedflib.EdfReader(directory+"\\ST7081J0-PSG.edf")
eeg14=np.float32(data.readSignal(1))
eog14=np.float32(data.readSignal(2))
del data
h14 = edfxRead(directory+"\\ST7081JW-Hypnogram.edf")
eeg14[eeg14 < -400] = -400
eeg14[eeg14 > 400] = 400
eeg14 = eeg14 - np.mean(eeg14)
eog14[eog14 < -600] = -600
eog14[eog14 > 600] = 600
eog14= eog14 - np.mean(eog14)



data = pyedflib.EdfReader(directory+"\\ST7041J0-PSG.edf")
eeg15=np.float32(data.readSignal(1))
eog15=np.float32(data.readSignal(2))
del data
h15 = edfxRead(directory+"\\ST7041JO-Hypnogram.edf")
eeg15[eeg15 < -400] = -400
eeg15[eeg15 > 400] = 400
eeg15 = eeg15 - np.mean(eeg15)
eog15[eog15 < -600] = -600
eog15[eog15 > 600] = 600
eog15 = eog15 - np.mean(eog15)




data = pyedflib.EdfReader(directory+"\\ST7111J0-PSG.edf")
eeg16=np.float32(data.readSignal(1))
eog16=np.float32(data.readSignal(2))
del data
h16 = edfxRead(  directory+"\\ST7111JE-Hypnogram.edf")
eeg16[eeg16 < -400] = -400
eeg16[eeg16 > 400] = 400
eeg16 = eeg16 - np.mean(eeg16)
eog16[eog16 < -600] = -600
eog16[eog16 > 600] = 600
eog16 = eog16 - np.mean(eog16)




data = pyedflib.EdfReader(  directory+"\\ST7072J0-PSG.edf")
eeg17=np.float32(data.readSignal(1))
eog17=np.float32(data.readSignal(2))
del data
h17 = edfxRead(  directory+"\\ST7072JA-Hypnogram.edf")
eeg17[eeg17 < -400] = -400
eeg17[eeg17 > 400] = 400
eeg17 = eeg17 - np.mean(eeg17)
eog17[eog17 < -600] = -600
eog17[eog17 > 600] = 600
eog17 = eog17 - np.mean(eog17)




data = pyedflib.EdfReader(  directory+"\\ST7151J0-PSG.edf")
eeg18=np.float32(data.readSignal(1))
eog18=np.float32(data.readSignal(2))
del data
h18 = edfxRead(  directory+"\\ST7151JA-Hypnogram.edf")
eeg18[eeg18 < -400] = -400
eeg18[eeg18 > 400] = 400
eeg18 = eeg18 - np.mean(eeg18)
eog18[eog18 < -600] = -600
eog18[eog18 > 600] = 600
eog18 = eog18 - np.mean(eog18)







data = pyedflib.EdfReader(  directory+"\\ST7191J0-PSG.edf")
eeg19=np.float32(data.readSignal(1))
eog19=np.float32(data.readSignal(2))
del data
h19 = edfxRead(  directory+"\\ST7191JR-Hypnogram.edf")
eeg19[eeg19 < -400] = -400
eeg19[eeg19 > 400] = 400
eeg19 = eeg19 - np.mean(eeg19)
eog19[eog19 < -600] = -600
eog19[eog19 > 600] = 600
eog19 = eog19 - np.mean(eog19)



data = pyedflib.EdfReader(  directory+"\\ST7201J0-PSG.edf")
eeg20=np.float32(data.readSignal(1))
eog20=np.float32(data.readSignal(2))
del data
h20 = edfxRead(  directory+"\\ST7201JO-Hypnogram.edf")
eeg20[eeg20 < -400] = -400
eeg20[eeg20 > 400] = 400
eeg20 = eeg20 - np.mean(eeg20)
eog20[eog20 < -600] = -600
eog20[eog20 > 600] = 600
eog20 = eog20 - np.mean(eog20)




signals = {'eeg1': eeg1,'eeg2': eeg2,'eeg3': eeg3,'eeg4': eeg4,
           'eeg5': eeg5,'eeg6': eeg6,'eeg7': eeg7,'eeg8': eeg8,
           'eeg9': eeg9,'eeg10': eeg10,'eeg11': eeg11,'eeg12': eeg12,
           'eeg13': eeg13,'eeg14': eeg14,'eeg15': eeg15,'eeg16': eeg16,
           'eeg17': eeg17,'eeg18': eeg18,'eeg19': eeg19,'eeg20': eeg20,}

hypnograms = {'h1': h1,'h2': h2,'h3': h3,'h4': h4,
           'h5': h5,'h6': h6,'h7': h7,'h8': h8,
           'h9': h9,'h10': h10,'h11': h11,'h12': h12,
           'h13': h13,'h14': h14,'h15': h15,'h16': h16,
           'h17': h17,'h18': h18,'h19': h19,'h20': h20,}


keys = list(signals)

se = 1
ft = 'PSD'
fv_eeg = {}
for i in range(20):
    fv_eeg["fv_eeg{0}".format(i)] = feature_extraction(signal = signals[keys[i]], featurel_type = ft, sub_epoch=se, fs=100  )
    print (i)
    
    


keys = list(fv_eeg)
h_keys = list(hypnograms)

for i in range(20):
    fv_eeg[keys[i]] = fv_eeg[keys[i]][:,:len(hypnograms[h_keys[i]])]





fv1 = np.concatenate((fv_eeg[keys[0]],fv_eeg[keys[1]],fv_eeg[keys[2]],fv_eeg[keys[3]],fv_eeg[keys[4]],
                         fv_eeg[keys[5]],fv_eeg[keys[6]],fv_eeg[keys[7]],fv_eeg[keys[8]],
                         fv_eeg[keys[9]],fv_eeg[keys[10]],fv_eeg[keys[11]],fv_eeg[keys[12]],
                         fv_eeg[keys[13]],fv_eeg[keys[14]],fv_eeg[keys[15]],fv_eeg[keys[16]],
                         fv_eeg[keys[17]],fv_eeg[keys[18]],fv_eeg[keys[19]]), axis = 1)


h=np.concatenate((h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18,h19,h20), axis=0)

h[h == 5] = 4
POS = np.where(h > 5)
fv1 = np.delete(fv1,POS,1)







se = 1
ft = 'nonlinear_stats_features'
keys = list(signals)
fv_eeg = {}
for i in range(20):
    fv_eeg["fv_eeg{0}".format(i)] = feature_extraction(signal = signals[keys[i]], featurel_type = ft, sub_epoch=se, fs=100  )
    print (i)




keys = list(fv_eeg)
h_keys = list(hypnograms)

for i in range(20):
    fv_eeg[keys[i]] = fv_eeg[keys[i]][:,:len(hypnograms[h_keys[i]])]








fv2 = np.concatenate((fv_eeg[keys[0]],fv_eeg[keys[1]],fv_eeg[keys[2]],fv_eeg[keys[3]],fv_eeg[keys[4]],
                         fv_eeg[keys[5]],fv_eeg[keys[6]],fv_eeg[keys[7]],fv_eeg[keys[8]],
                         fv_eeg[keys[9]],fv_eeg[keys[10]],fv_eeg[keys[11]],fv_eeg[keys[12]],
                         fv_eeg[keys[13]],fv_eeg[keys[14]],fv_eeg[keys[15]],fv_eeg[keys[16]],
                         fv_eeg[keys[17]],fv_eeg[keys[18]],fv_eeg[keys[19]]), axis = 1)



fv2 = np.delete(fv2,POS,1)





se = 1
ft = 'nonlinear_stats_features'
keys = list(signals)
fv_eog = {}
for i in range(20):
    fv_eog["fv_eog{0}".format(i)] = feature_extraction(signal = signals[keys[i]], featurel_type = ft, sub_epoch=se, fs=100  )
    print (i)





keys = list(fv_eog)
h_keys = list(hypnograms)

for i in range(20):
    fv_eog[keys[i]] = fv_eog[keys[i]][:,:len(hypnograms[h_keys[i]])]





fvo2 = np.concatenate((fv_eog[keys[0]],fv_eog[keys[1]],fv_eog[keys[2]],fv_eog[keys[3]],fv_eog[keys[4]],
                         fv_eog[keys[5]],fv_eog[keys[6]],fv_eog[keys[7]],fv_eog[keys[8]],
                         fv_eog[keys[9]],fv_eog[keys[10]],fv_eog[keys[11]],fv_eog[keys[12]],
                         fv_eog[keys[13]],fv_eog[keys[14]],fv_eog[keys[15]],fv_eog[keys[16]],
                         fv_eog[keys[17]],fv_eog[keys[18]],fv_eog[keys[19]]), axis = 1)


fvo2 = np.delete(fvo2,POS,1)



se = 1
ft = 'Wavelet1'
keys = list(signals)
fv_eog = {}
for i in range(20):
    fv_eog["fv_eog{0}".format(i)] = feature_extraction(signal = signals[keys[i]], featurel_type = ft, sub_epoch=se, fs=100  )
    print (i)



keys = list(fv_eog)
h_keys = list(hypnograms)

for i in range(20):
    fv_eog[keys[i]] = fv_eog[keys[i]][:,:len(hypnograms[h_keys[i]])]




fvo = np.concatenate((fv_eog[keys[0]],fv_eog[keys[1]],fv_eog[keys[2]],fv_eog[keys[3]],fv_eog[keys[4]],
                         fv_eog[keys[5]],fv_eog[keys[6]],fv_eog[keys[7]],fv_eog[keys[8]],
                         fv_eog[keys[9]],fv_eog[keys[10]],fv_eog[keys[11]],fv_eog[keys[12]],
                         fv_eog[keys[13]],fv_eog[keys[14]],fv_eog[keys[15]],fv_eog[keys[16]],
                         fv_eog[keys[17]],fv_eog[keys[18]],fv_eog[keys[19]]), axis = 1)



fvo = np.delete(fvo,POS,1)
h = np.delete(h,POS,0)



fv2 = np.concatenate((fvo2,fv2), axis = 0)





# removing outlier epochs using z-score

fv11=normal(fv1)
fvoo = normal(fvo)
fv22 = normal(fv2)
hh = h

z = np.abs(stats.zscore(fv11[2]))
z1 = np.array(np.where(z > 4))

z = np.abs(stats.zscore(fvoo[0]))
z2 = np.array(np.where(z > 10))

z = np.abs(stats.zscore(fvoo[2]))
z3 = np.array(np.where(z > 4))

z = np.abs(stats.zscore(fvoo[9]))
z4 = np.array(np.where(z > 15))

z = np.abs(stats.zscore(fvoo[17]))
z5 = np.array(np.where(z > 5))

z = np.abs(stats.zscore(fvoo[18]))
z6 = np.array(np.where(z > 6))

z = np.abs(stats.zscore(fv22[4]))
z7 = np.array(np.where(z > 10))

z = np.abs(stats.zscore(fv22[7]))
z8 = np.array(np.where(z > 15))

z = np.abs(stats.zscore(fv22[9]))
z9 = np.array(np.where(z > 15))

outliers=np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9), axis=1)

fv11 = np.delete(fv11,np.reshape(outliers, (-1, 1)),1)
fvoo = np.delete(fvoo,np.reshape(outliers, (-1, 1)),1)
fv22 = np.delete(fv22,np.reshape(outliers, (-1, 1)),1)
hh = np.delete(hh,np.reshape(outliers, (-1, 1)),0)

fv11=normal(fv11)
fvoo = normal(fvoo)
fv22 = normal(fv22)


z = np.abs(stats.zscore(fv22[7]))
z24 = np.array(np.where(z > 16))

z = np.abs(stats.zscore(fv22[5]))
z23 = np.array(np.where(z > 5))


z = np.abs(stats.zscore(fv22[2]))
z22 = np.array(np.where(z > 13))

z = np.abs(stats.zscore(fv22[1]))
z21 = np.array(np.where(z > 6))

z = np.abs(stats.zscore(fvoo[17]))
z20 = np.array(np.where(z > 12))

z = np.abs(stats.zscore(fvoo[18]))
z19 = np.array(np.where(z > 5))

z = np.abs(stats.zscore(fvoo[12]))
z18 = np.array(np.where(z > 9))


z = np.abs(stats.zscore(fvoo[7]))
z17 = np.array(np.where(z > 4))

z = np.abs(stats.zscore(fvoo[2]))
z11 = np.array(np.where(z > 4))

z = np.abs(stats.zscore(fvoo[4]))
z12 = np.array(np.where(z > 14))

z = np.abs(stats.zscore(fvoo[14]))
z13 = np.array(np.where(z > 20))

z = np.abs(stats.zscore(fvoo[19]))
z14 = np.array(np.where(z > 18))

z = np.abs(stats.zscore(fv22[2]))
z15 = np.array(np.where(z > 12))

z = np.abs(stats.zscore(fvoo[9]))
z16 = np.array(np.where(z > 16))

outliers=np.concatenate(( z11, z12,z13,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24), axis=1)

fv11 = np.delete(fv11,np.reshape(outliers, (-1, 1)),1)
fvoo = np.delete(fvoo,np.reshape(outliers, (-1, 1)),1)
fv22 = np.delete(fv22,np.reshape(outliers, (-1, 1)),1)
hypn = np.delete(hh,np.reshape(outliers, (-1, 1)),0) #hypnogram (labels)

fv11=normal(fv11)
fvoo = normal(fvoo)
fv22 = normal(fv22)


v=0
while(v<len(fvoo)):
    if (np.mean(fvoo[v])<1.5):
        fvoo[v]=-fvoo[v]
        v = v+1
        
    else:
        v=v+1
            
v=0
while(v<len(fv22)):
    if (np.mean(fv22[v])<1.5):
        fv22[v]=-fv22[v]
        v = v+1
        
    else:
        v=v+1
        
        
        
#final feature vectors
fvoo = normal(fvoo)
fv11 = normal(fv11)
fv22 = normal(fv22)

fv22=np.concatenate((fvoo,fv22),axis = 0)










