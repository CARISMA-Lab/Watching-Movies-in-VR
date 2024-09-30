# Gary Bente, 2023
# This program requires the raw data to be postioned in a subdirectory of the current folder
# named "results". Within this folder it requires two subdirectories 'VR' and 'TV' that hold the 
# data of the two experimental groups. 

import datetime
import heartpy as hp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.stats as stats
from sklearn import preprocessing
import pandas as pd
from os import listdir
import os
from os.path import isfile, join
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import butter,lfilter
import warnings
import math
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("error")
np.set_printoptions(edgeitems=300)
np.set_printoptions(suppress=True)

"""
global sample_rate
global cond
global c1_flag
global c2_flag
global totalN
global FoID

totalN=0
"""

def rep_nan(arr):
    #replace nans with last valid precdent value
    # Create a copy of the input array to avoid modifying the original array
    global FoID
    result = np.copy(arr)
    last_valid_value = None
    for i, value in enumerate(result):
        if np.isnan(value):
            # If the current value is NaN, update it with the last valid value
            result[i] = last_valid_value
        else:
            # If the current value is not NaN, update the last valid value
            last_valid_value = value
    return result

def baseline(b,base):
    out=b-base
    return out

def normalize(dd):
    dd=np.asarray(dd)
    dd=dd.reshape(-1,1)
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(dd)
    return scaled_data

def z_trans(d):
    m=np.mean(d)
    s=np.std(d)
    #x=(d-m)/s
    x=d/s
    return x

def detrend(b):
    
    # Simulate a simple linear time series
    x = np.arange(0, len(b))  # Replace this with your actual x-values
    y = b  # Replace this with your actual y-values
    #print (len(x),len(y))
    # Fit a linear regression model
    coefficients = np.polyfit(x, y, 1)
    trend = np.polyval(coefficients, x)
    
    # Calculate the detrended series
    detrended_series = y - trend
    return detrended_series

def filter_it(b,f):
    global sample_rate

    dat=b.values.flatten()    
    output=  rep_nan(dat)
    order = 2
    #cutoff=0.1
    cutoff_frequency = 1.0  #Hz 
    cutoff=cutoff_frequency / (sample_rate / 2)   #percentage of Nyquist frequency == sample_rate/2
    
    if f==0:
        output= butter_lowpass_filter(output, cutoff, sample_rate, order)
        return output
    #output=output.reshape(-1,1)
    if f==1:
        try:
            output= detrend(output)
        except:
            pass
        #print (output)
        #output=normalize(output)
    output=baseline(output,np.mean(output[300:3600]))
    output=z_trans(output)
    output= butter_lowpass_filter(output, cutoff, sample_rate, order)
  

    return output
pList=['hr',
'RR_list',
'peaklist',
'RR_indices',
'binary_peaklist',
'RR_masklist',
'RR_list_cor']

path=os.path.dirname(os.path.abspath(__file__))+'/'


def dt(unix_time_ms):
    unix_time_seconds = int(unix_time_ms)
    microseconds = int((unix_time_ms - unix_time_seconds) * 1000000)
    dt_object = datetime.datetime.fromtimestamp(unix_time_seconds)
    micros_formatted = f"{microseconds:06d}"
    dt_object_with_micros = dt_object.replace(microsecond=microseconds)
    date_time_str = dt_object_with_micros.strftime('%Y-%m-%d %H:%M:%S') + f".{micros_formatted}"
    return date_time_str[-15:-3]

def minmax(df):
    d1 = df['DataA'].interpolate().values
    d2 = df['DataB'].interpolate().values
    x = d1
    y = d2
    dist=50
    peaks1, _ = find_peaks(x, height=0, distance =dist)
    peaks2, _ = find_peaks(y, height=0, distance =dist)
    poks1, _ = find_peaks(-x, height=0, distance =dist)
    poks2, _ = find_peaks(-y, height=0, distance =dist)

def interpol(dat,fl):
    #print (dat)
    a = pd.Series(dat)
    if fl==1:
        a = a.mask(a < 500)
    else:
        a.replace(0, np.NaN, inplace=True)
    a.interpolate()
    #print ( np.asarray(a))
    return np.asarray(a)

def get_res(tt,a):
#print (tt)
    sz=tt[-1]+1
    arr=np.zeros
    arr = np.empty((sz))
    arr[:] = -1
    
    for i,t in enumerate (tt):
        arr[t] =int(a[i])
    old=0
    for i in range(len(arr)):
        if arr[i]==-1 :
            arr[i]=old
        else:
            old=arr[i]
    dl=[]
    tl=[]
    z=0
    arr_int=interpol(arr)    
    
    for i in range (0,len(arr),40):
        tl.append(z)
        dl.append(arr[i])
        z+=40
    
    arr=np.asarray(dl)    
    tir=np.asarray(tl)
    return arr,tir,arr_int    
    
def resample (tt,G,P,S):
     G,t,ain=get_res(tt,G)
     P,t,ain=get_res(tt,P)
     S,t,ain=get_res(tt,S)
     return t,G,P,S,ain

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def median_filter(df,wi):
    df.rolling(window=wi,center=True).median()
    return df

def getPeaks(df):

    d = df['PPG'].interpolate().values
    dist=15
    peaks, _ = find_peaks(d, height=0, distance =dist)
    return d,peaks

def detect_outlier(data_1):
    outliers = []
    threshold = 1.5
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)
    for y in data_1:
        z_score = (y - mean_1) / std_1
        if z_score > threshold or z_score < -threshold:
            outliers.append(y)
    return outliers

def npmean(a):
    if a.sum==0:
        return 0
    else:
        return a[a!=0].mean()
def npstd(a):
    if a.sum==0:
        return 0
    else:
        return a[a!=0].std()

def get_mil(x):
    #print (x)
    hh= int(x[0:2])*3600000
    mi= int(x[3:5])*60000
    se= int(x[6:8])*1000
    mil= int(x[9:12])
    tx=int((hh+mi+se+mil))
    return tx        

def fisher_transform(r):
    """Perform Fisher transformation on correlation coefficient."""
    return round(0.5 * np.log((1 + r) / (1 - r)),3)

def inverse_fisher_transform(z):
    """Inverse Fisher transformation to get correlation coefficient back."""
    e_z = np.exp(2 * z)
    return (e_z - 1) / (e_z + 1)

def corr(df1,df2):
    r, p = stats.pearsonr(df1[~np.isnan(df1)],df2[~np.isnan(df2)])
    if r==1:
        r=.99999
    if r==-11:
        r=-.99999
    rz = fisher_transform (r)
    
    return r,rz,p

def ISC(dats):
    datseg=np.asarray(dats)
    y=datseg.shape[1]

    rsum=0
    rx=0
    psum=0
    #now for all combinations of subjects and group --> ISC
    
    for col in range(y):
           b=datseg[:,col]
           c = np.delete(datseg, col, 1)  # delete individualcolumn of C
           
           #b=resample(b,25)
           #c=resample(c,25)
           
           d=c.mean(axis=1)
           #check for constant time series exclude from corrs 
           result = np.all(b == b[0])
           #print (result)
           if not result:
               #draw_res(b,d)
               #b[0]=b[0]+0.0001
               
               r,rz,p=corr (b,d)
               rsum=rsum+rz
               p+=.00001
               psum=psum+math.log(p)
               rx+=1
               #when summed up divde by n and re transform

    rr=inverse_fisher_transform(rsum/rx)
    rs=round(rr,2)
    pp=-2*(psum/rx)
    return rs,pp

def show_it_err(m1,m2,e1,e2,raw1,raw2,tL,mark,label): #label,lab,mrk1,mrk2,mrk3,mrk4,crow,caws,Vx,sam):
    f,ax = plt.subplots(figsize=(18,3))
    alpha_fill=.2
    x1 = np.arange(len(m1))
    x2 = np.arange(len(m2))

    plt.plot( m1, label='VR',linewidth=2)
    ymin = m1 - e1
    ymax = m1 + e1
    col=plt.gca().lines[-1].get_color()
    ax.fill_between(x1, ymax, ymin, color=col,alpha=alpha_fill)
 
    plt.plot( m2, label='TV',linewidth=2) #,linestyle=':')
    ymin = m2 - e2
    ymax = m2 + e2
    col=plt.gca().lines[-1].get_color()
    ax.fill_between(x2, ymax, ymin,color=col,alpha=alpha_fill)

    if label=='Inter Beat Interval':
        ax.set_ylim(-1.,1.25)
    if label=='Skin Conductance Level':
        ax.set_ylim(-1,3)
    if label=='Pulse Volume Amplitude':
        ax.set_ylim(-1.5,1.75)
    if label=='Continuous Response Measure':
        ax.set_ylim(-2.,4.)
    
    ymin,ymax=plt.ylim()
    #ymin=ymin-(ymax-ymin)/10
    #ax.set_ylim(ymin,ymax)
    yL=[ymin,ymin+(ymax-ymin)/10]


    ymin1,ymax1=plt.ylim()
    
    for xx in mark:
        plt.vlines(xx, ymin=ymin1,ymax=ymax1, colors = 'red',zorder=12) 

    for xx in tL:
        plt.vlines(xx, ymin=yL[0],ymax=yL[1], colors = 'black',zorder=12) 

    for i in [0,4,8,12]:
        t1=mark[i]
        t2=mark[i+1]
        rect=mpatches.Rectangle((t1,ymin1),t2-t1,ymax1-ymin1, 
                        fill=True,alpha=.79,
                        color="lightgrey",
                        zorder=11,
                        linewidth=2)
                       #facecolor="red")
        plt.gca().add_patch(rect)
    #correlations    
    if label=='Continuous Response Measure':
        s1=np.array([])
        s2=np.array([])
        for i in [1,2,3,5,6,7,9,10,11]:
            t1=mark[i]
            t2=mark[i+1]
            arr1 = [m1[t1:t2]]
            arr2 = [m2[t1:t2]]
            
            s1=np.append(arr1,[s1])
            s2=np.append(arr2,[s2])
        
        r, p = stats.pearsonr(s1, s2)

    else:
        r, p = stats.pearsonr(m1, m2)
    
    if p<=.05:
        star='*'
    if p<=.01:
        star='**'
    if p<=.001:
        star='***'
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))    
    ax.set_title(label+' (r='+str(round(r,2))+star+')', pad=35,fontweight='bold')
    
    if label=='Continuous Response Measure':
        clis=[1,2,3,5,6,7,9,10,11]
    else:    
        clis=[0,1,2,3,4,5,6,7,8,9,10,11,12]

    iscList=[]    
    for i in clis:
        
        t1=mark[i]
        t2=mark[i+1]
        s1 = m1[t1:t2]
        s2 = m2[t1:t2]

        r, p = stats.pearsonr(s1, s2)
        r=round(r,2)
        #xp=t1+(t2-t1)/2-co
        sc1=raw1.iloc[t1:t2,:]
        sc2=raw2.iloc[t1:t2,:]
        
        #yp2=ymax-1.5*(ymax/7)
        r1,p1=ISC(sc1)
        r2,p2=ISC(sc2)
        
        iscList.append(['CLip'+str(i+1),fisher_transform(r),fisher_transform(r1),fisher_transform(r2)])

        yp=ymax-((ymax-ymin)/8.5)
        #plt.text(xp, yp, 'r='+str(round(r,2)), fontsize = 14)
        text='r='+str(r)
        x_values = [t1,t2] #[1, 3]
        y_value = yp #5
        midpoint_x = sum(x_values) / 2
        if p<=.05:
            star='*'
        if p<=.01:
            star='**'
        if p<=.001:
            star='***'
        plt.text(midpoint_x, y_value, text, ha='center', va='bottom', color='black', fontsize=14)

            #y_value = yp2 #5
            #plt.text(midpoint_x, y_value, star, ha='center', va='bottom', color='black', fontsize=14)
    ax.set(xlabel='Time (minutes)') #,ylabel=lab)
    #ax.set(title=label)
    ax.set_xticks(np.arange(0,len(m2)+1800, 1800))
    ax.set_xticklabels(np.arange(0,len(m2)/1800+1))
    
    ax.margins(x=0)
    ax.margins(y=0)  
    
    #plt.legend(loc='lower right')
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3)

    
    #ax.set_ylim(-2,2)
    #plt.yticks([-1,1])
    plt.show()
    f.savefig(path+label+'.png', bbox_inches = "tight")
    return iscList

#clip numbers for aggregation            
relax=[0,4,8,12]
scary=[1,2,3]
funny=[5,6,7]
sad=[9,10,11]

pd.set_option('display.float_format', '{:.2f}'.format)
Times=[2,120,282,280,133,160,224,176,155,133,110]
#generate markers for graphs of time series from duration list 
#Markers=[0,125,282,280,133,120,160,224,176,120,155,133,110,120]
Markers=[0,120,282,280,133,120,160,224,176,120,155,133,110,120]


#generate markers for graphs from duration list 
ClipNames=['blank','Relax','Conjuring2','Guest','Ring','ForestDance','Christinth','HotDog','MyGirl','MeMarley','GreenMile']
ClipList= pd.DataFrame([Times],columns=ClipNames)

header1=['ID','condition','clip','bpm','ibi','sdnn','sdsd','rmssd','pnn20','pnn50','hr_mad','sd1','sd2','s',
         'sd1sd2','breathingrate','AMPm','AMPs','IBIm','IBIs','SCLm','SCLs','RTRm','RTRs']

path=os.path.dirname(os.path.abspath(__file__))+'/'

x=path.replace('\\','/')+'results/'
paths=[]
paths.append(x+'VR/')
paths.append(x+'TV/')
dax=[]
datS=[]
zz=0
fl=0

# data frames for time series with all subjects as columns
tR1=pd.DataFrame()
tA1=pd.DataFrame()
tI1=pd.DataFrame()
tM1=pd.DataFrame()
tG1=pd.DataFrame()
tS1=pd.DataFrame()
tR2=pd.DataFrame()
tA2=pd.DataFrame()
tI2=pd.DataFrame()
tM2=pd.DataFrame()
tG2=pd.DataFrame()
tS2=pd.DataFrame()

c2_flag=0
DatRate=0
datN=0
HRV_dictVR={}
HRV_dictTV={}

labs=['Relax','Conjuring2','Guest','Ring','ForestDance','Christinth','HotDog','MyGirl','MeMarley','GreenMile']
HRV_dictVR= {label: np.array((0,2)) for label in labs}  
HRV_dictTV= {label: np.empty((0,2)) for label in labs}  

#with open("test.txt", "w") as myfile:
 #   myfile.write(' ')
tiS=[]
para2=[]
for cond in range(0,2):
    opath = paths[cond]
    onlyfiles = [f for f in listdir(opath) if isfile(join(opath, f)) 
                 and f.endswith('.csv')] # and ( (int(f[3:6])==5) or (int(f[3:6])==68) )] 
    s=0
    #print (onlyfiles)
    for l in onlyfiles:
        IDnum=int(l[3:6])
        #print (IDnum)
        relax_flag=False

        s+=1
        il=l[0:6]
        if cond==0:
            FoID=il.replace('SUB','VR')
        else:
            FoID=il.replace('SUB','TV')
        
        #individually mounted time series except black screens
        R=np.array([])
        A=np.array([])
        I=np.array([])
        M=np.array([])
        G=np.array([])
        S=np.array([])
        tiAll=np.array([])
        tiAll=np.append(tiAll,0.0)
        tip=0.0
        zz+=1
        vp=[]
        dar=[l]
        iz=0
            
        tt = pd.read_csv(opath+l,sep=',',usecols=[0]  ,names=['time'],header=1)
        daGSR= pd.read_csv(opath+l,sep=',',usecols=[1]  ,names=['GSR'],header=1)
        daPPG= pd.read_csv(opath+l,sep=',',usecols=[2]  ,names=['PPG'],header=1)
        daSCR= pd.read_csv(opath+l,sep=',',usecols=[3]  ,names=['SCR'],header=1)
        daSCR.replace(to_replace=0, value=5,inplace=True)

        daSCR['SCR'] = daSCR['SCR'] - 5
        
        t1c=(tt.iloc[0,0])
        t2c=(tt.iloc[-1,0])
        tix=get_mil(t1c)    #hh+mi+se+mil   #time zone
        
        tLis=[]
        
        for t in range(len(tt)):
            
            ti=tt.iloc[t,0]
            tx=get_mil(ti)   #int((hh+mi+se+mil))    #time zone
            tLis.append(tx)

        tir=np.asarray(tLis)
        GSR=daGSR.values.flatten()
        PPG=daPPG.values.flatten()
        SCR=daSCR.values.flatten()
        
        #values for heart PY
        rawx=PPG
        tirx=tir
        
        #read rating file with extra time stamps for the stimuli
        tStamps=l.replace('.csv','_SCR.DAT')
        tSt = pd.read_csv(opath+tStamps,sep='\t')

        start=tSt.iloc[0,1] #start rating File 1. clip ca. 3 secs later because of mp4 laoding
        sti=-1
        ste=0
        za=0
        
        lastSCR=0
        z=0
       
        for i in range(len(tSt)):
           
            za+=1
            fi=0
            ID=tSt.iloc[i,0].replace ('.mp4','')
            
            duration=ClipList.loc[0,ID]

            #get start and end index of time segment frm rating fil
            h1=(tSt.iloc[i,1])
            h2=(tSt.iloc[i,2])
            st=get_mil(dt(h1))
            en=st+duration*1000
            while sti< st:
                z+=1
                sti=tir[z]
            zan=z
            ste=sti
            
            while ste< en:
                z+=1
                try:
                    ste=tir[z]
                except:
                    z=len(tir)-1
                    ste=en
            zen=z    
            tiS.append([ID,zan,zen])    

            if ID !=  'blank':   #add only if not blank 

                if ID=='Relax':
                    iz+=1
                    IDx=ID+str(iz)
                    SCR[zan:zen]=[0]*(zen-zan)  #lastSCR

                else:
                    IDx=ID                            

                #datseg=PPG[stz+30:sez-30]
                datseg=rawx[zan:zen]
                tirseg=tirx[zan:zen]
                tirseg=tirseg-tirseg[0]
                tirseg=np.round(tirseg)
                
                dar=[l[0:6],str(cond+1),IDx]
                sample_rate = hp.get_samplerate_mstimer(tirseg)
                try:
                    wdx, mx = hp.process(datseg, sample_rate)
                    #hp.plotter(wdx, mx)
                except:
                    pass
             
                datseg = hp.enhance_peaks(datseg, iterations=2)
                wd, m = hp.process(datseg, sample_rate)
                DatRate+=sample_rate
                fs=sample_rate
                datN+=1
                
                for measure in m.keys():
                    # print('%s: %f' %(measure, m[measure]))
                    x=m[measure]
                    if measure=='breathingrate':
                        x=x*60
                    if measure=='ibi':
                        x=x
                    x=np.round(x,4)
                    dar.append (x)
                
                #calculate AMP and create IBI AMP GSR SCR time series
                #then get means for AMP GSR SCR for the slot and dd to the parameters
                #first peak not used last peak eventullay canceled
                #peaklist for no. 1 1. relax , e.g 144, RR_list 143, RR_list_cor 142
                #if masklist=0 then use peak always ignore first peak in peaklist
                # or in peak indices start with first value of first touple.
                
                def get_AMP_IBI_fill(w):
                    l=len(w['hr'])
                    amp=np.zeros((l))
                    ibi=np.zeros((l))
                    raw=np.asarray(w['hr'])
                    ind=np.asarray(w['RR_indices'])
                    msk=np.asarray(w['RR_masklist'])
                    dst=np.asarray(w['RR_list'])
                    i=0
                    for x1,x2 in (ind):
                        if msk[i]==0:
                            clip=raw[x1:x2]
                            mi=np.min(clip)
                            ma=raw[x2]
                            amp[x2]=ma-mi
                            ibi[x2]=dst[i]
                            #print (amp,ibi)
                        i+=1                  
                    ibialt=0
                    ampalt=0
                    
                    for i  in range(l):
                        if ibi[i]==0:
                            ibi[i]=ibialt
                            amp[i]=ampalt
                        else:
                            ibialt=ibi[i]
                            ampalt=amp[i]
                 
                    #f,ax = plt.subplots(figsize=(18,3))
                    
                    #plt.plot(ibi)
                    amp=interpol(amp,0)
                    ibi=interpol(ibi,1)
                    #plt.plot(ibi)
                    #plt.show()                    
                    #fill invalid values at beniing and end of data 
                    #with last precedent vaid value
                    for i in range (1000,-1,-1):
                        if math.isnan(amp[i])==True or amp[i]==0:
                            amp[i]=amp[i+1]
                        if math.isnan(ibi[i])==True or ibi[i]==0:
                            ibi[i]=ibi[i+1]
                    for i in range (len(amp)-1000,len(amp)-1):
                        if math.isnan(amp[i])==True or amp[i]==0:
                            amp[i]=amp[i-1]
                        if math.isnan(ibi[i])==True or ibi[i]==0:
                            ibi[i]=ibi[i-1]

                    return amp,ibi
                
                
                #get amplitute and fill IBI and AMP timeseries of segment
                #using hp output as raw data, peaklist and rr_valid, 
                #IBI is redundant with heartPy output, but we need time series
                
                amp,ibi=get_AMP_IBI_fill(wdx)    
                #print (len(ibi))                
                #get also time series of segments for gsr and scr not processed by heartPy
                gsr=GSR[zan:zen]
                scr=SCR[zan:zen]

                #get baseline mean from the first relax sequence
                if ID=='Relax' and relax_flag==False:
                    basemean_amp=np.mean(amp[300:3600])
                    basemean_ibi=np.mean(ibi[300:3600])
                    basemean_gsr=np.mean(gsr[300:3600])
                    relax_flag=True    
            
                #last valid SCR before relaxation remains constant during relax, 
                #mistake in recording, should be reset to zero
                lastSCR=scr[-1]                
                
                #get means of segment data IBI is redundant with hp reults, just as cotroll
                Am=npmean(amp)
                As=npstd(amp)
                Im=npmean(ibi)
                Is=npstd(ibi)
                Gm=np.mean(gsr)
                Gs=np.std(gsr)
                Sm=np.mean(scr)
                Ss=np.std(scr)
                
                #print (Am,Im,Mm,Gm,Sm)
                dar.append(round(Am,4))
                dar.append(round(As,4))
                dar.append(round(Im,4))
                dar.append(round(Is,4))
                dar.append(round(Gm,4))
                dar.append(round(Gs,4))
                dar.append(round(Sm,4))
                dar.append(round(Ss,4))
        
                dax.append(dar)
                
                #now append cut time sries segments to one complete file 

                #raw data not used further on just for control
                ppg=PPG[zan:zen]
                R=np.append(R,ppg)
                
                #append data time series of segment
                A=np.append(A,amp)
                I=np.append(I,ibi)
                G=np.append(G,gsr)
                S=np.append(S,scr)
                tip+=(zen-zan)
                tiAll=np.append(tiAll,tip)
                
        
        #filter normalize baseline correct etc. it here 
        s1=int(tiAll[1])
        of_st=300
        Abase=np.mean(A[of_st:s1])
        Ibase=np.mean(I[of_st:s1])
        Gbase=np.mean(G[of_st:s1])
        A=A-Abase
        I=I-Abase
        G=G-Abase

        Af=filter_it(pd.DataFrame(A),1)
        If=filter_it(pd.DataFrame(I),2)
        Gf=filter_it(pd.DataFrame(G),1)
        Sf=pd.DataFrame(S)
        
        par=[IDnum,cond]
        for zt in range (len(tiAll)-1):
            az=int(tiAll[zt])
            ae=int(tiAll[zt+1])
            avA=np.mean(Af[az:ae])
            avI=np.mean(If[az:ae])
            avG=np.mean(Gf[az:ae])
            avS=np.mean(Sf[az:ae])
            avA=np.round(avA,3)
            avI=np.round(avI,3)
            avG=np.round(avG,3)
            avS=np.round(avS,3)
            
            par+=[avA,avI,avG,avS]
           
        para2.append (par)   
        st=len(dax)-13 #(zz-1*12)-1
        #dax.append(en)
        ndim=24
        
        en=[dar[0],dar[1],'Relax']    
        for j in range(3,ndim):
            su=0
            for cl in relax:
                #print (cl,j,VpDf.iloc[cl,j])
                su=su+dax[cl+st][j]
            su=np.round(su/len(relax),4)
            en.append(su)
        #dax.append(en)
        datS.append(en)

        en=[dar[0],dar[1],'Scary']    
        for j in range(3,ndim):
            su=0
            for cl in scary:
                #print (cl,j,VpDf.iloc[cl,j])
                su=su+dax[cl+st][j]
            su=np.round(su/len(scary),3)
            en.append(su)
        #dax.append(en)
        datS.append(en)
        
        en=[dar[0],dar[1],'Funny']    
        for j in range(3,ndim):
            su=0
            for cl in funny:
                #print (cl,j,VpDf.iloc[cl,j])
                su=su+dax[cl+st][j]
            su=np.round(su/len(funny),3)
            en.append(su)
        #dax.append(en)
        datS.append(en)

        en=[dar[0],dar[1],'Sad']    
        for j in range(3,ndim):
            su=0
            for cl in sad:
                #print (cl,j,VpDf.iloc[cl,j])
                su=su+dax[cl+st][j]
            su=np.round(su/len(sad),3)
            en.append(su)
        datS.append(en)
        
        def cut(column_name,data,  df=None):
            #adjust length of data matrix wih all subjects time series if individual data longer
            #or adjust individual time series if shorter 
            
            # If df is not provided, create an empty DataFrame
            if df is None:
                df = pd.DataFrame()
            # Check the length of the data
            data_length = len(data)
            # Fill the existing DataFrame with zeros if needed
            if len(df) < data_length:
                num_rows_to_add = data_length - len(df)
                zeros_data = np.zeros((num_rows_to_add, 1))
                df_to_add = pd.DataFrame(zeros_data, columns=[column_name])
                df = pd.concat([df, df_to_add], ignore_index=True)
            # If the data length is shorter, extend the data with zeros
            elif len(df) > data_length:
                data = np.pad(data, (0, len(df) - data_length), 'constant')
            df[column_name] = data
        
            return df 

        if cond==0:
            tA1= cut(FoID,A,tA1)
            tI1= cut(FoID,I,tI1)
            tG1= cut(FoID,G,tG1)
            tS1= cut(FoID,S,tS1)
        else:
            tA2= cut(FoID,A,tA2)
            tI2= cut(FoID,I,tI2)
            tG2= cut(FoID,G,tG2)
            tS2= cut(FoID,S,tS2)
            
        if cond==0:
            tA1.loc[:,FoID]=filter_it(tA1.loc[:,FoID],1)
            tI1.loc[:,FoID]=filter_it(tI1.loc[:,FoID],2)
            tG1.loc[:,FoID]=filter_it(tG1.loc[:,FoID],1)
            tS1.loc[:,FoID]=filter_it(tS1.loc[:,FoID],0)
        else:
            tA2.loc[:,FoID]=filter_it(tA2.loc[:,FoID],1)
            tI2.loc[:,FoID]=filter_it(tI2.loc[:,FoID],2)
            tG2.loc[:,FoID]=filter_it(tG2.loc[:,FoID],1)
            tS2.loc[:,FoID]=filter_it(tS2.loc[:,FoID],0)


h2=['ID','Condition']
for i in range (13):
    nam='C'+str(i+1)
    h2+=[nam+'_AMP',nam+'_IBI',nam+'_SCL',nam+'_SCR']
paraNew=pd.DataFrame(para2, columns=h2)
paraNew.to_csv(path+'outFilter.txt', sep='\t',index=False,header=h2)

marks=[]
tt1=0
sr=DatRate/datN
for i in Markers:
    tt1+=round(i*sr)
    #print (tt1)
    marks.append(tt1)


def fill_end(df):
    # Iterate over each column in the DataFrame
    for column in df.columns:
        for j in range(len(df)):
            if pd.isna(df.loc[j,column]):
                df.loc[j,column]= df.loc[j-1,column]
    return df

#fill data ending of all parametr arrays and all colums (subjects) if nan, with last valid value        
fill_end(tA1)
fill_end(tI1)
fill_end(tG1)
fill_end(tS1)
fill_end(tA2)
fill_end(tI2)
fill_end(tG2)
fill_end(tS2)

def get_graphpars(d1,d2):
    
    #get mean time series across all subjects in group for graphs
    m1=d1.mean(axis=1)
    m2=d2.mean(axis=1)
    print (m1)
    s1=d1.std(axis=1)
    s2=d2.std(axis=1)
    
    n1=d1.shape[1]
    n2=d2.shape[1]
 
    e1=s1/np.sqrt(n1)
    e2=s2/np.sqrt(n2)
    

    return m1,m2,e1,e2

def t_test(d1,d2):
    reslist=[]
    for i in range(len(d1)):

        a = d1.iloc[i,:].dropna().values
        b = d2.iloc[i,:].dropna().values
        
        if len(a)!=0 and len(b)!=0:
        #try:
            t, p = ttest_ind(a, b, equal_var=False)
            
            if p <=.05:
                sig=1
                if i>=300:
                    reslist.append(i)
            else:
                sig=0
    Tline=np.asarray(reslist)
    return Tline


namesAv=['IBI','PVA','GSR','RTR']
p=['Inter Beat Interval','Pulse Volume Amplitude','Skin Conductance Level','Continuous Response Measure']
vartup=[[tI1,tI2],[tA1,tA2],[tG1,tG2],[tS1,tS2]]

j=0
def fill_zeros_and_nans(arr):
    filled_arr = arr.copy()
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        mask = (col != 0) & (~np.isnan(col))
        if np.any(mask):
            first_nonzero_idx = np.argmax(mask)
            filled_arr[:first_nonzero_idx, col_idx] = col[first_nonzero_idx]
    return filled_arr

iL=[]
for t1,t2 in vartup:
    l1=t1.shape[0]
    l2=t2.shape[0]
    if l1<l2:
        l=l1
    else:
        l=l2
        
    t1=t1[:l]
    t2=t2[:l]
    ts=t_test(t1,t2)
    m1,m2,e1,e2=get_graphpars(t1,t2)
    
    iLis=show_it_err(m1,m2,e1,e2,t1,t2,ts,marks,p[j])
    j+=1
    iL+=iLis
    
di = pd.DataFrame(iL,columns=['clip','rVR_TV','ISC_VR','ISC_TV'])
di.to_csv(path+'ISC.txt', sep='\t',index=False)

### Data storage  ###
def restructure(d,ni,nh,oh):
#restructure data so that clips are no longer cases but repeated measures within each subject
    
    nHead=[]
    nn=len(nh)
    for ii in ni:
        nHead.append(ii)
   
    for z in range (nn):
        for ii in oh:
    #        print (ii)
            ii=ii.replace('breathingrate','breath')
            nHead.append(nh[z]+'_'+ii) # (da.iloc[i*4+j,:])
    
    N=int(len(d)/nn)
    da=pd.DataFrame(d)
    ll=[]
  
    for i in range(N):
        nLis=[]
        for ii in da.iloc[i*nn,0:2]:
            nLis.append(ii)
        for j in range(nn):
            for ii in da.iloc[i*nn+j,3:]:
                nLis.append(ii) # (da.iloc[i*4+j,:])
        #print (nLis)
        ll.append(nLis)
    
    db=pd.DataFrame(ll,columns=nHead) 
    return db,nHead


#safe the various output DataFrames: structired and unstructured
df = pd.DataFrame(dax,columns=header1)
df.to_csv(path+'outRAW.txt', sep='\t',index=False,header=header1)

ind=header1[0:2]
# store data aggregated across stimulus category, restructured as repeated measures 
newh=['rel','sca','fun','sad']
df,nHead=restructure (datS,ind,newh,header1[3:])
df.to_csv(path+'outAgg.txt', sep='\t',index=False,header=nHead)

#create new header for restructured repeated measurement file
newh=[]
for i in range (1,14):
    newh.append('C'+str(i))
#store repated measuremnt file
df,nHead=restructure (dax,ind,newh,header1[3:])
df.to_csv(path+'outREP.txt', sep='\t',index=False,header=nHead)


