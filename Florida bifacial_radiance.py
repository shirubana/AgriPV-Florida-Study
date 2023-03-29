#!/usr/bin/env python
# coding: utf-8

# # Florida AgriPV Study with raytrace

# ![image.png](attachment:image.png)

# In[1]:


import os
from pathlib import Path
import bifacial_radiance as br
import numpy as np
import datetime
import pickle
import pandas as pd

# Making folders for saving the simulations
basefolder = os.path.join(os.getcwd(), 'TEMP')


# In[2]:


ft2m = 0.3048  # Conversion factor


# In[3]:


# Setups


# In[5]:


# Now per testbed...
resolutionGround = 0.1  # use 1 for faster test runs
xp = 10
setup = 1

if setup == 1:
    hub_height = 4.6*ft2m # 
    pitch = 18*ft2m
    xgap = 0.01 # m. Default 
    bedsWanted = 3
    moduletype = 'basicModule'
    sazm = 180 # Tracker N-S axis orientation
    fixed_tilt_angle = None
    
if setup == 2:
    hub_height = 4.6*ft2m # 
    pitch = 33*ft2m
    xgap = 0.01 # m. Default 
    bedsWanted = 6
    moduletype = 'basicModule'
    sazm = 180 # Tracker N-S axis orientation
    fixed_tilt_angle = None
    
if setup == 3:
    hub_height = 8*ft2m # 
    pitch = 18*ft2m
    xgap = 0.01 # m. Default 
    bedsWanted = 3
    moduletype = 'basicModule'
    sazm = 180 # Tracker N-S axis orientation
    fixed_tilt_angle = None
    
if setup == 4:
    hub_height = 8*ft2m # 
    pitch = 33*ft2m
    xgap = 0.01 # m. Default 
    bedsWanted = 6
    moduletype = 'basicModule'
    sazm = 180 # Tracker N-S axis orientation
    fixed_tilt_angle = None
    
if setup == 5:
    hub_height = 8*ft2m # 
    pitch = 18*ft2m
    xgap = 1.0 # m
    bedsWanted = 3
    moduletype = 'spacedModule'
    sazm = 180 # Tracker N-S axis orientation
    fixed_tilt_angle = None
    
if setup == 6:
    hub_height = 6.4*ft2m # 
    pitch = 28.3*ft2m
    xgap = 0.01 # m
    bedsWanted = 6
    xp = 2
    moduletype = 'basicModule'
    sazm = 90 # VERTICAL facing East-West
    fixed_tilt_angle = 90


# In[9]:


lat = 30.480671646128137
lon = -83.92997540675283

#tilt = 25
sazm = 90 #

albedo = 0.2 # 'grass'

# Field size. Just going for 'steady state'
nMods = 20
nRows = 7

startdates = [pd.to_datetime('2021-01-01 6:0:0'),
              pd.to_datetime('2021-02-01 6:0:0'),
              pd.to_datetime('2021-03-01 6:0:0'),
              pd.to_datetime('2021-04-01 6:0:0'),
                pd.to_datetime('2021-05-01 6:0:0'), 
                pd.to_datetime('2021-06-01 6:0:0'),
                pd.to_datetime('2021-07-01 6:0:0'),
                pd.to_datetime('2021-08-01 6:0:0'),
                pd.to_datetime('2021-09-01 6:0:0'),
                pd.to_datetime('2021-10-01 6:0:0'),
                pd.to_datetime('2021-11-01 6:0:0'),
                pd.to_datetime('2021-12-01 6:0:0'),
                pd.to_datetime('2021-01-01 6:0:0'),
                pd.to_datetime('2021-05-01 6:0:0')]
enddates = [pd.to_datetime('2021-01-31 20:0:0'),
            pd.to_datetime('2021-02-28 20:0:0'),
            pd.to_datetime('2021-03-31 20:0:0'),
            pd.to_datetime('2021-04-30 20:0:0'),
            pd.to_datetime('2021-05-31 20:0:0'),    # May
            pd.to_datetime('2021-06-30 20:0:0'),   # June
            pd.to_datetime('2021-07-31 20:0:0'),   
            pd.to_datetime('2021-08-31 20:0:0'),
            pd.to_datetime('2021-09-30 20:0:0'), 
            pd.to_datetime('2021-10-31 20:0:0'), 
            pd.to_datetime('2021-11-30 20:0:0'), 
            pd.to_datetime('2021-12-31 20:0:0'), 
            pd.to_datetime('2021-01-31 20:0:0')]
            


# ## Setups 1-5

# In[ ]:


for setup in range(5,7):
    for jj in range(0, 1):
        startdate = startdates[jj]
        enddate = enddates[jj]
        mymonthstart = startdate.month
        mymonthend = enddate.month
        print("STARTING ", setup, " from ", mymonthstart, " to ", mymonthend)

        simpath = f'Setup_{setup}_from_{mymonthstart}TO{mymonthend}'
        testfolder = os.path.join(basefolder, simpath)

        if not os.path.exists(testfolder):
            os.makedirs(testfolder)

        if setup == 1:
            hub_height = 4.6*ft2m # 
            pitch = 18*ft2m
            xgap = 0.01 # m. Default 
            bedsWanted = 3
            moduletype = 'basicModule'
            sazm = 180 # Tracker N-S axis orientation
            fixed_tilt_angle = None

        if setup == 2:
            hub_height = 4.6*ft2m # 
            pitch = 33*ft2m
            xgap = 0.01 # m. Default 
            bedsWanted = 6
            moduletype = 'basicModule'
            sazm = 180 # Tracker N-S axis orientation
            fixed_tilt_angle = None

        if setup == 3:
            hub_height = 8*ft2m # 
            pitch = 18*ft2m
            xgap = 0.01 # m. Default 
            bedsWanted = 3
            moduletype = 'basicModule'
            sazm = 180 # Tracker N-S axis orientation
            fixed_tilt_angle = None

        if setup == 4:
            hub_height = 8*ft2m # 
            pitch = 33*ft2m
            xgap = 0.01 # m. Default 
            bedsWanted = 6
            moduletype = 'basicModule'
            sazm = 180 # Tracker N-S axis orientation
            fixed_tilt_angle = None

        if setup == 5:
            hub_height = 8*ft2m # 
            pitch = 18*ft2m
            xgap = 1.0 # m
            bedsWanted = 3
            moduletype = 'spacedModule'
            sazm = 180 # Tracker N-S axis orientation
            fixed_tilt_angle = None

        if setup == 6:
            hub_height = 6.4*ft2m # 
            pitch = 28.3*ft2m
            xgap = 0.01 # m
            bedsWanted = 6
            xp = 2
            moduletype = 'basicModule'
            sazm = 90 # VERTICAL facing East-West
            fixed_tilt_angle = 90


        radObj = br.RadianceObj('Setup',testfolder)

        radObj.setGround(albedo) 

        epwfile = radObj.getEPW(lat, lon) 
        metData = radObj.readWeatherFile(epwfile) 

        # -- establish tracking angles
        trackerParams = {'limit_angle':50,
                         'angledelta':5,
                         'backtrack':True,
                         'gcr':2/pitch,
                         'cumulativesky':True,
                         'azimuth': sazm,
                         'fixed_tilt_angle': fixed_tilt_angle,
                         }

        trackerdict = radObj.set1axis(**trackerParams)

        # -- generate sky   
        trackerdict = radObj.genCumSky1axis()

        sceneDict = {'pitch':pitch, 
                     'hub_height': hub_height,
                     'nMods': 19,
                     'nRows': 7,
                     'tilt': fixed_tilt_angle,  # CHECK IF THIS WORKS! 
                     'sazm': sazm}

        modWanted = 10
        rowWanted = 4

        basicModule = radObj.makeModule(name='basicModule', x=1, y=2)
        spacedModule = radObj.makeModule(name='spacedModule', x=1, y=2, xgap = 1)
        trackerdict = radObj.makeScene1axis(module=moduletype,sceneDict=sceneDict) 

        trackerdict = radObj.makeOct1axis()

        # -- run analysis
        # Analysis for Module
        trackerdict = radObj.analysis1axis(trackerdict, customname = 'Module',
                                           sensorsy=9, modWanted=modWanted,
                                           rowWanted=rowWanted)
        trackerdict = radObj.calculateResults(agriPV=False)
        ResultPVWm2Back = radObj.CompiledResults.iloc[0]['Wm2Back']
        ResultPVWm2Front = radObj.CompiledResults.iloc[0]['Gfront_mean']

        # Modify modscanfront for Ground
        numsensors = int((pitch/resolutionGround)+1)
        modscanback = {'xstart': 0, 
                        'zstart': 0.05,
                        'xinc': resolutionGround,
                        'zinc': 0,
                        'Ny':numsensors,
                        'orient':'0 0 -1'}

        # Analysis for GROUND
        trackerdict = radObj.analysis1axis(trackerdict, customname = 'Ground',
                                           modWanted=modWanted, rowWanted=rowWanted,
                                            modscanback=modscanback, sensorsy=1)


        filesall = os.listdir('results')
        filestoclean = [e for e in filesall if e.endswith('_Front.csv')]
        for cc in range(0, len(filestoclean)):
            filetoclean = filestoclean[cc]
            os.remove(os.path.join('results', filetoclean))
        trackerdict = radObj.calculateResults(agriPV=True)
        ResultPVGround = radObj.CompiledResults.iloc[0]['Wm2Back']
        ghi_sum = metData.ghi.sum()
        ghi_sum

        # GROUND TESTBEDS COMPILATION
        # Tracker Projection of half the module into the ground, 
        # for 1-up module in portrait orientation


        df_temp = ResultPVGround
        # Under panel irradiance calculation
        edgemean = np.mean(df_temp[:xp] + df_temp[-xp:])
        edge_normGHI = edgemean / ghi_sum

        # All testbeds irradiance average
        insidemean = np.mean(df_temp[xp:-xp])
        inside_normGHI = insidemean / ghi_sum



        # Length of each testbed between rows
        dist1 = int(np.floor(len(df_temp[xp:-xp])/bedsWanted))

        Astart = xp + dist1*0
        Bstart = xp + dist1*1
        Cstart = xp + dist1*2

        if bedsWanted == 3:
            Dstart = -xp # in this case it is Cend
        if bedsWanted > 3:
            Dstart = xp + dist1*3
            Estart = xp + dist1*4
            Fstart = xp + dist1*5
            Gstart = -xp  # in this case it is Fend
        if bedsWanted > 6:
            Gstart = xp + dist1*6
            Hstart = xp + dist1*7
            Istart = xp + dist1*8
            Iend = -xp # this is I end

        testbedA = df_temp[Astart:Bstart]
        testbedAmean = np.mean(testbedA)
        testbedA_normGHI = testbedAmean / ghi_sum

        testbedB = df_temp[Bstart:Cstart]
        testbedBmean = np.mean(testbedB)
        testbedB_normGHI = testbedBmean / ghi_sum

        testbedC = df_temp[Cstart:Dstart]
        testbedCmean = np.mean(testbedC)
        testbedC_normGHI = testbedCmean / ghi_sum

        # Will run for bedswanted 6 and 9
        if bedsWanted > 3:
            testbedD = df_temp[Dstart:Estart]
            testbedDmean = np.mean(testbedD)
            testbedD_normGHI = testbedDmean / ghi_sum

            testbedE = df_temp[Estart:Fstart]
            testbedEmean = np.mean(testbedE)
            testbedE_normGHI = testbedEmean / ghi_sum

            testbedF = df_temp[Fstart:Gstart]
            testbedFmean = np.mean(testbedF)
            testbedF_normGHI = testbedFmean / ghi_sum

        # Compiling for return
        if bedsWanted == 6:
            results = [setup, metData.latitude, metData.longitude, 
            mymonthstart, mymonthend,
            ghi_sum,
            ResultPVWm2Front, ResultPVWm2Back, ResultPVGround,
            edgemean, insidemean,
            testbedAmean, testbedBmean, testbedCmean,
            testbedDmean, testbedEmean, testbedFmean,
            edge_normGHI, inside_normGHI,
            testbedA_normGHI, testbedB_normGHI, testbedC_normGHI,
            testbedD_normGHI, testbedE_normGHI, testbedF_normGHI
            ]

        if bedsWanted == 3:
            results = [setup, metData.latitude, metData.longitude, 
                    mymonthstart, mymonthend,
                    ghi_sum,
                    ResultPVWm2Front, ResultPVWm2Back, ResultPVGround,
                    edgemean, insidemean,
                    testbedAmean, testbedBmean, testbedCmean,
                    edge_normGHI, inside_normGHI,
                    testbedA_normGHI, testbedB_normGHI, testbedC_normGHI
                    ]

        # save to folder

        with open('results.pkl', "wb") as fp:   #Pickling
            pickle.dump(results, fp)


# In[13]:


trackerdict.keys()


# In[ ]:


GHIs = []
for mmonth in range(0, len(startdts)):
    startdt = startdts[mmonth]
    enddt = enddts[mmonth]
    metdata = radObj.readWeatherFile(epwfile, starttime=startdt, endtime=enddt, coerce_year=2021)
    GHIs.append(metdata.dni.sum())
GHIs


# In[ ]:


# SETUP 6 - FIXED TILT


# In[ ]:


for mmonth in range(0, len(startdts)):
    startdt = startdts[mmonth]
    enddt = enddts[mmonth]
    metdata = radObj.readWeatherFile(epwfile, starttime=startdt, endtime=enddt, coerce_year=2021) # read in the EPW weather data from above
    radObj.genCumSky(savefile=str(mmonth))
    #radObj.gendaylit(4020)  # Use this to simulate only one hour at a time. 

    sceneDict = {'tilt':tilt, 'pitch':pitch, 'hub_height':hub_height, 
                 'azimuth':sazm, 'nMods':nMods, 'nRows':nRows}  
    scene = radObj.makeScene(module=spacedmodule, sceneDict=sceneDict) 
    octfile = radObj.makeOct(radObj.getfilelist())  

    analysis = br.AnalysisObj(octfile, radObj.name)
    spacingbetweensamples = 0.05 # m
    sensorsy = int(np.floor(pitch/spacingbetweensamples)+1)
    sensorsx = 1

    # Module first
    frontscan, backscan = analysis.moduleAnalysis(scene, sensorsx = 1, sensorsy=10)
    analysis.analysis(octfile, 'MODULE_Month_'+str(mmonth+4)+'_setup_'+(str(setup)), frontscan, backscan)  # compare the back vs front irradiance  

    groundscan, backscan = analysis.moduleAnalysis(scene, sensorsx = 1, sensorsy=[sensorsy, 1])
    groundscan['zstart'] = 0.05  # setting it 5 cm from the ground.
    groundscan['zinc'] = 0   # no tilt necessary. 
    groundscan['yinc'] = spacingbetweensamples
    groundscan['ystart'] = 0
    groundscan['xinc'] = 0
    groundscan['xstart'] = 0

    analysis.analysis(octfile, 'GROUND_Month_'+str(mmonth+4)+'_setup_'+(str(setup)), groundscan, backscan)  # compare the back vs front irradiance  

filesall = os.listdir('results')

# Cleanup of Ground 'back' files
filestoclean = [e for e in filesall if e.endswith('_Back.csv')]
for cc in range(0, len(filestoclean)):
    filetoclean = filestoclean[cc]
    os.remove(os.path.join('results', filetoclean))


# ## 2. Plot Bifacial Gain Results

# In[ ]:


import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


# In[ ]:


font = {#'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)

#sns.set(rc={'figure.figsize':(5.7,4.27)})


# In[ ]:


testfolder


# In[ ]:


import calendar


# In[ ]:


hub_heights = [4.3, 3.5, 2.5, 1.5]
results_BGG=[]
results_GFront=[]
results_GRear=[]
results_GGround=[]
results_coordY=[]
setups = []
months = []
results_GHI = []
for ii in range(0, len(clearance_heights)):
    for jj in range(0, len(startdts)):
        
        if jj == (len(startdts)-1):
            months.append('Season')
        else:
            months.append(calendar.month_abbr[jj+4])
        setups.append(ii+1)
        # irr_GROUND_Month_6_setup_1_Row4_Module10_Back.csv
        fileground= os.path.join('results', f'irr_GROUND_Month_'+str(jj+4)+'_setup_'+str(ii+1)+'_Row4_Module10_Front.csv')
        filepv= os.path.join('results', f'irr_MODULE_Month_'+str(jj+4)+'_setup_'+str(ii+1)+'_Row4_Module10.csv')
        resultsGround = load.read1Result(fileground)
        resultsPV = load.read1Result(filepv)
        #  resultsDF = load.cleanResult(resultsDF).dropna() # I checked them they are good because even number of sensors
        results_GGround.append(list(resultsGround['Wm2Front']))
        results_coordY.append(list(resultsGround['y']))
        results_GFront.append(list(resultsPV['Wm2Front']))
        results_GRear.append(list(resultsPV['Wm2Back']))
        results_BGG.append(resultsPV['Wm2Back'].sum()*100/resultsPV['Wm2Front'].sum())
        results_GHI.append(GHIs[jj])


# In[ ]:


df = pd.DataFrame(list(zip(setups, months, results_GHI, results_coordY, results_GGround,
                          results_GFront, results_GRear, results_BGG)),
               columns =['Setup', 'Month', 'GHI', 'GroundCoordY', 'Gground', 'Gfront', 'Grear', 'BGG'])


# In[ ]:


# Example of selectiong one setup one month
df[(df['Setup']==1) & (df['Month']=='Apr')]


# In[ ]:


foo = df[(df['Setup']==1) & (df['Month']=='Apr')]
foo1 = df[(df['Setup']==1) & (df['Month']=='May')]
foo2 = df[(df['Setup']==1) & (df['Month']=='Jun')]

plt.plot(foo['GroundCoordY'][0], foo['Gground'].iloc[0], label='Apr')
plt.plot(foo['GroundCoordY'][0], foo1['Gground'].iloc[0], label='May')
plt.plot(foo['GroundCoordY'][0], foo2['Gground'].iloc[0], label='June')
plt.title('Setup 1')
plt.xlabel('Row to Row distance')
plt.ylabel('Cumulative Irradiance [Wh/m$_2$]')
plt.legend()


# In[ ]:


foo = df[(df['Setup']==1) & (df['Month']=='Apr')]
foo1 = df[(df['Setup']==1) & (df['Month']=='May')]
foo2 = df[(df['Setup']==1) & (df['Month']=='Jun')]

plt.plot(foo['GroundCoordY'][0], foo['Gground'].iloc[0]/foo['GHI'].iloc[0], label='Apr')
plt.plot(foo['GroundCoordY'][0], foo1['Gground'].iloc[0]/foo1['GHI'].iloc[0], label='May')
plt.plot(foo['GroundCoordY'][0], foo2['Gground'].iloc[0]/foo2['GHI'].iloc[0], label='June')
plt.title('Setup 1')
plt.xlabel('Row to Row distance')
plt.ylabel('Irradiance Factor')
plt.legend();


# In[ ]:


foo = df[(df['Setup']==1) & (df['Month']=='May')]
foo1 = df[(df['Setup']==2) & (df['Month']=='May')]
foo2 = df[(df['Setup']==3) & (df['Month']=='May')]

plt.plot(foo['GroundCoordY'].iloc[0], foo['Gground'].iloc[0], label='Setup 1')
plt.plot(foo1['GroundCoordY'].iloc[0], foo1['Gground'].iloc[0], label='Setup 2')
plt.plot(foo2['GroundCoordY'].iloc[0], foo2['Gground'].iloc[0], label='Setup 3')
plt.title('May')
plt.xlabel('Row to Row distance')
plt.ylabel('Cumulative Irradiance [Wh/m$_2$]')
plt.legend();


# In[ ]:


foo = df[(df['Setup']==1) & (df['Month']=='Jun')]
foo1 = df[(df['Setup']==2) & (df['Month']=='Jun')]
foo2 = df[(df['Setup']==3) & (df['Month']=='Jun')]

plt.plot(foo['GroundCoordY'].iloc[0], foo['Gground'].iloc[0]/foo['GHI'].iloc[0], label='Setup 1')
plt.plot(foo1['GroundCoordY'].iloc[0], foo1['Gground'].iloc[0]/foo1['GHI'].iloc[0], label='Setup 2')
plt.plot(foo2['GroundCoordY'].iloc[0], foo2['Gground'].iloc[0]/foo2['GHI'].iloc[0], label='Setup 3')
plt.title('June')
plt.xlabel('Row to Row distance')
plt.ylabel('Irradiance Factor')
plt.legend();


# ## 3. Testbed Calculations

# In[ ]:


xps = []
for cw in cws:
    xps.append(np.round(cw*np.cos(np.radians(tilt))/2,2))
xps


# In[ ]:


# SETUP Table
# Setups, by month, all irradiance on ground between rows

foo0 = df[(df['Setup']==1) & (df['Month']=='Apr')]
foo1 = df[(df['Setup']==1) & (df['Month']=='May')]
foo2 = df[(df['Setup']==1) & (df['Month']=='Jun')]
foo3 = df[(df['Setup']==1) & (df['Month']=='Jul')]
foo4 = df[(df['Setup']==1) & (df['Month']=='Aug')]
foo5 = df[(df['Setup']==1) & (df['Month']=='Sep')]
foo6 = df[(df['Setup']==1) & (df['Month']=='Season')]


setup1 = pd.DataFrame(list(zip(foo0['Gground'].iloc[0],
                      foo1['Gground'].iloc[0],
                      foo2['Gground'].iloc[0],
                      foo3['Gground'].iloc[0],
                      foo4['Gground'].iloc[0],
                      foo5['Gground'].iloc[0],
                      foo6['Gground'].iloc[0],
                     )), columns=['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Season'],
                     index=foo0['GroundCoordY'].iloc[0])

foo0 = df[(df['Setup']==2) & (df['Month']=='Apr')]
foo1 = df[(df['Setup']==2) & (df['Month']=='May')]
foo2 = df[(df['Setup']==2) & (df['Month']=='Jun')]
foo3 = df[(df['Setup']==2) & (df['Month']=='Jul')]
foo4 = df[(df['Setup']==2) & (df['Month']=='Aug')]
foo5 = df[(df['Setup']==2) & (df['Month']=='Sep')]
foo6 = df[(df['Setup']==2) & (df['Month']=='Season')]


setup2 = pd.DataFrame(list(zip(foo0['Gground'].iloc[0],
                      foo1['Gground'].iloc[0],
                      foo2['Gground'].iloc[0],
                      foo3['Gground'].iloc[0],
                      foo4['Gground'].iloc[0],
                      foo5['Gground'].iloc[0],
                      foo6['Gground'].iloc[0],
                     )), columns=['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Season'],
                     index=foo0['GroundCoordY'].iloc[0])

foo0 = df[(df['Setup']==3) & (df['Month']=='Apr')]
foo1 = df[(df['Setup']==3) & (df['Month']=='May')]
foo2 = df[(df['Setup']==3) & (df['Month']=='Jun')]
foo3 = df[(df['Setup']==3) & (df['Month']=='Jul')]
foo4 = df[(df['Setup']==3) & (df['Month']=='Aug')]
foo5 = df[(df['Setup']==3) & (df['Month']=='Sep')]
foo6 = df[(df['Setup']==3) & (df['Month']=='Season')]


setup3 = pd.DataFrame(list(zip(foo0['Gground'].iloc[0],
                      foo1['Gground'].iloc[0],
                      foo2['Gground'].iloc[0],
                      foo3['Gground'].iloc[0],
                      foo4['Gground'].iloc[0],
                      foo5['Gground'].iloc[0],
                      foo6['Gground'].iloc[0],
                     )), columns=['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Season'],
                     index=foo0['GroundCoordY'].iloc[0])


# In[ ]:


# Setup 1, by month, all irradiance on ground between rows
setup1


# In[ ]:


# edges is the irradiance undreneath the panels of the encompasing rows
# centers is where there is no panel
center1 = setup1[(setup1.index > xps[0]) & (setup1.index <= (setup1.index[-1] - xps[0]))]
edges1 = setup1[~(setup1.index > xps[0]) & (setup1.index <= (setup1.index[-1] - xps[0]))]
center2 = setup2[(setup2.index > xps[1]) & (setup2.index <= (setup2.index[-1] - xps[1]))]
edges2 = setup2[~(setup2.index > xps[1]) & (setup2.index <= (setup2.index[-1] - xps[1]))]
center3 = setup3[(setup3.index > xps[2]) & (setup3.index <= (setup3.index[-1] - xps[2]))]
edges3 = setup3[~(setup3.index > xps[2]) & (setup3.index <= (setup3.index[-1] - xps[2]))]

edges1


# In[ ]:


# Calculating the 3 testbeds. equidistant
dist1 = (center1.index[-1]-center1.index[0])/3
dist2 = (center2.index[-1]-center2.index[0])/3
dist3 = (center3.index[-1]-center3.index[0])/3

setup1testbedA = center1[center1.index <= (center1.index[0] + dist1)]
setup1testbedB = center1[(center1.index > (center1.index[0] + dist1)) & (center1.index <= (center1.index[0] + dist1*2))]
setup1testbedC = center1[center1.index > (center1.index[0] + dist1*2)]

setup2testbedA = center2[center2.index <= (center2.index[0] + dist2)]
setup2testbedB = center2[(center2.index > (center2.index[0] + dist2)) & (center2.index <= (center2.index[0] + dist2*2))]
setup2testbedC = center2[center2.index > (center2.index[0] + dist2*2)]

setup3testbedA = center3[center3.index <= (center3.index[0] + dist3)]
setup3testbedB = center3[(center3.index > (center3.index[0] + dist3)) & (center3.index <= (center3.index[0] + dist3*2))]
setup3testbedC = center3[center3.index > (center3.index[0] + dist3*2)]


# In[ ]:


IFtable = [np.round(list(setup1testbedA.mean()/GHIs),2),
np.round(list(setup1testbedB.mean()/GHIs),2),
np.round(list(setup1testbedC.mean()/GHIs),2),
np.round(list(setup2testbedA.mean()/GHIs),2),
np.round(list(setup2testbedB.mean()/GHIs),2),
np.round(list(setup2testbedC.mean()/GHIs),2),
np.round(list(setup3testbedA.mean()/GHIs),2),
np.round(list(setup3testbedB.mean()/GHIs),2),
np.round(list(setup3testbedC.mean()/GHIs),2),
np.round(list(edges1.mean()/GHIs),2),
np.round(list(edges2.mean()/GHIs),2),
np.round(list(edges3.mean()/GHIs),2),
GHIs]

IFresults = pd.DataFrame(IFtable, index = ['Setup 1 TB A', 'Setup 1 TB B', 'Setup 1 TB C',
                                             'Setup 2 TB A', 'Setup 2 TB B', 'Setup 2 TB C',
                                             'Setup 3 TB A', 'Setup 3 TB B', 'Setup 3 TB C',
                                          'Setup 1 Under Panel', 'Setup 2 Under Panel', 'Setup 3 Under Panel', 'GHIs [Wh/m2]'],
                        columns = setup1testbedB.columns)
IFresults


# In[ ]:


Meanstable = [np.round(list(setup1testbedA.mean()),2),
np.round(list(setup1testbedB.mean()),2),
np.round(list(setup1testbedC.mean()),2),
np.round(list(setup2testbedA.mean()),2),
np.round(list(setup2testbedB.mean()),2),
np.round(list(setup2testbedC.mean()),2),
np.round(list(setup3testbedA.mean()),2),
np.round(list(setup3testbedB.mean()),2),
np.round(list(setup3testbedC.mean()),2),
np.round(list(edges1.mean()),2),
np.round(list(edges2.mean()),2),
np.round(list(edges3.mean()),2),
GHIs]

Meansresults = pd.DataFrame(Meanstable, index = ['Setup 1 TB A [Wh/m2]', 'Setup 1 TB B [Wh/m2]', 'Setup 1 TB C [Wh/m2]',
                                             'Setup 2 TB A [Wh/m2]', 'Setup 2 TB B [Wh/m2]', 'Setup 2 TB C [Wh/m2]',
                                             'Setup 3 TB A [Wh/m2]', 'Setup 3 TB B [Wh/m2]', 'Setup 3 TB C [Wh/m2]',
                                          'Setup 1 Under Panel [Wh/m2]', 'Setup 2 Under Panel [Wh/m2]', 'Setup 3 Under Panel [Wh/m2]', 'GHIs [Wh/m2]'],
                        columns = setup1testbedB.columns)
Meansresults


# In[ ]:


# Describe dataframes for each testbed/setup
A1 = np.round(setup1testbedA.describe(),2)
B1 = np.round(setup1testbedB.describe(),2)
C1 = np.round(setup1testbedC.describe(),2)
A2 = np.round(setup2testbedA.describe(),2)
B2 = np.round(setup2testbedB.describe(),2)
C2 = np.round(setup2testbedC.describe(),2)
A3 = np.round(setup3testbedA.describe(),2)
B3 = np.round(setup3testbedB.describe(),2)
C3 = np.round(setup3testbedC.describe(),2)
E1 = np.round(edges1.describe())
E2 = np.round(edges2.describe())
E3 = np.round(edges3.describe())


# In[ ]:


with pd.ExcelWriter("Results_13Feb23.xlsx") as writer:
    # use to_excel function and specify the sheet_name and index
    # to store the dataframe in specified sheet
    IFresults.to_excel(writer, sheet_name="Irradiance Factors", index=True)
    Meansresults.to_excel(writer, sheet_name="Mean Irradiances", index=True)
    A1.to_excel(writer, sheet_name="Setup 1 TB C", index=True)
    B1.to_excel(writer, sheet_name="Setup 1 TB B", index=True)
    C1.to_excel(writer, sheet_name="Setup 1 TB C", index=True)    
    E1.to_excel(writer, sheet_name="Setup 1 UnderPanel", index=True)
    A2.to_excel(writer, sheet_name="Setup 2 TB C", index=True)
    B2.to_excel(writer, sheet_name="Setup 2 TB C", index=True)
    C2.to_excel(writer, sheet_name="Setup 2 TB C", index=True)
    E2.to_excel(writer, sheet_name="Setup 2 UnderPanel", index=True)
    A3.to_excel(writer, sheet_name="Setup 3 TB C", index=True)
    B3.to_excel(writer, sheet_name="Setup 3 TB C", index=True)
    C3.to_excel(writer, sheet_name="Setup 3 TB C", index=True)
    E3.to_excel(writer, sheet_name="Setup 1 UnderPanel", index=True)


# In[ ]:


# EXPANDED Describe dataframes for each testbed/setup


# In[ ]:


np.round(setup1testbedA.describe(),2)


# In[ ]:


np.round(setup1testbedB.describe())


# In[ ]:


np.round(setup1testbedC.describe())


# In[ ]:


np.round(setup2testbedA.describe())


# In[ ]:


np.round(setup2testbedB.describe())


# In[ ]:


np.round(setup2testbedC.describe())


# In[ ]:


np.round(setup3testbedA.describe())


# In[ ]:


np.round(setup3testbedB.describe())


# In[ ]:


np.round(setup3testbedC.describe())


# In[ ]:


np.round(edges1.describe())


# In[ ]:


np.round(edges2.describe())


# In[ ]:


np.round(edges3.describe())

