import pandas as pd
import numpy as np
import os
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import seaborn

fileIn1 = 'Dissolved Inorganic Nutrients.csv'
fileIn2 = 'chlorophyllAnnual.csv'
fileIn3 = 'Chlorophyll.csv'
fileIn4 = 'seaIceMonth.txt'
fileIn5 = 'seaIceAnnual.txt'
fileIn6 = 'Primary Production.csv'
fileIn7 = 'Dissolved Oxygen.csv'


colInOrg    = ['studyName','eventNum','cast','bottle','datetime','stationID','line',\
'Station','lat','lon','percentIrradiance','depth','PO4','SiO4','NO2','NO3','NH4','Nitrite',\
'notes']

colChlA = ["studyName","Date","ChlA"]
colPP = ["studyName","eventNum","datetimeGMT","lat","lon","julianDay","stationName","percentIrradiance","depth","primaryProd","primProdSTD","notes"]
colO2 = ["studyName","eventNum","datetimeGMT","gridLine","gridStation","lat","lon","stationName","bottleNumber","depth","O2_ml/L","O2_micromol/L","notes"]
if(os.path.isfile(fileIn1)):
    inOrg = pd.read_csv(fileIn1,sep=',',skiprows=1,names=colInOrg,na_filter=True)
else:
    print fileIn1,"is either unreadable or DNE."
    raise SystemExit(0)

if(os.path.isfile(fileIn2)):
    chlA = pd.read_csv(fileIn2,sep=',',skiprows=1,names=colChlA,parse_dates=True,\
    infer_datetime_format=True,na_filter=True)
else:
    print fileIn2,"is either unreadable or DNE."
    raise SystemExit(0)

PP = pd.read_csv(fileIn6,sep=',',skiprows=1,names=colPP,na_filter=True)
O2 = pd.read_csv(fileIn7,sep=',',skiprows=1,names=colO2,na_filter=True)

skip = -999

inOrg =inOrg[inOrg.datetime>'2005-12-31 00:00:00'] # start date
inOrg.lat=inOrg.lat.map('{:,.2f}'.format).astype(float)
inOrg.lon=inOrg.lon.map('{:,.2f}'.format).astype(float)

# Plotting variables
lblSize = 12
colorSwatch = ['r','orange','g','k']
mark      =   ['o','p','d','v']

#cordLat = inOrg.lat.unique()
#for i in range(len(cordLat)):
#    temp1 = inOrg.loc[inOrg.lat==cordLat[i]]
#    cordLon = temp1.lon.unique() #check if this is just 1...
#    for j in range(len(cordLon)):
#        # intialize plot
#        temp2 = temp1.loc[temp1.lon==cordLon[j]]
#        if len(temp2.index)<3:
#            print("too short")
#            break
#        cordTime = temp2.datetime.unique()
#        fig,ax = plt.subplots(ncols=2,figsize=(8,6)) 
#        for k in range(len(cordTime)):
#            temp3 = temp2.loc[temp2.datetime==cordTime[k]]
#            search = temp3[inOrg.Nitrite!=skip]
#            search = search[search.Nitrite.notnull()]
#            
##            print cordTime[k]
#            
#            x = search.Nitrite.values
#            y = search.depth.values*-1
#            
#            
#            ax[0].scatter(x,y ,marker=mark[i%4],s=12,\
#                               color=colorSwatch[i%4],
#                                alpha=.7,zorder=10)
#            ax[0].plot(x,y,colorSwatch[i%4])
#        for k in range(len(cordTime)):
#            temp3 = temp2.loc[temp2.datetime==cordTime[k]]
#            search2 = temp3[inOrg.PO4!=skip]
#            search2 = search[search.PO4.notnull()]
#            X = search.PO4.values 
#            ax[1].scatter(X,y ,marker=mark[i%4],s=12,\
#                               color=colorSwatch[i%4],
#                                alpha=.7,zorder=10,label=search2.datetime.unique())
#            ax[1].plot(X,y,colorSwatch[i%4]);
#        art = []
#        lgd = ax[1].legend(scatterpoints=1,loc='upper center',markerscale=2,
#                     bbox_to_anchor=(0.5, -0.08),ncol=2,prop={'size':12})
#        art.append(lgd) 
#        
#        maxDep = temp2.depth.max()*-1
#        [a.set_ylim([maxDep-20,0]) for a in ax]
#        ax[0].set_title(r'Nitrite',size=16)
#        ax[1].set_title(r'PO4',size=16)
#        ax[0].set_xlabel(r'Nitrite [$\mu$m/L]',size=lblSize )
#        ax[0].set_ylabel('Water column height [m]',size=lblSize )
#        ax[1].set_xlabel(r'PO4 [$\mu$m/L]',size=lblSize)
#        ax[1].set_ylabel('Water column height [m]',size=lblSize)
#        plt.tight_layout()
#        plt.show()

#cordLat = inOrg.lat.unique()
#cordLatL = len(cordLat)
#for i in range(cordLatL):
#    temp1 = inOrg.loc[inOrg.lat==cordLat[i]]
#    cordLon = temp1.lon.unique() #check if this is just 1...
#    cordLonL = len(cordLon)
#    for j in range(cordLonL):
#        print "Coordinate",cordLat[i],cordLon[j]
#        temp2 = temp1.loc[temp1.lon==cordLon[j]]
#        temp2 = temp2[temp2.Nitrite!=skip]
#        temp2 = temp2[temp2.Nitrite.notnull()]
#        if len(temp2.index)<4:
#            break
#        cordTime = temp2.datetime.unique()
#        cordTimeL = len(cordTime)
#        # intialize plot
#        if(cordTimeL>=4):
#            plt.figure(figsize=(4,8))
#            maxDep = temp2.depth.max()*-1
#            plt.ylim([maxDep-20,0])
#            plt.xlabel(r'Nitrite [$\mu$m/L]',size=lblSize )
#            plt.ylabel('Water column height [m]',size=lblSize )
#            
#            for k in range(cordTimeL):
#                search = temp2.loc[temp2.datetime==cordTime[k]]
#                if len(search.index)<3:
#                    break
#                x = search.Nitrite.values
#                y = search.depth.values*-1
#                
#                plt.scatter(x,y ,marker=mark[k%4],s=12,\
#                                   color=colorSwatch[k%4],
#                                    alpha=.7,zorder=10,label=search.datetime.unique())
#                plt.plot(x,y,colorSwatch[k%4])
#        
#            plt.legend(scatterpoints=1,loc='upper center',markerscale=2,
#                         bbox_to_anchor=(0.5, -0.08),ncol=2,prop={'size':10})
#    
#            plt.title(r'Nitrite',size=16)
#            plt.show()
#            

stationMaster = inOrg.stationID.unique()
spatialSteps = len(stationMaster)
for i in range(spatialSteps):
    spatial = inOrg.loc[inOrg.stationID==stationMaster[i]]
    spatial = spatial[spatial.Nitrite!=skip]
    spatial = spatial[spatial.Nitrite.notnull()]
    temporalSteps = spatial.datetime.unique()
#    if len(temporalSteps)<3:sp
#        break

    plt.figure(figsize=(4,8))
    maxDep = spatial.depth.max()*-1
    plt.ylim([maxDep-20,0])
    plt.xlabel(r'Nitrite [$\mu$m/L]',size=lblSize )
    plt.ylabel('Water column height [m]',size=lblSize )
        
    for j in range(temporalSteps):
        temporal = spatial.loc[spatial.datetime==temporalSteps[j]]
        x = temporal.Nitrite.values
        y = temporal.depth.values*-1
                    
        plt.scatter(x,y ,marker=mark[j%4],s=12,\
                           color=colorSwatch[j%4],
                            alpha=.7,zorder=10,label=search.datetime.unique())
        plt.plot(x,y,colorSwatch[j%4])
    
    plt.legend(scatterpoints=1,loc='upper center',markerscale=2,
                     bbox_to_anchor=(0.5, -0.08),ncol=2,prop={'size':10})

    plt.title(r'Nitrite',size=16)
    plt.show()
        
        
        
#cordLon = inOrg.lon.unique()
#cordLonL = len(cordLon)
#for i in range(cordLonL):
#    temp1 = inOrg.loc[inOrg.lon==cordLon[i]]
#    cordLat = temp1.lat.unique() #check if this is just 1...
#    cordLatL = len(cordLat)
#    for j in range(cordLatL):
#        print "Coordinate",cordLat[j],cordLon[i]
#        temp2 = temp1.loc[temp1.lat==cordLat[j]]
#        temp2 = temp2[temp2.Nitrite!=skip]
#        temp2 = temp2[temp2.Nitrite.notnull()]
#        if len(temp2.index)<4:
#            break
#        cordTime = temp2.datetime.unique()
#        cordTimeL = len(cordTime)
#        # intialize plot
#        plt.figure(figsize=(4,8))
#        maxDep = temp2.depth.max()*-1
#        plt.ylim([maxDep-20,0])
#        plt.xlabel(r'Nitrite [$\mu$m/L]',size=lblSize )
#        plt.ylabel('Water column height [m]',size=lblSize )
#        
#        for k in range(cordTimeL):
#            search = temp2.loc[temp2.datetime==cordTime[k]]
#            if len(search.index)<3:
#                break
#            x = search.Nitrite.values
#            y = search.depth.values*-1
#            
#            plt.scatter(x,y ,marker=mark[k%4],s=12,\
#                               color=colorSwatch[k%4],
#                                alpha=.7,zorder=10,label=search.datetime.unique())
#            plt.plot(x,y,colorSwatch[k%4])
#        
#        plt.legend(scatterpoints=1,loc='upper center',markerscale=2,
#                     bbox_to_anchor=(0.5, -0.08),ncol=2,prop={'size':10})
#
#        plt.title(r'Nitrite',size=16)
#        plt.show()


plt.figure(figsize=(12,12)) 
map = Basemap(width=2400000,height=1400000,
            resolution='l',projection='stere',\
            lat_ts=50,lat_0=-64.90,lon_0=-64.05)
parallels = np.arange(-90.,90.,1.)
map.drawparallels(parallels,labels=[False,False,False,True])
meridians = np.arange(-180.,181.,1.)
map.drawmeridians(meridians,labels=[True,False,False,False])
map.drawcoastlines()
map.fillcontinents()
map.drawmapboundary()

search = inOrg[inOrg.NO2!=skip]
search = search[search.NO2.notnull()]

x1 = search.lon.values.T.tolist()
y1 = search.lat.values.T.tolist() 
z1 = search.NO2.values.T.tolist()
xS, yS = map(x1, y1)
map.scatter(xS, yS, c=z1, marker='o',cmap='jet',s=12,linewidth=.02)
cbar = plt.colorbar(orientation='vertical',fraction=0.026, pad=0.04)
plt.title('Antartic InOrganic NO2',size=16)
plt.clim(0,0.5)
plt.show()



#cords.lat = inOrg.lat.round(decimals=1,out=None).unique()
#for i in range(len(cords)):
    