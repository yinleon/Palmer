import pandas as pd
import numpy as np
import os
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn
import colorsys

# Import files
fileIn1 = 'inOrganicNutrientsClean.csv'
fileIn4 = 'seaIceMonth.txt'
fileIn5 = 'seaIceAnnual.txt'
fileIn6 = 'Primary Production.csv'
fileIn7 = 'Dissolved Oxygen.csv'
fileIn8 = 'Dissolved Inorganic Carbon.csv'
fileIn9 = 'Basic Western Antarctic Peninsula Survey Grid.csv'


colInOrg    = ['studyName','eventNum','cast','bottle','datetime','stationID','gridLine',\
'gridStation','lat','lon','percentIrradiance','depth','PO4','SiO4','NO2','NO3','NH4','Nitrite',\
'notes']
colPP = ["studyName","eventNum","datetime","lat","lon","julianDay","stationID","percentIrradiance","depth","primaryProd","primProdSTD","notes"]
colO2 = ["studyName","eventNum","datetime","gridLine","gridStation","lat","lon","stationID","bottleNumber","depth","O2_ml/L","O2_micromol/L","notes"]
colCO2 = ["studyName","dateTime","eventNum","bottle","stationID","depth","dic1","dic2","alk1","alk2","notes"]
colGrid = ["studyName","gridLine","gridStation","name","lat","lon","gridCode"]

if(os.path.isfile(fileIn1)):
    inOrg = pd.read_csv(fileIn1,sep=',',skiprows=1,names=colInOrg,na_filter=True,parse_dates=True,infer_datetime_format=True)
    inOrg = inOrg[inOrg.lon.notnull()]
    inOrg = inOrg[inOrg.lat.notnull()]
    inOrg = inOrg[inOrg.datetime.notnull()]    
    inOrg =inOrg[inOrg.datetime>'1999-12-31 00:00:00'] # start date
    inOrg.datetime=pd.to_datetime(inOrg.datetime,format='%Y-%m-%d %H:%M:%S',exact=True)   
    inOrg['year']=pd.DatetimeIndex(inOrg['datetime']).year  
    inOrg['month']=pd.DatetimeIndex(inOrg['datetime']).month
#    inOrg['second']=pd.DatetimeIndex(inOrg['datetime']).second
    inOrg.lat=abs(inOrg.lat.map('{:,.5f}'.format).astype(float))*-1 # convert all to south
    inOrg.lon=abs(inOrg.lon.map('{:,.5f}'.format).astype(float))*-1 # convert all to west
    inOrg = inOrg.reset_index()
else:
    print (fileIn1,"is either unreadable or DNE.")
    raise SystemExit(0)
				
if(os.path.isfile(fileIn8)):
    co2 = pd.read_csv(fileIn8,sep=',',skiprows=1,names=colCO2,na_filter=True)
else:
    print (fileIn8,"is either unreadable or DNE.")
    raise SystemExit(0)
				
if(os.path.isfile(fileIn9)):
	grid = pd.read_csv(fileIn9,sep=',',skiprows=1,names=colGrid,na_filter=True)
else:
    print (fileIn9,"is either unreadable or DNE.")
    raise SystemExit(0)

#pP = pd.read_csv(fileIn6,sep=',',skiprows=1,names=colPP,na_filter=True)
#o2 = pd.read_csv(fileIn7,sep=',',skiprows=1,names=colO2,na_filter=True)


"""****************************************************************************
Before we can do any anaylsis, we need to wrangle the data into a usable format.
This function takes in the database in the form of a pandas dataframe (DF) 
along with the latitude and longitue arrays (df,df.lat,df.lon) and returns 
standardized coordaintes and stationID for interannual comparison.
****************************************************************************"""
def cord2stationID(df,latIn,lonIn):
	# first read in the grid standard
	fileIn = 'Basic Western Antarctic Peninsula Survey Grid.csv'
	colGrid = ["studyName","gridLine","gridStation","name","lat","lon","gridCode"]

	if(os.path.isfile(fileIn)):
		grid = pd.read_csv(fileIn,sep=',',skiprows=1,names=colGrid,na_filter=True)
	else:
	    print (fileIn,"is either unreadable or DNE.")
	    raise SystemExit(0)
	# Lets standardize the coordinates for SW
	latIn = -1*abs(latIn)
	lonIn = -1*abs(lonIn)
	
	# how many points are there?
	x=len(latIn)
	
	# slice the dataframe to the same dimensions as the lat and lon lists.
	df = df[:x]
	
	# Initialize some counters.
	latOut = []
	lonOut = []
	stationOut = []
	lost = 0
	found = 0
	outOfB = 0
	sumD = 0
	
	# Iterate through each lat lon pair in the list, and systematically populate the blank lists in parallel to the input lat/lon
	for i in range(x):
		# check if the values are withiin the grid's bounds.
		if (latIn[i]<min(grid.lat) or latIn[i]>max(grid.lat) or lonIn[i]<min(grid.lon) or lonIn[i]>max(grid.lon)):
			outOfB +=1
			latOut.append(latIn[i])
			lonOut.append(lonIn[i])
			stationOut.append("outGrid")
		# Find standard latitude that is closest to observed.		
		else:
			querry =     grid[grid.lat<=latIn[i]+.0645] #???
			querry = querry[querry.lat>=latIn[i]-.0645]
			# If DNE...
			if querry.empty:
				latOut.append(latIn[i])
				lonOut.append(lonIn[i])
				stationOut.append("notFound")
				lost += 1
                 # IF DE, let's look for standard longitude closest to observed.			
			else:
				querry = querry[querry.lon<=lonIn[i]+0.08]
				querry = querry[querry.lon>=lonIn[i]-0.08]
				qLen = len(querry.index)
				# If DNE...
				if querry.empty:
					latOut.append(latIn[i])
					lonOut.append(lonIn[i])
					stationOut.append("notFound")
					lost += 1
				# If there is perfect a match
				elif (qLen==1): 
					latOut.append(np.asarray(querry.lat, dtype=np.float)[0])
					lonOut.append(np.asarray(querry.lon, dtype=np.float)[0])
					stationOut.append(np.asarray(querry.name, dtype=object)[0])
					found += 1
				else: # the list has multiple values
					qLon = querry.lon.values
					qLat = querry.lat.values
					qStation = querry.name.values
					sumD += qLen
					minDist = 1e6
					# calculate which lat lon combo is closest to station values
					for j in range(qLen):
						lonDiff = abs(lonIn[i]-qLon[j])
						latDiff = abs(latIn[i]-qLat[j])	
						sumDist = lonDiff+latDiff
						if(sumDist<minDist):
							mindex = j
					latOut.append(qLat[mindex])
					lonOut.append(qLon[mindex])
					stationOut.append(qStation[mindex])
					found += 1
	print("found:",found,"\nlost:",lost,"\nsum:",sumD,'\nout',outOfB)
	
	df.lat = latOut[:x]
	df.lon = lonOut[:x]
	df.stationID = stationOut[:x]
	return df
"""****************************************************************************
Use this function to convert DF to to CVS, and write it out as a csv.
****************************************************************************"""
def convert2CSV(df,csvout):
    toCSV=pd.DataFrame.to_csv(df,sep=',',line_terminator='\n')
    csvout = open(csvout, 'w')      # opens the file to write into
    csvout.write(toCSV)             # writes df to csv... 
    print("Database transferred to csv...")
    csvout.close()
"""****************************************************************************
Create a color gradient for graphs
****************************************************************************"""
def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness   = (50 + np.random.rand() * 10) / 100.
        saturation  = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

# run the program, which wrangles the data into the correct format.			
#inOrgClean = cord2stationID(inOrg,inOrg.lat,inOrg.lon)
#inOrgClean.drop(inOrgClean[['year','month','second']],axis=1,inplace=True)
#convert2CSV(inOrgClean,'inOrganicNutrientsClean.csv')

# find the distance between observations
yearStart = ['1993-01-00 00:00:00','1994-01-00 00:00:00','1995-01-00 00:00:00','1996-01-00 00:00:00','1997-01-00 00:00:00','1998-01-00 00:00:00','1999-01-00 00:00:00','2000-01-00 00:00:00',\
'2001-01-00 00:00:00','2002-01-00 00:00:00','2003-01-00 00:00:00','2004-01-00 00:00:00','2005-01-00 00:00:00','2006-01-00 00:00:00','2007-01-00 00:00:00','2008-01-00 00:00:00','2009-01-00 00:00:00',\
'2010-01-00 00:00:00','2011-01-00 00:00:00','2012-01-00 00:00:00','2013-01-00 00:00:00', '2014-01-00 00:00:00']
yearEnd =   ['1993-12-30 00:00:00','1994-12-30 00:00:00','1995-12-30 00:00:00','1996-12-30 00:00:00','1997-12-30 00:00:00','1998-12-30 00:00:00','1999-12-30 00:00:00','2000-12-30 00:00:00',\
 '2001-12-30 00:00:00','2002-12-30 00:00:00','2003-12-30 00:00:00','2004-12-30 00:00:00','2005-12-30 00:00:00','2006-12-30 00:00:00','2007-12-30 00:00:00','2008-12-30 00:00:00','2009-12-30 00:00:00',\
 '2010-12-30 00:00:00','2011-12-30 00:00:00','2012-12-30 00:00:00','2013-12-30 00:00:00','2014-12-30 00:00:00']

# Global variables
skip 		= -999
lblSize 	= 12

years       = np.array(inOrg.year.unique())
yearVar     = len(years)
#colorSwatch = np.array(_get_colors(yearVar))
colorSwatch = cm.gray(np.linspace(0, 1, yearVar))
#tups        = [years,colorSwatch]
color = pd.DataFrame(colorSwatch,columns=['hue','lightness','saturation','other'])
color['year'] = years
#color.hue = 0

mark      	=   ['o','p','d','v']

#aveDelLat = []
#aveDelLon = []
#lowDelLat = []
#lowDelLon = []
#superLatList = []
#superLonList = []
#years = len(yearStart)
#for i in range(years):
#    q = inOrg.loc[inOrg.datetime>yearStart[i]]
#    q = q.loc[q.datetime<yearEnd[i]]
#    if q.empty:
#        break
#    qLat = q.lat.unique()
#    qLon = q.lon.unique()
#    qLat.sort()
#    qLon.sort()
#    
#    latList = []
#    lonList = []
#    
#    for j in range(len(qLat)-1):
#        latList.append(round(qLat[j]-qLat[j+1],SF))
#    aveLat = sum(latList)/(len(qLat)-1)
#    uppDel = min(latList)
#    lowDel = max(latList)
#    
#    superLatList.append(latList)        
#    aveDelLat.append(aveLat)
#    lowDelLat.append(lowDel)
#    
#    for j in range(len(qLon)-1):
#        lonList.append(round(qLon[j]-qLon[j+1],SF))
#    aveLon = sum(lonList)/(len(qLon)-1)
#    uppDel = min(lonList)
#    lowDel = max(lonList)
#    
#    superLonList.append(lonList)
#    aveDelLon.append(aveLon)
#    lowDelLon.append(lowDel)
#boundLat = max(lowDelLat)
#boundLon = max(lowDelLon)

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

nutrient = inOrg[inOrg.Nitrite!=skip]
nutrient = nutrient[nutrient.Nitrite.notnull()]
nutrient = nutrient.reset_index()
stationID = nutrient.stationID.unique()
#stationID.sort()
stationLen = len(stationID)
for i in range(stationLen):
	# get all values in specific station for all stationVar
	stationData    = nutrient[nutrient.stationID==stationID[i]]
	stationTime    = stationData.datetime.unique()
	stationVar     = len(stationTime)
	if(stationVar <= 6):
		print(stationID[i],"has little/no variability")
	else:
		plt.figure(figsize=(4,8))
		maxDep = stationData.depth.max()*-1
		plt.ylim([maxDep-20,0])
		plt.xlabel(r'PO4 [$\mu$m/L]',size=lblSize )
		plt.ylabel('Water column height [m]',size=lblSize )
		for j in range(stationVar):
			# get only values from selected stationInterannualVar
			target    = stationData[stationData.datetime==stationTime[j]]
#			eventList = target.datetime.unique()
#			events    = len(eventList)
			if (len(target.index)<=3):
				break
			else:
#			if (events>1):
#				for k in range(events):
#					subtarget = target[target.datetime==eventList[k]]
#					x = subtarget.Nitrite.values
#					y = subtarget.depth.values*-1
#					labels = np.asarray(subtarget.year, dtype=np.float)[0]
#					plt.scatter(x,y ,marker=mark[k%4],s=12,color=colorSwatch[k%10],alpha=.7,zorder=10,label=labels)
#					plt.plot(x,y,color=colorSwatch[k%10])
##			target = target.sort(columns='depth')
	#			else:
				x = target.Nitrite.values
				y = target.depth.values*-1
				colorYear = (color[['hue','lightness','saturation']][color.year==np.asarray(target.year, dtype=np.float)[0]]).values
				labels = round(np.asarray(target.year.unique(), dtype=np.float)[0],-1)
				plt.scatter(x,y ,marker=mark[j%4],s=12,color=colorSwatch[j%14],alpha=.7,zorder=10,label=labels)
				plt.plot(x,y,color=colorSwatch[j%14])
		art = []		
		plt.legend(scatterpoints=1,
			           loc='lower left',
			           ncol=3,
			           fontsize=8)
		plt.title(r'Nitrite at '+str(np.asarray(target.lat, dtype=np.float)[0])+'W,'+str(np.asarray(target.lon, dtype=np.float)[0])+'S',size=16)
		plt.show()
## plotting   
#cordLon = inOrg.lon.unique()
#cordLon.sort()
#cordLonL = len(cordLon)
#for i in range(cordLonL):
#    temp1 = inOrg.loc[inOrg.lon==cordLon[i]]
##    temp1 = inOrg.loc[inOrg.lon>=cordLon[i]-.015] # lower bound
##    temp1 = temp1.loc[temp1.lon<=cordLon[i]+.015] # uppwer bound
#    cordLat = temp1.lat.unique() #check if this is just 1...
#    cordLatL = len(cordLat)
#    for j in range(cordLatL):
##        print "Coordinate",cordLat[j],cordLon[i]
#        temp2 = temp1.loc[temp1.lat==cordLat[j]]
##        temp2 = temp1.loc[temp1.lat>=cordLat[j]-.01]
##        temp2 = temp2.loc[temp2.lat<=cordLat[j]+.01]
#        temp2 = temp2[temp2.Nitrite!=skip]
#        temp2 = temp2[temp2.Nitrite.notnull()]
#        cordTime = temp2.datetime.unique()
#        cordTimeL = len(cordTime)
#        if (cordTimeL<6):
#            break
#        if (len(temp2.Nitrite.unique())<6):
#            break
#        # intialize plot
#        plt.figure(figsize=(4,8))
#        maxDep = temp2.depth.max()*-1
#        plt.ylim([maxDep-20,0])
#        plt.xlabel(r'PO4 [$\mu$m/L]',size=lblSize )
#        plt.ylabel('Water column height [m]',size=lblSize )
#        for k in range(cordTimeL):
#            search = temp2.loc[temp2.datetime==cordTime[k]]
##            print search.datetime.unique()
#            x = search.Nitrite.values
#            y = search.depth.values*-1
#            labels = str(pd.DatetimeIndex(search['datetime']).year[0])
#            plt.scatter(x,y ,marker=mark[k%4],s=12,\
#                               color=colorSwatch[k%10],
#                                alpha=.7,zorder=10,label=labels)
#            plt.plot(x,y,color=colorSwatch[k%10])
#        
#        
##        plt.legend(scatterpoints=1,
##           loc='lower left',
##           ncol=3,
##           fontsize=8)
#        art = []
#        lgd = plt.legend(loc='upper right',scatterpoints=1,ncol=1,
#                            bbox_to_anchor=(1.7, 1.00),prop={'size':10})
#
#        plt.title(r'PO4 at '+str(-1*round(cordLat[j],SF))+'W,'+str(-1*round(cordLon[i],SF))+'S',size=16)
#        plt.show()


#plt.figure(figsize=(12,12)) 
#map = Basemap(width=1600000,height=900000,
#            resolution='l',projection='stere',\
#            lat_ts=50,lat_0=-65.90,lon_0=-66.05)
#parallels = np.arange(-90.,90.,1.)
#map.drawparallels(parallels,labels=[False,False,False,True])
#meridians = np.arange(-180.,181.,1.)
#map.drawmeridians(meridians,labels=[True,False,False,False])
#map.drawcoastlines()
#map.fillcontinents()
#map.drawmapboundary()
#
#search = inOrgClean[inOrgClean.NO3!=skip]
##search = search[search.year==2014]
#search = search[search.NO3.notnull()]
#
#x1 = search.lon.values.T.tolist()
#y1 = search.lat.values.T.tolist() 
#z1 = search.NO3.values.T.tolist()
#xS, yS = map(x1, y1)
#map.scatter(xS, yS, c=z1, marker='o',cmap='jet',s=12,linewidth=.08,alpha=.9)
#cbar = plt.colorbar(orientation='vertical',fraction=0.026, pad=0.04)
#plt.title('Antartic InOrganic NO2',size=16)
##plt.clim(0,0.5)
#plt.show()

plt.figure(figsize=(12,12)) 
map = Basemap(width=1600000,height=900000,
            resolution='l',projection='stere',\
            lat_ts=50,lat_0=-65.90,lon_0=-66.05)
parallels = np.arange(-90.,90.,1.)
map.drawparallels(parallels,labels=[False,False,False,True])
meridians = np.arange(-180.,181.,1.)
map.drawmeridians(meridians,labels=[True,False,False,False])
map.drawcoastlines()
map.fillcontinents()
map.drawmapboundary()

search = inOrg[inOrg.NO3!=skip]
#search = search[search.year==2014]
search = search[search.NO3.notnull()]

x1 = search.lon.values.T.tolist()
y1 = search.lat.values.T.tolist() 
z1 = search.NO3.values.T.tolist()
xS, yS = map(x1, y1)
map.scatter(xS, yS, c=z1, marker='o',cmap='jet',s=12,linewidth=.08,alpha=.9)
cbar = plt.colorbar(orientation='vertical',fraction=0.026, pad=0.04)
plt.title('Antartic InOrganic NO2',size=16)
#plt.clim(0,0.5)
plt.show()



#cords.lat = inOrg.lat.round(decimals=1,out=None).unique()
#for i in range(len(cords)):

    