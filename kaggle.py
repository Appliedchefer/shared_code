import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import mpl_toolkits.basemap as Basemap
import pandas as pd
import matplotlib.patheffects as PathEffects

def nice_string_output(names, values, extra_spacing = 2):
    """ Function to create a nice string output for figures """

    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values) :
        string += "{0:s} {1:>{spacing}} \n".format(name, value,
                   spacing = extra_spacing + max_values + max_names - len(name))

    return string[:-2]

ufo_data = pd.read_csv('scrubbed.csv', usecols=[0, 1, 2, 9, 10], low_memory=False)
ufo_data['datetime'] = pd.to_datetime(ufo_data['datetime'], errors='coerce')
ufo_data.insert(1, 'year', ufo_data['datetime'].dt.year)
ufo_data['year'] = ufo_data['year'].fillna(0).astype(int)
ufo_data['city'] = ufo_data['city'].str.title()
ufo_data['state'] = ufo_data['state'].str.upper()
ufo_data['latitude'] = pd.to_numeric(ufo_data['latitude'], errors='coerce')
ufo_data = ufo_data.rename(columns={'longitude ':'longitude'})

us_states = np.asarray(['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                        'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                        'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                        'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                        'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'])
print (len(us_states))

#in millions
population_state = [0.739795, 4.874747, 3.004279, 7.016270, 39.536653, 5.607154,
3.588184, 0.693972, 0.961939, 20.984400, 10.429379, 1.427538, 3.145711,
1.716943, 12.802023, 6.666818, 2.913123, 4.454189, 4.684333, 6.859819, 6.052177, 1.335907,
9.962311, 5.576606, 6.113532, 2.984100, 1.050493, 10.273419, 0.755393,
1.920076, 1.342795, 9.005644, 2.088070, 2.998039, 19.849399, 11.658609, 3.930864, 4.142776,
12.805537, 1.059639, 5.024369, 0.869666, 6.715984, 28.304596,
3.101833, 8.470020, 0.623657, 7.405743, 5.795483, 1.815857, 0.579315]

print (len(population_state))

# UFO sightings in United States only (70,805 rows)
ufo_data = ufo_data[ufo_data['state'].isin(us_states)].sort_values('year')
ufo_data = ufo_data[(ufo_data.latitude > 15) & (ufo_data.longitude < -65)]
ufo_data = ufo_data[(ufo_data.latitude > 50) & (ufo_data.longitude > -125) == False]
ufo_data = ufo_data[ufo_data['city'].str.contains('\(Canada\)|\(Mexico\)') == False]

# Create subsets for selected states
az_ufo_data = ufo_data[ufo_data['state'].str.contains('AZ') == True]
fl_ufo_data = ufo_data[ufo_data['state'].str.contains('FL') == True]
ny_ufo_data = ufo_data[ufo_data['state'].str.contains('NY') == True]
oh_ufo_data = ufo_data[ufo_data['state'].str.contains('OH') == True]

month       = [] # Integer
day         = [] # Integer
year        = [] # Integer
time        = [] # Integer (24 hr clock on the form hhmm)
city        = [] # String
state       = ufo_data[ufo_data['state'].isin(us_states)].state.tolist() # String
#print state
shape       = [] # String
duration    = [] # String (needs to be parsed later)
description = [] # String



split = Counter(state)

state_US = []
for i in us_states:
    state_US.append(split[i])

state_US = np.array(state_US)
population_state = np.array (population_state)
#print population_state
state_US_normalized_to_pop = state_US/population_state
#print len(state_US_normalized_to_pop)
#print len(state_US)

plt.clf()
state_US_max = np.max(state_US)
state_US_normalized_to_pop_max = np.max(state_US_normalized_to_pop)

US_norm_max = state_US/float(state_US_max)
pop_norm_max = state_US_normalized_to_pop/state_US_normalized_to_pop_max
#print (US_norm_max,"\n", pop_norm_max)

KS_test = stats.ks_2samp(US_norm_max, pop_norm_max)
#print (KS_test)
width = 0.7
index = np.linspace(0,102,51)
print (index)

fig, ax = plt.subplots(figsize = (13,7))
ax.bar(index, US_norm_max, width = 0.7, color = "g", align = "center", label = "Observations in US")
ax.bar(index+width, pop_norm_max, width = 0.7, color = "r", align = "center", label = "Observations in US normalized to population")
plt.xticks(index + width, us_states, rotation=90)
plt.xlabel("States")
plt.ylabel("Normalized frequency")
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig("population.png")

plt.clf()
index_2 = np.linspace(0,8,5)
print (index_2)
fig, ax = plt.subplots(figsize = (13,7))
ax.bar(index_2, US_norm_max[0:5], width = 0.7, color = "g", align = "center", label = "Observations in US")
ax.bar(index_2+width, pop_norm_max[0:5], width = 0.7, color = "r", align = "center", label = "Observations in US normalized to population")
plt.xticks(index_2 + width, us_states[0:5], rotation=90)
plt.xlabel("States")
plt.ylabel("Normalized frequency")
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig("population_only5.png")

ca_ufo_data = ufo_data[ufo_data['state'].str.contains("CA") == True]
ca_ufo_years = ufo_data[ufo_data['year']!=0]

#ufo_years = ufo_data[ufo_data.year != 0]

#groupby_year = ufo_years['year'].groupby(ufo_years['year']).count().plot(kind='line')


ca_ufo_data.describe()

lat_SD = 32.82
lon_SD = -117.13
plt.figure(figsize=(12,8))
CA = Basemap.Basemap(projection='mill', llcrnrlat = 32, urcrnrlat = 43, llcrnrlon = -125, urcrnrlon = -114, resolution = 'c')
CA.drawcoastlines()
CA.drawcountries()
CA.drawstates()
x, y = CA(list(ca_ufo_data["longitude"].astype("float")), list(ca_ufo_data["latitude"].astype(float)))

x_SD, y_SD = CA(lat_SD, lon_SD)
#CA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "green")
CA.scatter(x, y, c = range(0,len(x)), cmap = "Reds")
plt.title('UFO Sightings in California')
plt.colorbar()
plt.tight_layout()

#plt.text(x_SD, y_SD, nice_string_output(["San Diego"], [""]),fontsize=12, transform=ax.transAxes, fontweight = 'bold', verticalalignment='top', color = "black")#,fontweight='bold')
names2 = ['San Diego']
values2 = [""]
names3 = ['Los Angeles']
values3 = [""]
names4 = ['San Fransisco']
values4 = [""]
names5 = ['Sacramento']
values5 = [""]
names6 = ['Fresno']
values6 = [""]
txt_1 = plt.text(0.55, 0.08, nice_string_output(names2, values2), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
txt_2 = plt.text(0.5, 0.18, nice_string_output(names3, values3), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
txt_3 = plt.text(0.3, 0.6, nice_string_output(names4, values4), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
txt_4 = plt.text(0.35, 0.67, nice_string_output(names5, values5), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
txt_5 = plt.text(0.45, 0.45, nice_string_output(names6, values6), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
txt_1.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
txt_2.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
txt_3.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
txt_4.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
txt_5.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

plt.savefig("please_work.png")


"""
plt.clf()

#plt.figure (figsize = (12,8))
#plt.plot(x_SD, y_SD, "go", markersize = 4, alpha = 0.8, color = "green")
#plt.text(x_SD, y_SD, 'San Diego',fontsize=12,fontweight='bold',
                    #ha='left',va='bottom',color='k')
#plt.savefig("works_here_perhaps.png")

plt.clf()



map = Basemap.Basemap(projection='ortho',
              lat_0=0, lon_0=0)

map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='#cc9955',lake_color='aqua')
map.drawcoastlines()

lon = 3.4
lat = 3.4

x, y = map(lon, lat)

plt.text(x, y, 'Lagos',fontsize=12,fontweight='bold',
                    ha='left',va='bottom',color='k')

lon = 2.1
lat = 41.

x, y = map(lon, lat)

plt.text(x, y, 'Barcelona',fontsize=12,fontweight='bold',
                    ha='left',va='center',color='k',
                    bbox=dict(facecolor='b', alpha=0.2))
plt.savefig("test.png")
plt.clf()
"""

plt.figure(figsize=(12,8))
USA = Basemap.Basemap(projection='mill', llcrnrlat = 23, urcrnrlat = 50, llcrnrlon = -125, urcrnrlon = -65, resolution = 'h')
USA.drawcoastlines()
USA.drawcountries()
USA.drawstates()
x, y = USA(list(ufo_data["longitude"].astype("float")), list(ufo_data["latitude"].astype(float)))
#USA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "green")
USA.scatter(x, y, c = range(0,len(x)), cmap = "Reds")
plt.title('UFO Sightings in the US')
plt.colorbar()
plt.tight_layout()
plt.savefig("ygnaknak.png")







ca_ufo_data = ufo_data[ufo_data['state'].str.contains("CA") == True]
ca_ufo_years = ufo_data[ufo_data['year']!=0]
print len(ca_ufo_data)
ca_ufo_data.describe()

ca_video_data = ufo_data[ufo_data['state'].str.contains("CA") == True]
print len (ca_video_data)

ca_video_years = ufo_data[ufo_data['year']>=1930]
print len(ca_video_years)
ca_video_years.describe()

ca_video_lon = ca_video_data.longitude
ca_video_lat = ca_video_data.latitude
print len(ca_video_lon)
print ca_video_lon[ca_video_data['year'] <= 1930]















# to get a list that only contains the data for 1930 and up and also for the lon and lat.
ufo_data_year = ufo_data.year[ufo_data.year >= 1930]
print len(ufo_data_year)

ufo_data_lat = ufo_data.latitude[ufo_data.year >= 1930]
print len(ufo_data_lat)

ufo_data_lon = ufo_data.longitude[ufo_data.year >= 1930]
print


ufo_data.describe()





plt.clf()
plt.figure(figsize=(12,8))
CA_video = Basemap.Basemap(projection='mill', llcrnrlat = 32, urcrnrlat = 43, llcrnrlon = -125, urcrnrlon = -114, resolution = 'c')
CA_video.drawcoastlines()
CA_video.drawcountries()
CA_video.drawstates()
x, y = CA_video(list(vid_1930_lon), list(vid_1930_lat))

x_SD, y_SD = CA(lat_SD, lon_SD)
#CA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "green")
CA.scatter(x, y, c = range(0,len(x)), cmap = "Reds")
plt.title('UFO Sightings in California before 1930')
plt.colorbar()
plt.tight_layout()

#plt.text(x_SD, y_SD, nice_string_output(["San Diego"], [""]),fontsize=12, transform=ax.transAxes, fontweight = 'bold', verticalalignment='top', color = "black")#,fontweight='bold')
names2 = ['San Diego']
values2 = [""]
names3 = ['Los Angeles']
values3 = [""]
names4 = ['San Fransisco']
values4 = [""]
names5 = ['Sacramento']
values5 = [""]
names6 = ['Fresno']
values6 = [""]
txt_1 = plt.text(1.75, 0.08, nice_string_output(names2, values2), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
txt_2 = plt.text(1.55, 0.25, nice_string_output(names3, values3), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
txt_3 = plt.text(0.9, 0.95, nice_string_output(names4, values4), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
txt_4 = plt.text(1.0, 1.1, nice_string_output(names5, values5), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
#txt_5 = plt.text(0.9, 0.95, nice_string_output(names6, values6), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
txt_1.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
txt_2.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
txt_3.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
txt_4.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
#txt_5.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

plt.savefig("try_this.png")




for i in test_list_2:
    vid_lon = ca_video_lon[ca_video_data['year'] <= i]
    vid_lat = ca_video_lat[ca_video_data['year'] <= i]
    print len(vid_lat)


plotting_list = np.arange(1930, 2016, 1)

#print (plotting_list)
test_list = [2015]#, 1935, 1940, 1945, 2000, 1930]
for i in plotting_list:
    vid_lon = ca_video_lon[ca_video_data['year'] <= i]
    vid_lat = ca_video_lat[ca_video_data['year'] <= i]
    plt.figure(figsize=(12,8))
    CA_video = Basemap.Basemap(projection='mill', llcrnrlat = 32, urcrnrlat = 43, llcrnrlon = -125, urcrnrlon = -114, resolution = 'c')
    CA_video.drawcoastlines()
    CA_video.drawcountries()
    CA_video.drawstates()
    x, y = CA_video(list(vid_lon), list(vid_lat))


    #CA.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "green")
    CA.scatter(x, y, c = range(0,len(x)), cmap = "Reds")
    plt.title('UFO Sightings in California from 1930 - 2015')
    plt.colorbar()
    plt.tight_layout()

    names2 = ['San Diego']
    values2 = [""]
    names3 = ['Los Angeles']
    values3 = [""]
    names4 = ['San Fransisco']
    values4 = [""]
    names5 = ['Sacramento']
    values5 = [""]
    names6 = ['Fresno']
    values6 = [""]
    txt_1 = plt.text(0.55, 0.08, nice_string_output(names2, values2), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
    txt_2 = plt.text(0.5, 0.18, nice_string_output(names3, values3), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
    txt_3 = plt.text(0.3, 0.6, nice_string_output(names4, values4), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
    txt_4 = plt.text(0.35, 0.67, nice_string_output(names5, values5), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
    txt_5 = plt.text(0.45, 0.45, nice_string_output(names6, values6), family='monospace', transform=ax.transAxes, fontsize=12, fontweight = 'bold', verticalalignment='top', color = "black")
    txt_1.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    txt_2.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    txt_3.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    txt_4.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    txt_5.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    plt.savefig("video" +str(i)+ ".png")
    plt.clf()



#TO MAKE A VIDEO RUN THIS IN TERMINAL!
#ffmpeg -framerate 7 -pattern_type glob -i 'video*.png' out.mp4
#https://trac.ffmpeg.org/wiki/Slideshow to understand wtf is going on



#
