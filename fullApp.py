#!/usr/bin/env python

# Environment
import xcdat
import os
import xarray as xr
import dash
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
from jupyter_dash import JupyterDash
import plotly
import plotly.graph_objects as go
import plotly.express as px
states_file = './shp/cb_2018_us_state_20m.shp' # https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
global states_df
states_df = gpd.read_file(states_file)
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
xr.set_options(keep_attrs=True)

import sys
sys.path.insert(0, './func/')

from stats import rmsc, rms, mae, bias, cor_xy, std


# NCA4 region dictionary with STUPS codes

nca4_regions = {'Northern Great Plains': ['MT', 'NE', 'SD', 'ND', 'WY'],             'Southern Great Plains': ['TX', 'OK', 'KS'],             'Southwest': ['CA', 'NV', 'AZ', 'NM', 'CO', 'UT'],             'Southeast': ['FL', 'GA', 'SC', 'MS', 'AL', 'AR', 'TN', 'KY', 'NC', 'VA', 'LA'],             'Northeast': ['WV', 'DC', 'PA', 'MA', 'NY', 'CT', 'RI', 'ME', 'NJ', 'VT', 'NH', 'MD', 'DE'],             'Northwest': ['WA', 'OR', 'ID'],             'Midwest':   ['IL', 'MI', 'WI', 'MO', 'IA', 'IN', 'OH', 'MN']}
            #'Alaska':['AK'], \
            #'Hawai\'i and Pacific Islands':['HI'], #, 'AS', 'GU', 'MP'], 
            #'Carribean': ['PR', 'VI']}


# adding an NCA4 region column to geoDF and creating a mask
nca_state = []
region_names = []
all_states = list(states_df['STUSPS'])

for state in states_df['STUSPS']:
    for region in nca4_regions.keys(): 
        if state in nca4_regions[region]: 
            all_states.remove(state)
            nca_state.append(region)
for not_contained_state in all_states:
    states_df = states_df[states_df['STUSPS'] != not_contained_state]
            
states_df['NCA4_region'] = nca_state

# Regional and State mask
regional_df = states_df.dissolve(by = 'NCA4_region')
regional_df['NAME'] = regional_df.index
regional_df = regional_df.reset_index(drop = True)

gpd_dict = {'NCA4 Region':regional_df, 'States':states_df}

nca_mask    = regionmask.from_geopandas(regional_df, names = "NAME", name = "NCA4 regions")
state_mask  = regionmask.from_geopandas(states_df, names = "STUSPS", name = "States")

# Continental US mask
us = gpd.read_file('./shp/cb_2018_us_nation_20m.shp')
us_mask = regionmask.from_geopandas(us, names = "NAME", name = "United States")
region_dict = {'NCA4 Region': nca_mask, "States": state_mask, 'United States': us_mask}


region_dict = {'NCA4 Region': nca_mask, "States": state_mask, 'United States': us_mask}


from bs4 import BeautifulSoup
import requests

def listFD(url, ext=''):
    """
    list subdirectories of given url data host
    """

    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

## NERSC-hosted data

source_diri = {'loca2':'https://portal.nersc.gov/cfs/m2637/cddData/loca2/NCA5/', 'star':'https://portal.nersc.gov/cfs/m2637/cddData/star/NCA5/'}

# dictionary with file names for a given model, simulation, variable, and member. May need to add time period as another differentiator once SSP simulations are included
method_vars = {}
method_models = {}
method_sims = {}
method_members = {}

method_files = {}

for method in list(source_diri.keys()):
    diri = source_diri[method]
    model_list = []
    sim_list   = []
    var_list   = []
    member_list= []
    fname_dict = {}


    for fpath in listFD(source_diri[method], '.nc'):
        if fpath.startswith('.') or ('mask' in fpath):
            pass
        else:
            fname = fpath.split('/')[-1]
            var   = fname.split('.')[0]
            #print(var)
            model = fname.split('.')[1]
            sim   = fname.split('.')[2]
            member= fname.split('.')[3]
            var_list.append(var)
            model_list.append(model)
            sim_list.append(sim)
            member_list.append(member)

            key = (var, model, sim, member)

            fname_dict[key] = source_diri[method] + fname 

    variables = np.unique(var_list) 
    models    = np.unique(model_list)
    sims      = np.unique(sim_list)
    members   = np.unique(member_list)
    
    method_vars[method]    = variables
    method_models[method]  = models
    method_sims[method]    = sims
    method_members[method] = members
    method_files[method]   = fname_dict


def greater_less_correction(da, fname): ## fillValue incorrectly input as 1.0 in file creation
    """ 
    removes spurious 1.0 fillValue in threshold files
    """
    
    if ('_ge_' in fname) or ('_le_' in fname):
        return da.where(da!=1.0)
    else:
        return da
    
def temp_conversion2(da, fname): 
    """ 
    convert to Kelvin to Celcius
    """ 
    
    if ('txx' in fname) or ('tasmax' in fname) or ('tasmean' in fname) or ('tnn' in fname) or ('tasmin' in fname) or ('TMAX' in fname) or ('TMIN' in fname) or ('TMEAN' in fname):
        if ('_ge_' not in fname) & ('_le_' not in fname) & ('PRISM' in fname) & ('STAR' not in fname):
            da += 273.15
            return da
        else:
            return da 
    else:
        return da
    
def precip_conversion(da, fname): # to mm 
    """ 
    Convert from kg/m2/s to mm
    """
    
    if ('pr' in fname) or ('pxx' in fname): 
        if ('PRISM' not in fname) and ('STAR' not in fname):
            da*=86400
            return da
        else:
            return da
    else:
        return da
    
def tnn_correction(da, fname):
    """ 
    removes high fillValue in threshold files
    """
    if 'annual_tnn' in fname: 
        return da.where(da<500)
    else:
        return da

def dataCorrection(da, fname): 
    d = greater_less_correction(temp_conversion2(precip_conversion(da, fname),fname), fname)
    return d

def getLatLonNames(d): 
    try:
        lat_name = [i for i in list(d.coords) if 'lat'  in i][0]
        lon_name = [i for i in list(d.coords) if 'lon' in i][0]
    except:
        lat_name = [i for i in list(d.dims) if 'lat'  in i][0]
        lon_name = [i for i in list(d.dims) if 'lon' in i][0]
    return lat_name, lon_name

    
def getData(fname, seas, mask, qvar = None):
    """
	download data from NERSC and return
	return: xArray dataArray
    """ 
 
    if 'quantile' in fname: 
        decode_times = False
    else:
        decode_times = True
    
    estr = ''
    if fname.startswith('https'):
        estr = '#mode=bytes'
    
    data = xcdat.open_dataset(fname + estr, decode_times = decode_times)
    lat_name, lon_name = getLatLonNames(data)
    
    ds_mask = mask.mask_3D(data.to_array(), lat_name=lat_name, lon_name=lon_name)
    data = data.where(ds_mask)
        
    try:
        if len(data.time.data) == 1 and (seas != 'annual'): 
            raise ValueError('Not seasonal data')
        if len(data.time.data) > 1 and (seas == 'annual'):
            #print(data.time)
            raise ValueError('Not annual data')
    except:
        pass
    
    if 'quantile' in fname: 
        if qvar is None: 
            raise ValueError('Provide quantile level')
        dvar = qvar
    else:
        dvar = list(data.keys())[0]
    # print(dvar)
    
    if 'quantile' in fname: 
        #print('here')
        return dataCorrection(data[dvar], fname)
    
    if seas == 'annual':
        try:
            return dataCorrection(data[dvar].isel(time = 0), fname)
        except:
            return dataCorrection(data[dvar], fname)
        
    else:
        return dataCorrection(data[dvar].isel(time = seas), fname)

from mpl_toolkits.basemap import Basemap
from plotly.graph_objects import Scatter
from plotly.graph_objs.scatter import Line

def make_scatter(x,y):
    """ 
    return Plotly.Scatter object of given cartographic boundary
    """ 

    return Scatter(
        x=x,
        y=y,
        mode='lines',
        line=Line(color="black"),
        name='',  # no name on hover
        showlegend = False, 
        hoverinfo = 'skip'
    )

# Functions converting coastline/country polygons to lon/lat traces. https://plotly.com/python/v3/ipython-notebooks/basemap-maps/
def polygons_to_traces(m, poly_paths, N_poly):
    ''' 
    pos arg 1. (poly_paths): paths to polygons
    pos arg 2. (N_poly): number of polygon to convert
    '''
    traces = []  # init. plotting list 

    for i_poly in range(N_poly):
        poly_path = poly_paths[i_poly]
        
        # get the Basemap coordinates of each segment
        coords_cc = np.array(
            [(vertex[0],vertex[1]) 
             for (vertex,code) in poly_path.iter_segments(simplify=False)]
        )
        
        # convert coordinates to lon/lat by 'inverting' the Basemap projection
        lon_cc, lat_cc = m(coords_cc[:,0],coords_cc[:,1], inverse = True)

        traces.append(make_scatter(lon_cc,lat_cc))
     
    return traces

# Function generating coastline lon/lat traces
def get_coastline_traces(m):
    poly_paths = m.drawcoastlines().get_paths() # coastline polygon paths
    N_poly = len(poly_paths)  # use only the 91st biggest coastlines (i.e. no rivers)
    return polygons_to_traces(m, poly_paths, N_poly)

# get state traces 
def get_states_traces(m): 
    poly_paths = m.drawstates().get_paths()
    N_poly = len(poly_paths)
    return polygons_to_traces(m, poly_paths, N_poly)


# Function generating country lon/lat traces
def get_country_traces(m):
    poly_paths = m.drawcountries().get_paths() # country polygon paths
    N_poly = len(poly_paths)  # use all countries
    return polygons_to_traces(m, poly_paths, N_poly)

def scatterMap(dataArray):
    """
    create Scatter points of cartgraphic boundaries for given data
    """ 

    offset = 0 

    min_lat = min(dataArray.lat.data)-1
    max_lat = max(dataArray.lat.data)+1
    min_lon = min(dataArray.lon.data)-1
    max_lon = max(dataArray.lon.data)+1 #calculateLonOffset(dx)
    
    m       = Basemap(llcrnrlon=min_lon, llcrnrlat=min_lat, urcrnrlat=max_lat, urcrnrlon=max_lon, resolution = 'l')
    traces  = get_coastline_traces(m)+get_country_traces(m)+get_states_traces(m)
    
    return traces

def get_xrbounds(da): 
    try:
        lat = da.lat.data
        lon = da.lon.data 
    except:
        lat = da.latitude.data
        lon = da.longitude.data 
    # print(lat)
    dy = lat.max() - lat.min()

    mean_lat = (2*lat.max() + lat.min())/3 #(lat.max() + lat.min())/2
    dx = (lon.max()-lon.min())*np.cos(mean_lat*np.pi/180)
    
    if dy/dx<0.3: 
        dx*=0.5
                                      
                                
    return [dy, dx]

def get_z(data, diffBool): 
    
    stdev = np.nanstd(data)
    
    mn = np.nanmin(data)
    mx = np.nanmax(data)
    
    if (mn*mx<0) & (diffBool): 
        return -4*stdev, 4*stdev
        # val = np.max([np.abs(mn), np.abs(mx)])
        # return -val, val
    else:
        return mn, mx
    
def calcDifference(data, diff, diffType): 
    if diffType == 'Relative': 
        return ((data-diff)/diff)*100
    else:
        return data-diff
    
# ## Creating a dictionary of border traces to improve runtime

trace_dict = {}
fname = method_files['loca2'][('annualmean_pr',
  'ACCESS-CM2',
  'historical',
  'r1i1p1f1')]
trace_dict = {}

for mask in region_dict.values():
    data = getData(fname, 'annual', mask, qvar = None)
    for region_name in data.region.names.data:
        region_ind   = np.where(data.region.names.data == region_name)[0] # index of the subregion 
        subData      = data.isel(region = region_ind).squeeze().dropna(dim = 'lat', how = 'all').dropna(dim = 'lon', how = 'all')
        trace_dict[region_name] = scatterMap(subData)
import matplotlib as ml

from matplotlib import cm
import numpy as np

# annual average colormaps
precip_cmap  = ml.colormaps.get_cmap('YlGnBu')
tmax_cmap    = ml.colormaps.get_cmap('Spectral_r')
tmin_cmap    = ml.colormaps.get_cmap('RdYlBu_r')
txx_cmap     = ml.colormaps.get_cmap('YlOrRd')
tnn_cmap     = ml.colormaps.get_cmap('BuPu_r')
tx_thresh    = ml.colormaps.get_cmap('Reds')
tn_thresh    = ml.colormaps.get_cmap('Blues')

# difference colormaps
dt_cmap      = ml.colormaps.get_cmap('coolwarm')
dp_cmap      = ml.colormaps.get_cmap('BrBG')

def matplotlib_to_plotly(cmap, pl_entries):
    """ 
    matplotlib standard cmap to plotly standard cmap
    """
    
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        #print(C)
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

def matplotlib_to_plotly_arr(cmap_arr, pl_entries): 
    """ 
    matplotlib cmap(linspace(x,x,255) to plotly standard cmap
    """
    
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    k = 0
    
    for row in cmap_arr:
        new_row = list(map(np.uint8, np.array(row[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((new_row[0], new_row[1], new_row[2]))])
        k+=1
    return pl_colorscale

# average colormaps
precip_cmap = matplotlib_to_plotly(precip_cmap, 255)
tmax_cmap   = matplotlib_to_plotly(tmax_cmap,   255)
tmin_cmap   = matplotlib_to_plotly(tmin_cmap,   255)
txx_cmap    = matplotlib_to_plotly(txx_cmap,    255)
tnn_cmap    = matplotlib_to_plotly(tnn_cmap,    255)
txt_cmap    = matplotlib_to_plotly(tx_thresh,   255)
tnt_cmap    = matplotlib_to_plotly(tn_thresh,   255)

# difference colormaps
dt_cmap     = matplotlib_to_plotly(dt_cmap,     255)
dp_cmap    = matplotlib_to_plotly(dp_cmap,      255)


colorbar_dict  = {}
dcolorbar_dict = {}
zmin_dict      = {}
zmax_dict      = {}

for vari in method_vars['star']: ## accounting for new STAR variable
    if ('_ge_' in vari):
        colorbar_dict[vari]  = txt_cmap
        dcolorbar_dict[vari] = dt_cmap
        zmin_dict[vari] = 0
        zmax_dict[vari] = 0.3
    elif ('_le_' in vari): 
        colorbar_dict[vari] = tnt_cmap
        dcolorbar_dict[vari] = dt_cmap
        zmin_dict[vari] = 0
        zmax_dict[var]  = 1
    elif ('tasmax' in vari) or ('tasmean' in vari):
        colorbar_dict[vari] = tmax_cmap
        dcolorbar_dict[vari] = dt_cmap
        zmin_dict[vari] = -10
        zmax_dict[vari] = 40
    elif ('txx' in vari): 
        colorbar_dict[vari] = txx_cmap
        dcolorbar_dict[vari] = dt_cmap
        zmin_dict[vari] = 20
        zmax_dict[vari] = 50
    elif ('tasmin' in vari):
        dcolorbar_dict[vari] = dt_cmap
        colorbar_dict[vari] = tmin_cmap
        zmin_dict[vari] = -10
        zmax_dict[vari] = 40
    elif ('tnn' in vari):
        dcolorbar_dict[vari] = dt_cmap
        colorbar_dict[vari] = tnn_cmap
        zmin_dict[vari] = -50
        zmax_dict[vari] = 0
    elif ('pxx' in vari) or ('pr' in vari): 
        if 'pxx' in vari: 
            zmin_dict[vari] = 0
            zmax_dict[vari] = 350
        else:
            zmin_dict[vari] = 0
            zmax_dict[vari] = 20
        colorbar_dict[vari] = precip_cmap
        dcolorbar_dict[vari] = dp_cmap

qlevs      = {10:'01',20:'05',30:'10',40:'25',50:'50',60:'75',70:'90',80:'95',90:'99',99:'99p9'}
label_dict = {'NCA4 Region': 'NAME', 'States': 'STUSPS'}


# ## Initial Figures

def genLayout(title_str): 
    layout = go.Layout(
      margin=go.layout.Margin(
            l=0, #left margin
            r=0, #right margin
            b=0, #bottom margin
            t=40, #top margin
        ), 
        title = title_str, 
        title_x = 0.5
    )
    return layout

method = 'loca2'
key = ('annual_pxx', 'ACCESS-CM2', 'historical', 'r1i1p1f1')
data = getData(method_files[method][key], 'annual', us_mask)
lat_name, lon_name = getLatLonNames(data)
fig_map  = px.choropleth(regional_df,
                           geojson=regional_df.geometry,
                           locations=regional_df.index,
                           #scope="usa",
                           labels='',
                           title = None, 
                           projection = 'mercator',
                           custom_data=['NAME'],
                           hover_data = None,
                           #center = {'lat':0, 'lon':-120}
                        ).update_geos(lataxis_range = [23, 55], 
                                      lonaxis_range = [-130, -60]).update_layout(showlegend=False, geo=dict(bgcolor= 'rgba(0,0,0,0)'), margin = {"r":0,"t":0,"l":0,"b":0})

contour = go.Figure(
    data = [go.Heatmap(
        x = data[lon_name], 
        y = data[lat_name],
        z = data.isel(region = 0).data,
        colorscale = colorbar_dict[key[0]], 
        zmin = zmin_dict[key[0]], 
        zmax = zmax_dict[key[0]], 
        zauto = False,
        colorbar = {'orientation':'v', 'bgcolor':'white'})] + scatterMap(data), #'xanchor':'right'
    layout = genLayout(''), 
#go.Layout(title='')#period + ' United States<br>' + title_str[variable],  title_x = 0.5)
)#.update_layout(width = '100%', height = '100%')


import dash
from dash import dcc, ctx
from dash import html
import dash_daq as daq



app = JupyterDash();




method = 'loca2'

key1 = list(method_files[method].keys())[0]
key2 = list(method_files[method].keys())[1]

dy, dx  = get_xrbounds(data)
width_p = '{:.2f}'.format((dy/dx)*100)
default_width = 750
height        = (dy/dx)*default_width
#width   = (dx/dy)*default_height

app.layout = html.Div([
    html.H1('Downscaled Climate Data', style = {'textAlign':'center', 'fontSize':50}), 
    # top row
    html.Div(className= 'optionBar', children = [
        html.Div(children = [
            html.H2('Region'),
            dcc.Graph(figure = fig_map, config={'displayModeBar':False}, id = 'usmap', style = {'height':235, 'width':370}),
        ], style = {'marginLeft':50}), 

        dcc.RadioItems(['NCA4 Region', 'States'], 'NCA4 Region', id = 'regionSelect', style = {'marginTop':75, 'fontSize':20}), 
        
        html.Div(children = [
            html.H2('Downscaled Dataset'), 
            #dcc.Dropdown(list(source_diri.keys()), 'loca2', multi = False,  id = 'dropMethod', style = {'width':250, 'fontSize':15}),
            dcc.Dropdown(method_vars['loca2'],  key1[0], multi = False, id = 'dropVar',  style = {'width':250,   'fontSize':15})
        ], style = {'marginLeft':50}), 
        
        dcc.RadioItems(options=[
               {'label': 'Spring (MAM)', 'value': 1},
               {'label': 'Summer (JJA)', 'value': 2},
               {'label': 'Fall (SON)',   'value': 3}, 
               {'label': 'Winter (DJF)', 'value': 0},
           ],
           value = 1, id = 'seasonSelect', style = {'display':'none'}),
        
        html.Div(children = [
            html.H2('Model 1', id = 'modelTitle1'),
            dcc.Dropdown(list(source_diri.keys()), 'loca2', multi = False,  id = 'dropMethod1', style = {'width':250, 'fontSize':15}),
            dcc.Dropdown(method_models['loca2'],     key1[1], multi = False, id = 'dropModel1',style = {'width':250,  'fontSize':15}),
            dcc.Dropdown(method_members['loca2'],    key1[3], multi = False, id = 'dropMem1',  style = {'width':250,  'fontSize':15}), 
            html.Div(dcc.Slider(0,100, step = None, value = 50, marks = qlevs, id = 'qsliderOne'), id = 'sliderContainerOne', style = {'display':'none'}), 
        ], style = {'marginLeft':50}), 
        html.Div(children = [
            html.H2('Model 2', id = 'modelTitle2'), 
            dcc.Dropdown(list(source_diri.keys()), 'loca2', multi = False,  id = 'dropMethod2', style = {'width':250, 'fontSize':15}),
            dcc.Dropdown(method_models['loca2'],     key2[1], multi = False, id = 'dropModel2',style = {'width':250, 'fontSize':15}),
            dcc.Dropdown(method_models['loca2'],     key2[3], multi = False, id = 'dropMem2',  style = {'width':250, 'fontSize':15}), 
            html.Div(dcc.Slider(0,100, step = None, value = 50, marks = qlevs, id = 'qsliderTwo'), id = 'sliderContainerTwo', style = {'display':'none'})
        ], style = {'marginLeft':50}),  
        
        dcc.RadioItems(['Absolute', 'Relative'], 'Absolute', id = 'diffType', style = {'marginLeft':20, 'marginTop':75, 'fontSize':20}),
        html.Button('Generate', id = 'graphGen', disabled = False, style={'fontSize': '24px', 'width': '140px', 'height':'100px','verticalAlign':'bottom', 'marginLeft':50, 'marginTop':75}), 
        html.Div(children = [
            html.H2('Summary Statistics', style = {'textAlign':'center'}), 
            html.Div(className = 'summaryStats', children = [
                html.Div(children = [
                    html.Div(id = 'rmsc_xy_out', style = {'fontSize':20}),
                    html.Div(id = 'rms_xy_out',  style = {'fontSize':20}), 
                    html.Div(id = 'mae_xy_out',  style = {'fontSize':20}), 
                    html.Div(id = 'bias_xy_out', style = {'fontSize':20}), 
                    html.Div(id = 'cor',         style = {'fontSize':20})]), 
            ], style = {'display':'flex', 'flexDirection':'row', 'height':200, 'width':400, 'marginLeft':20, 'marginTop':20}, id = 'statbox'), 
        ]), 
    ], style = {'display':'flex', 'flexDirection':'row', 'marginLeft':100, 'marginRight':100}), 
    # middle row
    html.Div(children = [
        dcc.Graph(figure = contour, id = 'contourOne',  style = {'height':height, 'width': default_width, 'marginLeft':0}),
        dcc.Graph(figure = contour, id = 'contourTwo',  style = {'height':height, 'width': default_width, 'marginLeft':70}), 
        dcc.Graph(figure = contour, id = 'contourDiff', style = {'height':height, 'width': default_width, 'marginLeft':70}),
    ], style = {'display':'flex', 'flexDirection':'row', 'marginTop':50, 'marginLeft':100}), #'alignItems': 'center', 'justifyContent': 'center'}), # 'margin-left':50, 
    
    dcc.Store(id = 'previous_click-store-contour1'),
    dcc.Store(id = 'previous_click-store-contour2'), 
    dcc.Store(id = 'dataArrStoreOne'), 
    dcc.Store(id = 'dataArrStoreTwo'),
])

# Callbacks that only deal with user input, no data
from dash import Input, Output, State
from dash import ctx
@app.callback(Output(component_id = 'dropVar', component_property = 'options'), 
              Input(component_id  = 'dropMethod1', component_property = 'value'), 
              Input(component_id  = 'dropMethod2', component_property = 'value'))

def updateVariables(method1, method2):
    var1 = method_vars[method1]
    var2 = method_vars[method2] 
    
    return np.sort(list(set(var1) | set(var2)))



@app.callback(Output(component_id = 'seasonSelect', component_property = 'style'), 
              Input(component_id  = 'dropVar',      component_property = 'value'))
def showSeason(variable): 
    if 'season' in variable: 
        return {'display':'block', 'marginTop':60}
    else:
        return {'display':'none'}
@app.callback(Output(component_id = 'dropModel1',  component_property = 'options'), 
              Input(component_id  = 'dropVar',    component_property = 'value'), 
              Input(component_id  = 'dropMethod1', component_property = 'value'), 
              prevent_initial_call = False)
def updateModel1(variable, method): 
    varlist = np.array([i[0] for i in list(method_files[method].keys())])
    modlist = np.array([i[1] for i in list(method_files[method].keys())])
    
    return  np.unique(modlist[np.where(variable == varlist)[0]])

@app.callback(Output(component_id = 'dropMem1',   component_property = 'options'), 
              Input(component_id  = 'dropModel1', component_property = 'value'), 
              Input(component_id  = 'dropMethod1', component_property = 'value'), 
             prevent_initial_call = False)
def updateMember1(model, method): 
    modlist = np.array([i[1] for i in list(method_files[method].keys())])
    memlist = np.array([i[3] for i in list(method_files[method].keys())])
    
    return  np.unique(memlist[np.where(model == modlist)[0]])

@app.callback(Output(component_id = 'dropModel2',  component_property = 'options'), 
              Input(component_id  = 'dropVar',    component_property = 'value'), 
              Input(component_id  = 'dropMethod2', component_property = 'value'), 
             prevent_initial_call = True)

def updateModel2(variable, method): 
    varlist = np.array([i[0] for i in list(method_files[method].keys())])
    modlist = np.array([i[1] for i in list(method_files[method].keys())])
    
    return  np.unique(modlist[np.where(variable == varlist)[0]])

@app.callback(Output(component_id = 'dropMem2',   component_property = 'options'), 
              Input(component_id  = 'dropModel2', component_property = 'value'), 
              Input(component_id  = 'dropMethod2', component_property = 'value'), 
             prevent_initial_call = False)
def updateMember2(model, method): 
    modlist = np.array([i[1] for i in list(method_files[method].keys())])
    memlist = np.array([i[3] for i in list(method_files[method].keys())])
    
    return  np.unique(memlist[np.where(model == modlist)[0]])

sliderDict = {'sliderContainerOne':'dropVar', 'sliderContainerTwo':'dropVar'}

for component in list(sliderDict.keys()):
    varInput = sliderDict[component]
    
    @app.callback(Output(component_id = component, component_property = 'style'), 
                  Input(component_id  = varInput,  component_property = 'value')) 
    
    def qUpdate(variable): 
        if 'quantile' in variable: 
            return {'display':'block'}
        else:
            return {'display':'none'}


def add_selection_col(df, region, subregion):
    label_dict = {'NCA4 Region': 'NAME', 'States': 'STUSPS'}
    
    col = ['Not Selected']*len(df['geometry'])
    ind = np.where(df[label_dict[region]] == subregion)[0][0]
    col[ind] = 'Selected'
    df['selection'] = col
    return df

# Choropleth handling
@app.callback(Output(component_id = 'usmap',        component_property = 'figure'),
              Input(component_id  = 'usmap',        component_property = 'clickData'),
              Input(component_id  = 'regionSelect', component_property = 'value'), 
              prevent_initial_call = False)

def select_region(click, region): 
    df = gpd_dict[region]
    default_height = 250
    callback = ctx.triggered_id
    
    try:
        if callback == 'usmap':
            subregion = click['points'][0]['customdata'][0]
            df = add_selection_col(df, region, subregion)
        else:
            df['selection'] = ['Not Selected']*len(df['geometry'])
        
        new_figure = px.choropleth(df,
                       geojson=df.geometry,
                       locations=df.index,
                       title = None,
                       hover_data = None, #[label_dict[region]], 
                       custom_data=[label_dict[region]],
                       color = df.selection,
                       color_discrete_map={'Selected':'royalblue', 'Not Selected':'powderblue'}, 
                       projection = 'mercator').update_geos(lataxis_range = [23, 55], 
                                                            lonaxis_range = [-130, -60]).update_layout(showlegend=False, geo=dict(bgcolor= 'rgba(0,0,0,0)'), margin = {"r":0,"t":0,"l":0,"b":0})
    except:
        return px.choropleth(df,
                           geojson=df.geometry,
                           locations=df.index,
                           title = None,
                           hover_data = None, #[label_dict[region]], 
                           custom_data=[label_dict[region]],
                           projection = 'mercator').update_geos(lataxis_range = [23, 55], 
                                                                lonaxis_range = [-130, -60]).update_layout(showlegend=False, geo=dict(bgcolor= 'rgba(0,0,0,0)'), margin = {"r":0,"t":0,"l":0,"b":0})
    return new_figure



## Splitting the act of creating the figures into two parts, such that we can use the stored data

comp_dictionary = {'dataArrStoreOne':['dropMethod1', 'dropVar', 'dropModel1', 'dropMem1', 'qsliderOne', 'previous_click-store-contour1'], 
                   'dataArrStoreTwo':['dropMethod2', 'dropVar', 'dropModel2', 'dropMem2', 'qsliderTwo', 'previous_click-store-contour2']}


for component in list(comp_dictionary.keys()): 
    method      = comp_dictionary[component][0]
    variable    = comp_dictionary[component][1]
    model       = comp_dictionary[component][2]
    member      = comp_dictionary[component][3]
    slider      = comp_dictionary[component][4]
    storedC     = comp_dictionary[component][5]
    
    @app.callback(Output(component_id = component, component_property='data'), 
                  Output(component_id = storedC, component_property = 'data'),
                  #Output(component_id = 'graphGen', component_property = 'value'), 
                  Input(component_id  = 'usmap', component_property = 'clickData'),  ## clicked on a subregion

                  Input(component_id  = method,     component_property = 'value'), 
                  Input(component_id  = model,      component_property = 'value'), 
                  Input(component_id  = member,        component_property = 'value'), 

                  Input(component_id  = 'regionSelect',  component_property = 'value'), ## changing the type of region (NCA4/States()
                  Input(component_id  = 'seasonSelect',  component_property = 'value'), 
                  Input(component_id  = variable,        component_property = 'value'), 
                  Input(component_id  = slider,          component_property = 'value'), 

                  State(component_id  = storedC,  component_property = 'data'), 
                  #Input(component_id  = 'graphGen',        component_property = 'n_clicks'),
                  prevent_initial_call = False)

    def update(click, method, model, member, region, season, variable, slider, previous_selection): 
        """ 
        updates the heatmaps
        """
        
        scenario = 'historical'
        triggered_input = ctx.triggered_id ## input that triggered the callback function

        if 'season' in variable: 
            season_ind = season
        else:
            season_ind = 'annual'

        if 'quantile' in variable: 
            qvar = 'q' + qlevs[slider]
        else:
            qvar = None
        
        #print(qvar)

        key = (variable, model, scenario, member)
        
        if key not in list(method_files[method].keys()):
            if (triggered_input == 'region_select') or (previous_selection == 'usmap') or (triggered_input is None): # storing the previous map as continental US 
                stored_value = 'usmap'
            else: # Generate button was the triggered input
                subregion  = previous_selection
                stored_value = subregion

            return {}, stored_value
        else:
            if triggered_input == 'usmap': # if user selected a region, get the regional data
                #print('region clicked')
                dataArr = getData(method_files[method][key], season_ind, region_dict[region], qvar = qvar)
                lat_name, lon_name = getLatLonNames(dataArr)
                subregion = click['points'][0]['customdata'][0] # name of the selected region
                region_ind   = np.where(dataArr.region.names.data == subregion)[0]
                dataArr = dataArr.isel(region = region_ind).squeeze().dropna(dim = lon_name, how = 'all').dropna(dim = lat_name, how = 'all') # select the region and remove unneeded nan slices 
                stored_value = subregion # storing the previous selection as the current view

            else: # using a CUS view or a stored value as the subregion 
                if (triggered_input == 'regionSelect') or (previous_selection == 'usmap') or (triggered_input is None) or (previous_selection is None): # US view
                    subregion = 'United States'
                    dataArr   = getData(method_files[method][key], season_ind, region_dict[subregion], qvar = qvar).isel(region = 0)
                    stored_value = 'usmap'
                else: # subregion view
                    dataArr = getData(method_files[method][key], season_ind, region_dict[region], qvar = qvar)
                    lat_name, lon_name = getLatLonNames(dataArr)
                    subregion    = previous_selection
                    region_ind   = np.where(dataArr.region.names.data == subregion)[0] # index of the subregion 
                    dataArr      = dataArr.isel(region = region_ind).squeeze().dropna(dim = lat_name, how = 'all').dropna(dim = lon_name, how = 'all') # select regiona and removing nan slices
                    stored_value = subregion

                    
            return dataArr.to_dict(), stored_value, #'Regenerate'


comp_dictionary = {'contourOne':['previous_click-store-contour1', 'dataArrStoreOne', 'dataArrStoreTwo', 'dropVar', 'dropModel1', 'dropMem1'], 
                   'contourTwo':['previous_click-store-contour2', 'dataArrStoreTwo', 'dataArrStoreOne', 'dropVar', 'dropModel2', 'dropMem2']}


def fullMaxMin(da1, da2):
    """
    inputs: two 2D dataArray
    returns: min/max value contained within either dataArray. Used to correct figure colorbars
    """ 

    stdev = np.nanstd(data)
    
    mn = np.nanmin([np.nanmin(da1), np.nanmin(da2)])
    mx = np.nanmax([np.nanmax(da1), np.nanmax(da2)])
    
    # if (mn*mx<0) & (diffBool): 
    #     return -3*stdev, 3*stdev
    # else:
    return mn, mx


for component in list(comp_dictionary.keys()): 
    storedC  = comp_dictionary[component][0]
    storedD  = comp_dictionary[component][1]
    sisterDS = comp_dictionary[component][2]
    variable = comp_dictionary[component][3]
    model    = comp_dictionary[component][4]
    member   = comp_dictionary[component][5]

    @app.callback(Output(component_id = component, component_property='figure'),
                  # Output(component_id = 'graphGen', component_property='value'),
                  State(component_id  = variable, component_property = 'value'),
                  State(component_id  = storedC,  component_property = 'data'), 
                  State(component_id  = storedD,  component_property = 'data'),
                  State(component_id  = sisterDS, component_property = 'data'), 
                  State(component_id  = model,    component_property = 'value'), 
                  State(component_id  = member,   component_property = 'value'), 
                  Input(component_id  = 'graphGen',component_property = 'n_clicks'),
                  prevent_initial_call = True)

    def update(variable, subregion, stored_data, sisterDS, model, member, button): 
        """ 
        updates the heatmaps
        """
        if len(list(stored_data.keys())) == 0: 
            return go.Figure(data = None, 
                    layout = go.Layout(title='', coloraxis_showscale=False)
                    ).add_annotation(x = 0.5, y = 0.5, text = 'No data', 
                                                       showarrow = False, 
                                                       font = {'size': 50}, 
                                                       xref = 'paper', 
                                                       yref = 'paper')
        if subregion == 'usmap':
            subregion = 'United States'
        
        
        dataArr = xr.DataArray.from_dict(stored_data)
        dataArr = dataArr.where(dataArr!=None).astype(float)
        
        sisterArr = xr.DataArray.from_dict(sisterDS)
        sisterArr = sisterArr.where(sisterArr!=None).astype(float)
        
        lat_name, lon_name = getLatLonNames(dataArr)
        
        
        dy, dx  = get_xrbounds(dataArr)  # scaling the longitude dimension by the cosine of the latitude
        default_width = 750              # default height. Relative to the console window height - not ideal.
        height   = (dy/dx)*default_width 

        cmap_dict = colorbar_dict  # using predefined colormaps for average value variables
        colormap = cmap_dict[variable]
        
        z_min, z_max = fullMaxMin(dataArr, sisterArr) # returns the same zmin/zmax values during each function call


        return go.Figure(data = [go.Heatmap(
                            x = dataArr[lon_name], 
                            y = dataArr[lat_name],
                            z = dataArr.data,
                            colorscale = colormap,
                            zmin = z_min, #zmin_dict[variable], 
                            zmax = z_max, #zmax_dict[variable], 
                            zauto= False, 
                            colorbar = {'orientation':'v', 'bgcolor':'white'})] + trace_dict[subregion],  #scatterMap(dataArr), 
                        layout = genLayout('{model_name} {model_member}'.format(model_name = model, model_member = member))).update_layout({'width': default_width, 'height': height})


@app.callback(Output(component_id = 'contourDiff',     component_property='figure'), 
              Input(component_id  = 'graphGen',    component_property='n_clicks'),
              # Input(component_id  = 'usmap',       component_property='clickData'), 
              State(component_id  = 'dataArrStoreOne', component_property='data'), 
              State(component_id  = 'dataArrStoreTwo', component_property='data'), 
              State(component_id  = 'diffType',        component_property='value'), 
              State(component_id  = 'dropVar',         component_property='value'), 
              State(component_id = 'dropMethod1', component_property='value'),
              State(component_id = 'dropModel1', component_property = 'value'), 
              State(component_id = 'dropMem1', component_property = 'value'),
              State(component_id = 'dropMethod2', component_property='value'),
              State(component_id = 'dropModel2', component_property = 'value'),
              State(component_id = 'dropMem2', component_property = 'value'),
              State(component_id  = 'previous_click-store-contour1', component_property = 'data'), prevent_initial_call = True)

def differenceUpdate(click, dataDictOne, dataDictTwo, diffType, variable, method1, model1, member1, method2, model2, member2, subregion): 
    if subregion == 'usmap':
        subregion = 'United States'

    d1 = xr.DataArray.from_dict(dataDictOne)
    d2 = xr.DataArray.from_dict(dataDictTwo)
    lat_name, lon_name = getLatLonNames(d1)
    
    d1[lat_name] = d2[lat_name]
    d1[lon_name] = d2[lon_name]

    dataArr = calcDifference(d1.where(d1!=None), d2.where(d2!=None), diffType)


    z_min, z_max = get_z(dataArr, True)
    dy, dx  = get_xrbounds(dataArr) # scaling the longitude dimension by the cosine of the latitude
    default_width = 750             # default height. Relative to the console window height - not ideal.
    height   = (dy/dx)*default_width
    colormap = dcolorbar_dict[variable]

    if (z_min*z_max>=0):
        if ('pr' in variable) or ('pxx' in variable): 
            cm = ml.cm.get_cmap('BrBG')
        if ('tas' in variable) or ('tnn' in variable) or ('txx' in variable):
            cm = ml.cm.get_cmap('coolwarm')
        if z_min<=0: 
            colormap = matplotlib_to_plotly_arr(cm(np.linspace(0, 0.5, 255)), 255) # scaling the colormap to only the negative values 
        else:
            colormap = matplotlib_to_plotly_arr(cm(np.linspace(0.5, 1, 255)), 255) # """ positive values
    lat_name, lon_name = getLatLonNames(dataArr)
    
    figure = go.Figure(data = [go.Heatmap(
                        x = dataArr[lon_name], 
                        y = dataArr[lat_name],
                        z = dataArr.data,
                        colorscale = colormap,
                        zmin = z_min, # zmin_dict[variable], 
                        zmax = z_max, # zmax_dict[variable], 
                        zauto= False,
                        colorbar = {'orientation':'v', 'bgcolor':'white'})] + trace_dict[subregion], 
                     layout = genLayout('Difference')
                    ).update_layout({'width': default_width, 'height': height})
                    

    return figure

    # except:
    #     return go.Figure(data = None, 
    #                 layout = go.Layout(title='', coloraxis_showscale=False)
    #                 ).add_annotation(x = 0.5, y = 0.5, text = 'No data', 
    #                                                    showarrow = False, 
    #                                                    font = {'size': 50}, 
    #                                                    xref = 'paper', 
    #                                                    yref = 'paper')


def getVal_df(df, variable, model): 
    return df[variable].iloc[np.where(df.index == model)[0]].values[0]
    
## statistics 
@app.callback(Output(component_id='rmsc_xy_out', component_property='children'),
              Output(component_id='rms_xy_out',  component_property='children'), 
              Output(component_id='mae_xy_out',  component_property='children'), 
              Output(component_id='bias_xy_out', component_property='children'), 
              Output(component_id='cor',         component_property='children'),
              Input(component_id = 'graphGen',   component_property='n_clicks'),
              State(component_id = 'dataArrStoreOne', component_property = 'data'), 
              State(component_id = 'dataArrStoreTwo', component_property = 'data'), 
              State(component_id = 'dropVar', component_property = 'value'), 
              State(component_id = 'dropMethod1', component_property='value'),
              State(component_id = 'dropModel1', component_property = 'value'), 
              State(component_id = 'dropMem1', component_property = 'value'),
              State(component_id = 'dropMethod2', component_property='value'),
              State(component_id = 'dropModel2', component_property = 'value'),
              State(component_id = 'dropMem2', component_property = 'value'),
              State(component_id = 'previous_click-store-contour1', component_property = 'data'),
              
              prevent_initial_call = True)

def show_stats(click, dm, do, variable, method1, model1, member1, method2, model2, member2, region): 
    ## use preloaded data if available

    #if (method1 == 'loca2') & (method2 == 'loca2'):
    #    if (model2 == 'PRISM'):
    #        if (member1 == 'r1i1p1f1') & (member2 == 'obs'):
    #            if region in np.unique([i[0] for i in list(df_dict.keys())]):
    #                val_rmsc = getVal_df(df_dict[(region, 'rmsc')], variable, model1)
    #                val_rms = getVal_df(df_dict[(region, 'rms')], variable, model1)
    #                val_bias = getVal_df(df_dict[(region, 'bias')], variable, model1)
    #                val_mae = getVal_df(df_dict[(region, 'mae')], variable, model1)
    #                val_corr = getVal_df(df_dict[(region, 'corr')], variable, model1)
    #                
    #                return 'RMSC: {:.4f}'.format(val_rmsc), 'RMS: {:.4f}'.format(val_rms), 'MAE: {:.4f}'.format(val_mae), 'Bias: {:.4f}'.format(val_bias), 'Correlation: {:.4f}'.format(val_corr)
    
## processing data so that it can be passed to stats functions
    
    dm = xr.DataArray.from_dict(dm).to_dataset()
    do = xr.DataArray.from_dict(do).to_dataset()
    
    dm = dm.where(dm!=None).astype(float)
    do = do.where(do!=None).astype(float)
    
    lat_name, lon_name = getLatLonNames(dm)
    dm[lat_name] = do[lat_name] ## make sure coords are cast to the same values
    dm[lon_name] = do[lon_name]
    
    var = list(do.data_vars)[0]
    
    dm = dm.where(do[var]!=np.nan)
    
    return 'RMSC: {:.4f}'.format(rmsc(dm, do)), 'RMS: {:.4f}'.format(rms(dm, do)), 'MAE: {:.4f}'.format(mae(dm, do)), 'Bias: {:.4f}'.format(bias(dm, do)), 'Correlation: {:.4f}'.format(cor_xy(dm, do))



if __name__ == '__main__':
    app.run_server(debug=True, port = 8050, use_reloader=False)
