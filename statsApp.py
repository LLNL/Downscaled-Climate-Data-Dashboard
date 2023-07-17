#!/usr/bin/env python
# coding: utf-8

## App page that shows PCMDI-style metric plots
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
import time
xr.set_options(keep_attrs=True)


import sys
sys.path.insert(0, './func/')

from stats import rmsc, rms, mae, bias, cor_xy, std


# NCA4 region dictionary with STUPS codes

nca4_regions = {'Northern Great Plains': ['MT', 'NE', 'SD', 'ND', 'WY'], \
            'Southern Great Plains': ['TX', 'OK', 'KS'], \
            'Southwest': ['CA', 'NV', 'AZ', 'NM', 'CO', 'UT'], \
            'Southeast': ['FL', 'GA', 'SC', 'MS', 'AL', 'AR', 'TN', 'KY', 'NC', 'VA', 'LA'], \
            'Northeast': ['WV', 'DC', 'PA', 'MA', 'NY', 'CT', 'RI', 'ME', 'NJ', 'VT', 'NH', 'MD', 'DE'], \
            'Northwest': ['WA', 'OR', 'ID'], \
            'Midwest':   ['IL', 'MI', 'WI', 'MO', 'IA', 'IN', 'OH', 'MN']}
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
us_mask = regionmask.from_geopandas(us, names = "NAME", name = "NCA4 regions")
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

def greater_less_correction(da, fname): ## fillValue incorrectly input as 1.0 in file creation
    """ 
    removes spurious 1.0 fillValue in threshold files
    """
    
    if ('_ge_' in fname) or ('_le_' in fname):
        return da.where(da!=1.0)
    else:
        return da
    
def temp_conversion(da, fname): 
    """ 
    convert to Kelvin to Celcius
    """ 
    
    if ('txx' in fname) or ('tasmax' in fname) or ('tasmean' in fname) or ('tnn' in fname) or ('tasmin' in fname):
        if ('_ge_' not in fname) & ('_le_' not in fname) & ('PRISM' not in fname):
            da -= 273.15
            return da
        else:
            return da 
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
    return temp_conversion2(precip_conversion(da, fname),fname)


def getLatLonNames(d): 
    try:
        lat_name = [i for i in list(d.coords) if 'lat'  in i][0]
        lon_name = [i for i in list(d.coords) if 'lon' in i][0]
    except:
        lat_name = [i for i in list(d.dims) if 'lat'  in i][0]
        lon_name = [i for i in list(d.dims) if 'lon' in i][0]
    return lat_name, lon_name

def getData(fname, seas, mask, qvar = None):
    import time
    
    if 'quantile' in fname: 
        decode_times = False
    else:
        decode_times = True
    
    data = xcdat.open_dataset(fname, decode_times = decode_times)
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
    
# def roundCoords(dataArr): # bad way of handling
#     dataArr['lat'] = np.round(dataArr['lat'], 6)
#     dataArr['lon'] = np.round(dataArr['lon'], 6)
    
#     return dataArr

metric_dict = {'rms': rms, 'rmsc': rmsc, 'mae': mae, 'bias': bias, 'corr':cor_xy}

import pickle as pkl
with open('./metricFiles/modelPrismData.pkl', 'rb') as f:
    df_dict = pkl.load(f)


import pickle as pkl 
with open('./metricFiles/modelPrismStarData.pkl', 'rb') as f:
    df_dictStar = pkl.load(f)


data = {'LOCA2': df_dict, 'STAR': df_dictStar}


ndata = {}
median_data = {}

for comparison_ds in ['LOCA2', 'STAR']:#, 'STAR']:
    print(comparison_ds)
    ndf_dict = {}
    param_median_dict = {}
    for region in list(nca4_regions.keys()):
        for metric in list(metric_dict.keys()):
            try:
                key = (region, metric)


                ndf  = data[comparison_ds][key].copy(deep = True)
                median_dict = {}

                for column in ndf.columns:
                    column_data = np.array([k for k in ndf[column] if k != np.nan])

                    median = np.nanmedian(column_data)

                    #calculate interquartile range 
                    q3, q1 = np.nanpercentile(column_data, [75 ,25])
                    iqr = q3 - q1
                    ndf[column] = np.round([(k-median)/iqr for k in ndf[column]], 3)
                    median_dict[column] = median
                param_median_dict[key] = median_dict
                ndf_dict[key] = ndf
            except:
                pass
        ndata[comparison_ds] = ndf_dict
        median_data[comparison_ds] = param_median_dict

    


style_dict = {}

for comparison_ds in ['LOCA2', 'STAR']:
    all_styles = {}
    for region in list(nca4_regions.keys()):
        for metric in list(metric_dict.keys()):
            try:
                key = (region, metric)

                df = ndata[comparison_ds][key].reset_index()
                style_data_conditional = []

                for variable in df.columns[1:]:
                    vals = df[variable].values
                    if all([np.isnan(i) for i in vals]): ## nan columns
                        style_data_conditional.append({'if': {'column_id':variable}, 'backgroundColor':'lightgray'})
                        continue
                    nmin = np.nanmin(vals)
                    nmax = np.nanmax(vals)

                    mx, mn = np.max(np.abs([nmin, nmax])), -np.max(np.abs([nmin, nmax]))

                    arr = np.linspace(mn, mx,  8)

                    for bound_index, color in zip(range(0, len(arr)), ['mediumblue', 'cornflowerblue', 'skyblue', 'white', 'mistyrose', 'lightcoral', 'crimson']): 
                        style_data_conditional.append({
                            'if': {
                                'filter_query':('{{{variable}}}>={min_bound} && {{{variable}}}<={max_bound}').format(variable = variable, min_bound = arr[bound_index], max_bound = arr[bound_index + 1]), 
                                'column_id': variable
                        }, 
                            'backgroundColor': color, 
                            'fontWeight':'bold'                     
                        })

                all_styles[key] = style_data_conditional
            except:
                pass
    style_dict[comparison_ds] = all_styles


import dash
from dash import dcc
from dash import html
from dash import dash_table


# In[253]:


def getImageURL(region, model, variable, comparison_ds):
    addStr = ''
    if comparison_ds == 'STAR':
        addStr = 'STAR'
    return ('_').join([variable, model, region.replace(' ', ''), 'difference']) + addStr + '.png'


app = JupyterDash()


tooltips = {}

for comparison_ds in ['LOCA2', 'STAR']:
    tooltip_dict = {}
    tdf = data[comparison_ds][('Northern Great Plains', 'rms')]
    for region in list(nca4_regions.keys()): 
        try:
            tooltip_data=[
                {column: {'value': f'![](assets/images/{getImageURL(region, model_name, column, comparison_ds)})', 'type': 'markdown'}
                    for column, value in row.items()
                } for row, model_name in zip(data[comparison_ds][(region, 'rms')].to_dict('records'), tdf.index)
            ]
            tooltip_dict[region] = tooltip_data
        except:
            pass
    tooltips[comparison_ds] = tooltip_dict

tableDict = {}

for comparison_ds in ['LOCA2', 'STAR']:
    for region in list(nca4_regions.keys()):
        #print(region)
        for metric in list(metric_dict.keys()):
            new_key = ('').join([comparison_ds, region, metric])
            key = (region, metric)
            try:
                tableDict[new_key] = dash_table.DataTable(data[comparison_ds][key].reset_index().round(3).to_dict('records'), [{"name": i, "id": i} for i in data[comparison_ds][key].reset_index().columns],              
                                                      tooltip_data = tooltips[comparison_ds][region], sort_action = 'native',         
                                                      tooltip_delay=0,
                                                      tooltip_duration=None)
            except:
                pass
        
                                              #         tooltip_data = [
                                              #     {
                                              #         'annual_pxx':{
                                              #             'value': 'Location at Bakersfield\n\n![Bakersfield](assets/images/annual_pxx_ACCESS-ESM1-5_NorthernGreatPlains_difference.png)', #(https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Big_Bear_Valley%2C_California.jpg/1200px-Big_Bear_Valley%2C_California.jpg)', 
                                              #             'type':'markdown', 
                                              #         }
                                              #     }
                                              # ],


style_dict['LOCA2'][('Northern Great Plains', 'rms')] == style_dict['STAR'][('Northern Great Plains', 'rms')]
ntableDict = {}

for comparison_ds in ['LOCA2', 'STAR']:
    print(comparison_ds)
    for region in list(nca4_regions.keys()):
        try:
            for metric in list(metric_dict.keys()):
                new_key = ('').join([comparison_ds, region, metric])
                key = (region, metric)
                ntableDict[new_key] = dash_table.DataTable(data = ndata[comparison_ds][key].reset_index().to_dict('records'), 
                                                           columns = [{"name": i, "id": i} for i in ndata[comparison_ds][key].reset_index().columns], 
                                                           style_data_conditional = style_dict[comparison_ds][key], sort_action = 'native', 
                                                           tooltip_data = tooltips[comparison_ds][region], 
                                                           tooltip_delay = 0, 
                                                           tooltip_duration = None)
        except:
            pass

metric_names = {'RMS':'rms', 'Centered RMS': 'rmsc', 'Mean Absolute Error': 'mae', 'Bias':'bias', 'Correlation':'corr'}

app.layout = html.Div([
    html.H1(list(nca4_regions.keys())[0], id = 'title', style = {'textAlign':'center'}),
    html.Div(children = [
        dcc.Dropdown(list(nca4_regions.keys()), list(nca4_regions.keys())[0], multi = False,  id = 'dropRegion', style = {'width':250, 'fontSize':15}),  
        dcc.Dropdown(list(metric_names.keys()), list(metric_names.keys())[0], multi = False,  id = 'dropMetric', style = {'width':250, 'fontSize':15}),
        dcc.Dropdown(list(['LOCA2', 'STAR']), 'LOCA2', multi = False, id = 'dropComparison', style = {'width':250, 'fontSize':15}),  
        dcc.RadioItems(['Raw', 'Normalized'], 'Raw', id = 'dataType')
    ], style = {'display':'flex', 'flexDirection':'row'}), 

    html.Div(id = 'metricTable', children = [
        tableDict[('').join(['LOCA2', list(nca4_regions.keys())[0], metric_names[list(metric_names.keys())[0]]])]
    ], style = {'width':200}), #), 
    
    html.Div([
        dcc.Store(data = tableDict, id = 'raw'),
        dcc.Store(data = ntableDict,id = 'normalized'),
        dcc.Store(data = metric_names, id = 'metricNames')
    ]), 
], style = {'marginLeft':200})
        
                      


from dash import clientside_callback, Input, Output, State

app.clientside_callback(
    """
    function(region, metricLongName, dtype, comparison_ds, tableDict, ntableDict, metric_names) {
        if (dtype == 'Raw') { 
            return tableDict[comparison_ds + region + metric_names[metricLongName]]
        } else { 
            return ntableDict[comparison_ds + region + metric_names[metricLongName]]
        }
        
    }
    """,
        Output(component_id= 'metricTable', component_property='children'), 
        Input(component_id = 'dropRegion', component_property='value'), 
        Input(component_id = 'dropMetric',  component_property='value'), 
        Input(component_id = 'dataType',   component_property='value'),
        Input(component_id = 'dropComparison', component_property = 'value'),
        State(component_id = 'raw', component_property = 'data'), 
        State(component_id = 'normalized', component_property = 'data'), 
        State(component_id = 'metricNames', component_property = 'data')
)

app.clientside_callback(
    """
    function(region) { 
        return region
    }
    """,
    Output(component_id='title', component_property='children'), 
    Input(component_id ='dropRegion', component_property = 'value'))
    


if __name__ == '__main__':
    app.run_server(debug=True, port = 8050, use_reloader=False)




