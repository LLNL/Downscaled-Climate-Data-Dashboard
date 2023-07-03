## stats files for dashboard application ##
def getLatLonNames(d): 
    lat_name = [i for i in list(d.coords) if 'lat'  in i][0]
    lon_name = [i for i in list(d.coords) if 'lon' in i][0]
    
    return lat_name, lon_name

def rmsc(modelDataArr, obsDataArr): ## needs to be a dataset
    import numpy as np
    modelVar = list(modelDataArr.data_vars)[0]
    obsVar   = list(obsDataArr.data_vars)[0]
    
    modelDataArr = modelDataArr.bounds.add_missing_bounds()
    obsDataArr   = obsDataArr.bounds.add_missing_bounds()
    
    weights = modelDataArr.spatial.get_weights(axis=['X', 'Y'])
    
    # compute anomalies
    
    lat_name, lon_name = getLatLonNames(modelDataArr)
    
    modelAnom = modelDataArr[modelVar] - modelDataArr[modelVar].weighted(weights.fillna(0)).mean([lon_name, lat_name], skipna = True)
    obsAnom   = obsDataArr[obsVar]   - obsDataArr[obsVar].weighted(weights.fillna(0)).mean([lon_name, lat_name], skipna = True)
    
    square_diff = (modelAnom - obsAnom)**2
    
    return float(np.sqrt(square_diff.weighted(weights.fillna(0)).mean(([lon_name, lat_name]), skipna = True)).data)
    

def rms(modelDataArr, obsDataArr): 
    import numpy as np
    
    modelVar = list(modelDataArr.data_vars)[0]
    obsVar   = list(obsDataArr.data_vars)[0]
    
    modelDataArr = modelDataArr.bounds.add_missing_bounds()
    obsDataArr   = obsDataArr.bounds.add_missing_bounds()
    
    lat_name, lon_name = getLatLonNames(modelDataArr)
    
    
    weights = modelDataArr.spatial.get_weights(axis=['X', 'Y'])
    
    diff_square = (modelDataArr[modelVar] - obsDataArr[obsVar])**2
    
    return float(np.sqrt(diff_square.weighted(weights.fillna(0)).mean([lon_name, lat_name], skipna = True)).data)

def mae(modelDataArr, obsDataArr): 
    import numpy as np
    
    modelVar = list(modelDataArr.data_vars)[0]
    obsVar   = list(obsDataArr.data_vars)[0]
    
    modelDataArr = modelDataArr.bounds.add_missing_bounds()
    obsDataArr   = obsDataArr.bounds.add_missing_bounds()
    
    lat_name, lon_name = getLatLonNames(modelDataArr)
    
    weights = modelDataArr.spatial.get_weights(axis=['X', 'Y'])
    
    diff = abs(modelDataArr[modelVar] - obsDataArr[obsVar])
    
    return float(diff.weighted(weights.fillna(0)).mean([lon_name, lat_name], skipna = True).data)

def bias(modelDataArr, obsDataArr): 
    import numpy as np
    
    modelVar = list(modelDataArr.data_vars)[0]
    obsVar   = list(obsDataArr.data_vars)[0]
    
    modelDataArr = modelDataArr.bounds.add_missing_bounds()
    obsDataArr   = obsDataArr.bounds.add_missing_bounds()
    
    lat_name, lon_name = getLatLonNames(modelDataArr)
    
    weights = modelDataArr.spatial.get_weights(axis=['X', 'Y'])
    
    diff = modelDataArr[modelVar] - obsDataArr[obsVar] 
    
    return float(diff.weighted(weights.fillna(0)).mean([lon_name, lat_name], skipna = True).data)

def std(d): 
    import numpy as np
    
    var = list(d.data_vars)[0]
    
    
    d = d.bounds.add_missing_bounds()
    
    lat_name, lon_name = getLatLonNames(d)
    
    
    weights = d.spatial.get_weights(axis=['X', 'Y'])
    
    avg = float(d[var].weighted(weights.fillna(0)).mean([lon_name, lat_name], skipna = True))
    anom = (d[var] - avg)**2
    variance = float(anom.weighted(weights.fillna(0)).mean([lon_name, lat_name], skipna = True))
    return np.sqrt(variance)

def cor_xy(modelDataArr, obsDataArr): 
    import numpy as np
    
    modelVar = list(modelDataArr.data_vars)[0]
    obsVar   = list(obsDataArr.data_vars)[0]
    
    modelDataArr = modelDataArr.bounds.add_missing_bounds()
    obsDataArr   = obsDataArr.bounds.add_missing_bounds()
    
    lat_name, lon_name = getLatLonNames(modelDataArr)
    
    
    #print(modelDataArr[modelVar])
    #print(obsDataArr[obsVar])
    
    weights = modelDataArr.spatial.get_weights(axis=['X', 'Y'])
    
    # m_avg = modelDataArr.spatial.average(modelVar, axis = ['X', 'Y'], weights = weights)[modelVar].values
    # o_avg = obsDataArr.spatial.average(obsVar, axis = ['X', 'Y'], weights = weights)[obsVar].values
    
    m_avg   = float(modelDataArr[modelVar].weighted(weights.fillna(0)).mean([lon_name, lat_name], skipna = True))
    o_avg   = float(obsDataArr[obsVar].weighted(weights.fillna(0)).mean([lon_name, lat_name], skipna = True ))

    
    covar = ((modelDataArr[modelVar] - m_avg)*(obsDataArr[obsVar] - o_avg)).weighted(weights.fillna(0)).mean(dim = [lon_name, lat_name], skipna = True).values
    
    #print(covar)
    mstd = std(modelDataArr)
    #print(mstd)
    ostd = std(obsDataArr)
    #print(ostd)
    
    return float(covar/(mstd*ostd))