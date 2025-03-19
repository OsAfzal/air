import xarray as xr
import glob
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point


def load_nc(file_path, chunks=None, variables=None):
     
    """
    Open multiple NetCDF files from a specified directory and return an 
    xarray.Dataset object. The function can optionally chunk the data for 
    efficient processing and select specific variables to include in the dataset.

    Parameters:
    -----------
    file_path : str
        The directory path where the NetCDF files are located. The function 
        will recursively search for files with the '.nc' extension.
        
    chunks : dict, optional
        A dictionary specifying chunk sizes for chunking the dataset. The 
        chunking is done when reading the files, which is useful for working 
        with large datasets that don't fit into memory. The dictionary keys 
        are variable names, and values are the chunk sizes (e.g., {'time': 10}).
        Default is None, meaning no chunking is applied.
        
    variables : list of str, optional
        A list of variable names to select from the dataset. If provided, only 
        these variables will be included in the returned dataset. Default is None, 
        meaning all variables will be included.

    Returns:
    --------
    xarray.Dataset
        The combined xarray.Dataset containing data from all NetCDF files in the 
        specified directory, with optional chunking and variable selection applied.

    Example:
    --------
    ds = merra.load_nc('/path/to/data/', chunks={'time': 10}, variables=['temperature', 'humidity'])
    """

    # Getting the list of files
    files = glob.glob(f'{file_path}**/*.nc', recursive=True)
     
    # Open the dataset
    ds = xr.open_mfdataset(files, combine='by_coords', chunks=chunks)
    
    # Select specific variables
    if variables:
        ds = ds[variables]
    
    return ds



def mean_polygon(ds: xr.Dataset, variable: str, shapefile: gpd.GeoDataFrame):
    """
    Calculate the mean of a variable within a polygon shape, averaging over lat-lon dimensions.
    
    Args:
        ds (xr.Dataset): xarray dataset containing the data with dimensions ('lat', 'lon', 'time').
        variable (str): Name of the variable in the dataset to compute the mean for.
        shapefile (gpd.GeoDataFrame): GeoDataFrame containing the polygon shape(s) for masking.
    
    Returns:
        xr.DataArray: Mean values of the variable, averaged over lat-lon for each time step.
    """
    
    # Extract the polygon geometries
    polygon = shapefile.geometry.union_all()  # Combine all polygons if there are multiple
    
    # Create a mask for the dataset based on the polygon
    lat, lon = ds['lat'], ds['lon']
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    
    # Check if each point in the grid is inside the polygon
    mask = np.array([polygon.contains(Point(p)) for p in points])
    mask = mask.reshape(lon_grid.shape)
    
    # Apply the mask to the dataset for the variable
    var_data = ds[variable].where(mask)
    
    # Compute the mean over lat-lon dimensions, keeping time intact
    mean_data = var_data.mean(dim=['lat', 'lon'])
    
    return mean_data



def grid_data(ds, lat_target, lon_target):
    """
    This function retrieves the data at the nearest latitude and longitude to the specified target coordinates
    in a given xarray Dataset.
    
    Parameters:
    ds (xarray.Dataset): The dataset containing 'lat' and 'lon' dimensions.
    lat_target (float): The target latitude.
    lon_target (float): The target longitude.
    
    Returns:
    xarray.DataArray: Data at the nearest latitude and longitude.
    """
    
    lat_idx = ds['lat'].sel(lat=lat_target, method='nearest')
    lon_idx = ds['lon'].sel(lon=lon_target, method='nearest')

    data_at_point = ds.sel(lat=lat_idx, lon=lon_idx, method='nearest')
    
    return data_at_point

