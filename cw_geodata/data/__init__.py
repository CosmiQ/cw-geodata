import os
import rasterio
import gdal
import geopandas as gpd

data_dir = os.path.abspath(os.path.dirname(__file__))


def sample_load_rasterio():
    return rasterio.open(os.path.join(data_dir, 'sample_geotiff.tif'))


def sample_load_gdal():
    return gdal.Open(os.path.join(data_dir, 'sample_geotiff.tif'))


def sample_load_geojson():
    return gpd.read_file(os.path.join(data_dir, 'sample.geojson'))


def sample_load_csv():
    return pd.read_file(os.path.join(data_dir, 'sample.csv'))
