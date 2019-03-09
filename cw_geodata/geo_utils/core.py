import numpy as np
from affine import Affine
import geopandas as gpd
import os
import rasterio
from rasterio.enums import Resampling
import ogr
import shapely
import pyproj
from warnings import warn


class CoordTransformer(object):
    """A transformer class to change coordinate space using affine transforms.

    Notes
    -----
    This function will take in an image or geometric object (Shapely or GDAL)
    and transform its coordinate space based on `dest_obj`. `dest_obj`
    should be an instance of :class:rasterio:`rasterio.`

    Arguments
    ---------
    src_obj
        A source image or geometric object to transform. The function will
        first try to extract georegistration information from this object
        if it exists; if it doesn't, it will assume unit (pixel) coords.
    dest_obj
        Object with a destination coordinate reference system to apply to
        `src_obj`. This can be in the form of an ``[a, b, d, e, xoff, yoff]``
        `list`, an :class:`affine.Affine` instance, or a source
        :class:`geopandas.GeoDataFrame` or geotiff with `crs` metadata to
        produce the transform from, or even just a crs string.
    src_crs : optional
        Source coordinate reference in the form of a :class:`rasterio.crs.CRS`
        object or an epsg string. Only needed if the source object provided
        does not have CRS metadata attached to it.
    src_transform : :class:`affine.Affine` or `list`
        The source affine transformation matrix as a :class:`affine.Affine`
        object or in an ``[a, b, c, d, xoff, yoff]`` `list`. Required if
        `src_obj` is a :class:`numpy.array`.
    dest_transform : :class:`affine.Affine` or `list`
        The destination affine transformation matrix as a
        :class:`affine.Affine` object or in an ``[a, b, c, d, xoff, yoff]``
        `list`. Required if `dest_obj` is a :class:`numpy.array`.
    """
    def __init__(self, src_obj=None, dest_obj=None, src_crs=None,
                 src_transform=None, dest_transform=None):
        self.src_obj = src_obj
        self.src_type = None
        self.dest_obj = dest_obj
        self.dest_type = None
        self.get_obj_types()  # replaces the None values above
        self.src_crs = src_crs
        if isinstance(self.src_crs, dict):
            self.src_crs = self.src_crs['init']
        if not self.src_crs:
            self.src_crs = self._get_crs(self.src_obj, self.src_type)
        self.dest_crs = self._get_crs(self.dest_obj, self.dest_type)
        self.src_transform = src_transform
        self.dest_transform = dest_transform

    def __repr__(self):
        print('CoordTransformer for {}'.format(self.src_obj))

    def load_src_obj(self, src_obj, src_crs=None):
        """Load in a new source object for transformation."""
        self.src_obj = src_obj
        self.src_type = None  # replaced in self._get_src_crs()
        self.src_type = self._get_type(self.src_obj)
        self.src_crs = src_crs
        if self.src_crs is None:
            self.src_crs = self._get_crs(self.src_obj, self.src_type)

    def load_dest_obj(self, dest_obj):
        """Load in a new destination object for transformation."""
        self.dest_obj = dest_obj
        self.dest_type = None
        self.dest_type = self._get_type(self.dest_obj)
        self.dest_crs = self._get_crs(self.dest_obj, self.dest_type)

    def load_src_crs(self, src_crs):
        """Load in a new source coordinate reference system."""
        self.src_crs = self._get_crs(src_crs)

    def get_obj_types(self):
        if self.src_obj is not None:
            self.src_type = self._get_type(self.src_obj)
            if self.src_type is None:
                warn('The src_obj type is not compatible with this package.')
        if self.dest_obj is not None:
            self.dest_type = self._get_type(self.dest_obj)
            if self.dest_type is None:
                warn('The dest_obj type is not compatible with this package.')
            elif self.dest_type == 'shapely Geometry':
                warn('Shapely geometries cannot provide a destination CRS.')

    @staticmethod
    def _get_crs(obj, obj_type):
        """Get the destination coordinate reference system."""
        # get the affine transformation out of dest_obj
        if obj_type == "transform matrix":
            return Affine(obj)
        elif obj_type == 'Affine':
            return obj
        elif obj_type == 'GeoTIFF':
            return rasterio.open(obj).crs
        elif obj_type == 'GeoDataFrame':
            if isinstance(obj, str):  # if it's a path to a gdf
                return gpd.read_file(obj).crs
            else:  # assume it's a GeoDataFrame object
                return obj.crs
        elif obj_type == 'epsg string':
            if obj.startswith('{init'):
                return rasterio.crs.CRS.from_string(
                    obj.lstrip('{init: ').rstrip('}'))
            elif obj.lower().startswith('epsg'):
                return rasterio.crs.CRS.from_string(obj)
        elif obj_type == 'OGR Geometry':
            return get_crs_from_ogr(obj)
        elif obj_type == 'shapely Geometry':
            raise TypeError('Cannot extract a coordinate system from a ' +
                            'shapely.Geometry')
        else:
            raise TypeError('Cannot extract CRS from this object type.')

    @staticmethod
    def _get_type(obj):
        if isinstance(obj, gpd.GeoDataFrame):
            return 'GeoDataFrame'
        elif isinstance(obj, str):
            if os.path.isfile(obj):
                if os.path.splitext(obj)[1].lower() in ['tif', 'tiff',
                                                        'geotiff']:
                    return 'GeoTIFF'
                elif os.path.splitext(obj)[1] in ['csv', 'geojson']:
                    # assume it can be loaded as a geodataframe
                    return 'GeoDataFrame'
            else:  # assume it's a crs string
                if obj.startswith('{init'):
                    return "epsg string"
                elif obj.lower().startswith('epsg'):
                    return "epsg string"
                else:
                    raise ValueError('{} is not an accepted crs type.'.format(
                        obj))
        elif isinstance(obj, ogr.Geometry):
            # ugh. Try to get the EPSG code out.
            return 'OGR Geometry'
        elif isinstance(obj, shapely.Geometry):
            return "shapely Geometry"
        elif isinstance(obj, list):
            return "transform matrix"
        elif isinstance(obj, Affine):
            return "Affine transform"
        elif isinstance(obj, np.array):
            return "numpy array"
        else:
            return None

    def transform(self, output_loc):
        """Transform `src_obj` from `src_crs` to `dest_crs`.

        Arguments
        ---------
        output_loc : `str` or `var`
            Object or location to output transformed src_obj to. If it's a
            string, it's assumed to be a path.
        """
        if not self.src_crs or not self.dest_crs:
            raise AttributeError('The source or destination CRS is missing.')
        if not self.src_obj:
            raise AttributeError('The source object to transform is missing.')
        if isinstance(output_loc, str):
            out_file = True
        if self.src_type == 'GeoTIFF':
            return rasterio.warp.reproject(rasterio.open(self.src_obj),
                                           output_loc,
                                           src_transform=self.src_transform,
                                           src_crs=self.src_crs,
                                           dst_trasnform=self.dest_transform,
                                           dst_crs=self.dest_crs,
                                           resampling=Resampling.bilinear)
        elif self.src_type == 'GeoDataFrame':
            if isinstance(self.src_obj, str):
                # load the gdf and transform it
                tmp_src = gpd.read_file(self.src_obj).to_crs(self.dest_crs)
            else:
                # just transform it
                tmp_src = self.src_obj.to_crs(self.dest_crs)
            if out_file:
                # save to file
                if output_loc.lower().endswith('json'):
                    tmp_src.to_file(output_loc, driver="GeoJSON")
                else:
                    tmp_src.to_file(output_loc)  # ESRI shapefile
                return
            else:
                # assign to the variable and return
                output_loc = tmp_src
                return output_loc
        elif self.src_type == 'OGR Geometry':
            dest_sr = ogr.SpatialReference().ImportFromEPSG(
                int(self.dest_crs.lstrip('epsg')))
            output_loc = self.src_obj.TransformTo(dest_sr)
            return output_loc
        elif self.src_type == 'shapely Geometry':
            if self.dest_type not in [
                    'Affine transform', 'transform matrix'
                    ] and not self.dest_transform:
                raise ValueError('Transforming shapely objects requires ' +
                                 'an affine transformation matrix.')
            elif self.dest_type == 'Affine transform':
                output_loc = shapely.affinity.affine_transform(
                    self.src_obj, [self.dest_obj.a, self.dest_obj.b,
                                   self.dest_obj.d, self.dest_obj.e,
                                   self.dest_obj.xoff, self.dest_obj.yoff]
                )
                return output_loc
            elif self.dest_type == 'transform matrix':
                output_loc = shapely.affinity.affine_transform(self.src_obj,
                                                               self.dest_obj)
                return output_loc
            else:
                if isinstance(self.dest_transform, Affine):
                    xform_mat = [self.dest_transform.a, self.dest_transform.b,
                                 self.dest_transform.d, self.dest_transform.e,
                                 self.dest_transform.xoff,
                                 self.dest_transform.yoff]
                else:
                    xform_mat = self.dest_transform
                output_loc = shapely.affinity.affine_transform(self.src_obj,
                                                               xform_mat)
                return output_loc
        elif self.src_type == 'numpy array':
            return rasterio.warp.reproject(
                self.src_obj, output_loc, src_transform=self.src_transform,
                src_crs=self.src_crs, dst_transform=self.dest_transform,
                dst_crs=self.dest_crs)


def get_crs_from_ogr(annoying_OGR_geometry):
    """Get a CRS from an :class:`osgeo.ogr.Geometry` object.

    Arguments
    ---------
    annoying_OGR_geometry: :class:`osgeo.ogr.Geometry`
        An OGR object which stores crs information in an annoying fashion.

    Returns
    -------
    An extremely clear, easy to work with ``'epsg[number]'`` string.
    """
    srs = annoying_OGR_geometry.GetSpatialReference()
    result_of_ID = srs.AutoIdentifyEPSG()  # if success, returns 0
    if result_of_ID == 0:
        return 'epsg:' + str(srs.GetAuthorityCode(None))
    else:
        raise ValueError('Could not determine EPSG code.')


def list_to_affine(xform_mat):
    """Create an Affine from a list or array-formatted [a, b, d, e, xoff, yoff]

    Arguments
    ---------
    xform_mat : `list` or :class:`numpy.array`
        A `list` of values to convert to an affine object.

    Returns
    -------
    aff : :class:`affine.Affine`
        An affine transformation object.
    """
    # first make sure it's not in gdal order
    if rasterio.transform.tastes_like_gdal(xform_mat):
        return Affine.from_gdal(*xform_mat)
    else:
        return Affine(*xform_mat)
