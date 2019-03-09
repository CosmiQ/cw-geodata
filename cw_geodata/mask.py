from .utils import _check_gdf_load, _check_df_load, _check_rasterio_im_load
import numpy as np
import rasterio
from rasterio import features


def df_to_px_mask(df, geom_col='geometry', channels=['footprint'],
                  boundary_width=5, contact_spacing=10, reference_im=None,
                  affine=None, shape=(900, 900), burn_value=255):
    """Convert a dataframe of geometries to a pixel mask.

    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``
    channels : list, optional
        The mask channels to generate. There are three values that this can
        contain:

        - ``"footprint"``: Create a full footprint mask, with 0s at pixels
            that don't fall within geometries and `burn_value` at pixels that
            do.
        - ``"boundary"``: Create a mask with geometries outlined. Use
            `boundary_width` to set how thick the boundary will be drawn.
        - ``"contact"``: Create a mask with regions between >= 2 closely
            juxtaposed geometries labeled. Use `contact_spacing` to set the
            maximum spacing between polygons to be labeled.

        Each channel correspond to its own `shape` array in a third dimension of
        the output.
    boundary_width : int, optional
        Thickness (in pixels) that boundaries will be drawn if ``"boundary"``
        is passed in the `channels` `list`. Defaults to ``5``.
    contact_spacing : int, optional
        Maximum spacing (in pixels) between two geometries that will be labeled
        if ``"contact"`` is passed in the `channels` `list`. Defaults to
        ``10``.
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        `affine` transformation matrix, the image extent, etc. If provided,
        `affine` and `shape` are ignored
    affine : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    burn_value : `int` or `float`
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`.

    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`.

    """


def footprint_mask(df, out_file=None, reference_im=None, geom_col='geometry',
                   affine_obj=None, shape=(900, 900), out_type='int',
                   burn_value=255, burn_field=None):
    """Convert a dataframe of geometries to a pixel mask.

    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    out_type : 'float' or 'int'
    burn_value : `int` or `float`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`. Ignored if `burn_field` is provided
    burn_field : str, optional
        Name of a column in `df` that provides values for `burn_value` for each
        independent object. If provided, `burn_value` is ignored.

    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`.
        """
    if out_file and not reference_im:
        raise ValueError(
            'If saving output to file, `reference_im` must be provided.')
    df = _check_df_load(df)
    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
        shape = reference_im.shape
        affine_obj = reference_im.transform

    # extract geometries and pair them with burn values
    if burn_field:
        if out_type == 'int':
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('uint8')))
        else:
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('uint8')))
    else:
        feature_list = list(zip(df[geom_col], [burn_value]*len(df)))

    output_arr = features.rasterize(shapes=feature_list, out_shape=shape,
                                    transform=affine_obj)
    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        if out_type == 'int':
            meta.update(dtype='uint8')
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(output_arr, indexes=1)

    return output_arr

def border_mask(footprint_mask=None, out_file=None, reference_im=None,
                boundary_width=3, **kwargs):
    """Convert a dataframe of geometries to a pixel mask.

    Arguments
    ---------
    footprint_mask : :class:`numpy.array`, optional
        A filled in footprint mask created using :func:`footprint_mask`. If not
        provided, one will be made by calling :func:`footprint_mask` before
        creating the border mask, and the required kwargs must be provided.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored
    boundary_width : int, optional
        The width of the boundary to be created in pixels. Defaults to 3.
    boundary_type : "inner" or "outer"
        Where to draw the boundaries: within the object (``"inner"``) or
        outside of it (``"outer"``).
    **kwargs : optional
        Additional arguments to pass to :func:`footprint_mask` if one needs to
        be created.

    Returns
    -------
    border_mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and the same value as the
        footprint mask `burn_value` for the boundaries of each object.

    Note: This function draws the boundaries within the edge of the object.

    """
    if not footprint_mask:
        footprint_mask = footprint_mask(**kwargs, )
    border_mask = np.zeros_like(labels, dtype='bool')
        for l in range(1, labels.max() + 1):
            tmp_lbl = labels == l
            _k = square(3)
            tmp = erosion(tmp_lbl, _k)
            tmp = tmp ^ tmp_lbl
            border_msk = border_msk | tmp
