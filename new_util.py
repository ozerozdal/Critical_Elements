import glob
from pathlib import Path
from datetime import datetime
import math
import rasterio
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import warnings
from osgeo import gdal, gdalconst
from geocube.api.core import make_geocube
import unicodedata
import re
import pathlib
import osgeo_utils.gdal_merge as gm
import xarray
import rioxarray as rio
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from osgeo import gdal, ogr
import sys
from osgeo import osr
from collections.abc import Container
from sklearn.linear_model import LogisticRegression
def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    Parameters
    ----------
    value : str
        string to be converted
    allow_unicode : bool, default=False
        Convert to ASCII if 'allow_unicode' is False
    Returns
    -------
    str
        Converted string into a file or url friendly format.
    """
    #reference : https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')
def geotiff_to_csv(input_geotiff, output_csv):
    """
    Converts a geotiff (.tif) file to a csv containing X, Y (coordinates) and Z the value of the first band.
    Only handles single-band for now.
    This version may be a little bit slow. It uses GDAL translate and the output needs to be rewritten by pandas to remove nan.
    There is an alternative with xarray which is 10 time as fast but it is very complex to implement and we would need adequate testing.
    Parameters
    ----------
    input_geotiff : str
        path of the input geotiff
    output_csv : str
        path of the resulting csv
    dropna : bool, default=False
        Whetever to drop or not the null values. Defaut is False. NOT IMPLEMENTED YET.
    verbose : bool, default=False
        Output the processing in the console.
    Returns
    -------
    None
    """
    # http://drr.ikcest.org/tutorial/k8022
    inDs = gdal.Open(input_geotiff)
    if (pathlib.Path(input_geotiff).exists() != True):
        print("The input geotiff " + str(input_geotiff) + " do not exists please double check")
        return
    band = inDs.GetRasterBand(1)
    nodata_val = band.GetNoDataValue()
    if (pathlib.Path(output_csv).exists()):
        os.remove(output_csv)
    outDs = gdal.Translate(output_csv, inDs, format='XYZ', creationOptions=["ADD_HEADER_LINE=YES"])
    outDs = None
    df = pd.read_csv(output_csv, sep=' ')
    if (nodata_val != None):
        df = df.replace(nodata_val, np.nan)
    df = df.dropna(how='any')
    df.to_csv(output_csv, sep=',', index=False)
    print("Conversion from Geotiff to CSV successful. See " + output_csv)
    return
def merge_geotiff(geotiffs_list, output_file, xRes, yRes, inputNoData=None, dstNoData=-9999, outputType='integer'):
    """
    Merges geotiffs (or other GDAL supported formats) together. It does not perform resampling or reprojection ! If layers are overlapping, the first layer in the list will predominate.
    Parameters
    ----------
    geotiffs_list : list
        list of input geotiffs files to merge
    output_file : str
        path to the file.tif of the resulting merge geotiff
    xRes : int
        target x resolution
    yRes : int
        target y resolution
    inputNoData : int, default=None
        nodata in input files meta. if nan write 'nan'. Leave empty to merge targets or categorical layers together.
    dstNoData : int, default=-9999
        nodata in output files meta
    outputType : str, default='integer'
        type for the output. If you are merging targets (0-1) layers or categorical layers, use 'integer'. Else use 'float'
    Returns
    -------
    None
    """
    if (outputType == 'integer'):
        #outputType = gdal.GDT_Int16
        outputType = 'Int16'
        #algo = 'nearest' no more resampling
    elif (outputType == 'float'):
        #outputType = gdal.GDT_Float32
        outputType = 'Float32'
        #algo = 'cubic' no more resampling
    #previous version was using VRT to pack layers together. However if layers were overlapping, the resulting output was only the first layer in the input list. Otherwise, if layers were not overlapped, result was merged fine.
    #vrt = gdal.BuildVRT(destName="", srcDSOrSrcDSTab=geotiffs_list, srcNoData=inputNoData, VRTNodata=dstNoData, xRes=xRes, yRes=yRes, outputSRS=destinationCRS, resampleAlg=algo)
    # https://gdal.org/python/osgeo.gdal-module.html#TranslateOptions
    #gdal.Translate(output_file, vrt, format='GTiff', noData=dstNoData)
    #vrt = None
    arguments = ['', '-init', '-9999', '-o', output_file, '-a_nodata', str(dstNoData),
     '-ot', outputType, '-ps', str(xRes), str(yRes)]
    if(inputNoData != None):
        arguments.append('-n')
        arguments.append(str(inputNoData))
    for item in geotiffs_list:
        arguments.insert(5, str(item))
    gm.main(arguments)
    return
def csv_to_raster(input_csv, output_directory, columns, categories, x_field, y_field, dstCRS):
    """
    Convert a .csv file to geotiff format. Can handle one or multiple columns.
    Resolution is handled behind the scene.
    Parameters
    ----------
    input_csv : str
        path to the input csv
    output_directory : str
        output directory where to save each geotiffs. Their names will be the columns names.
    columns : str or list
        name of the column to convert, or multiple names (in a list)
    categories : list
        name or names of the field which are categoric. If there ain't no category in the csv, write None or blank list.
    x_field : str
        name of the x_field
    y_field : str
        name of the y_field
    dstCRS : str
        coordinate system, example 'epsg:26921'
    Returns
    -------
    None
    Examples
    -------
        Convert one numerical column and one categorical column from a csv file.
        >>> csv_to_raster(input_csv = 'my_cube.csv', output_directory='my_output_folder/', columns = ['a_numeric', 'b_category'], categories = ['b_category'], x_field='x', y_field='y', dstCRS ='epsg:26921')
    """
    warnings.warn("If your data has spatial discontinuity. Use df_to_gtiffs() function built upon core rasterio")
    start_time = time.time()
    print("This function accepts dataframes from memory as well as from disk. In-memory dataframes will be very fast obviously.")
    if (isinstance(input_csv,pd.DataFrame)):
        df = input_csv
    else:
        df = pd.read_csv(input_csv)
    if (isinstance(columns, str)):
        columns = [columns]
    for c in columns:
        if c not in df.columns:
            print("Column \"" + c + "\" not found in header. Fix it. Processing stopped.")
            return
    if x_field not in df.columns or y_field not in df.columns:
        print("Columns \"" + x_field + "\" and/or \"" + y_field + "\ not in header. Fix it. Processing stopped.")
        return
    if (categories == None):
        categories = []
    elif (isinstance(categories, str)):
        categories = [categories]
    elif (isinstance(categories, list) != True):
        print('Categories should be either None, str or list. Please fix this. Processing stopped.')
        return
    df.rename(columns={x_field: 'x', y_field: 'y'}, inplace = True)
    df.set_index(['x', 'y'], inplace=True)
    xr = xarray.Dataset.from_dataframe(df)
    xr = xr.rio.set_spatial_dims('x', 'y')
    xr = xr.rio.write_crs(dstCRS)
    xr = xr.transpose('y', 'x')
    if (os.path.exists(output_directory) != True):
        os.makedirs(output_directory)
    print("Rasterizing...")
    if (isinstance(columns, list)):
        for col in columns:
            print("Column : " + col)
            output_path = os.path.join(output_directory, col + ".tif")
            if col in categories:
                xr[col] = xr[col].fillna(-9999)
                xr[col] = xr[col].astype(dtype='int16')
                xr[col] = xr[col].where(xr[col] != -9999)
            xr[col].rio.to_raster(raster_path=output_path, nodata=-9999)
            print("Done.  See file " + output_path)
    else:
        print("columns argument must be a string or a list.")
    end_time = time.time()
    print("Completed. Time elapsed (s) : ")
    print(end_time - start_time)
    return
def warp(input_geotiff, output_geotiff, dstSRS, xRes, yRes, format='GTiff', targetAlignedPixels=True, dstNodata=-9999,
         resampleAlg='near', **kwargs):
    """
    Encapsulate gdal.warp command and allows the user to reproject and resample a geotiff file.
    Accepts a dictionary (**kwargs) with all gdal parameters, see here https://gdal.org/python/osgeo.gdal-module.html#WarpOptions
    Parameters
    ----------
    input_geotiff : str
        input geotiff path
    output_geotiff : str
        output geotiff path
    dstSRS : str
        destination CRS, example 'epsg:26711'
    xRes : int
        x resolution
    yRes : int
        y resolution
    format : str, default='GTiff'
        output format
    targetAlignedPixels : bool, default=True
        whetever to align pixels to CRS or not
    dstNodata : int, default = -9999
        nodata value for output
    resampleAlg : str, default='near'
        resampling algorithm, default = 'near' for nearest neighbors
    kwargs : dict
        keywords arguments, see here https://gdal.org/python/osgeo.gdal-module.html#WarpOptions
    Returns
    -------
    None
    Examples
    --------
        Reproject a geological map
            >>> from GeoDS import utilities
            >>> kwargs = {'targetAlignedPixels':False, 'dstNodata':-1000}
            >>> utilities.warp('input_data/categorical/usgs_geology_reprojected.tif', 'test_geo.tif', dstSRS='epsg:26711', xRes=100, yRes=100, **kwargs)
    """
    # https://gdal.org/python/osgeo.gdal-module.html#WarpOptions
    # https://gis.stackexchange.com/questions/278627/using-gdal-warp-and-gdal-warpoptions-of-gdal-python-api/341693
    # warp_options = gdal.WarpOptions(**kwargs)
    # there are mandatory arguments that may lead to crashes if not provided, by example if dstSRS and targetAlignedPixels are not joined with xRes and yRes, it will crash.
    # Anyway
    if not os.path.isfile(input_geotiff):
        raise ValueError("The input file you provided %s is inexistant. Check your spelling." % input_geotiff)
    mandatory = {'format': format, 'dstSRS': dstSRS, 'xRes': xRes, 'yRes': yRes,
                 'targetAlignedPixels': targetAlignedPixels, 'dstNodata': dstNodata, 'resampleAlg': resampleAlg}
    kwargs.update(mandatory)
    ds = gdal.Warp(srcDSOrSrcDSTab=input_geotiff, destNameOrDestDS=output_geotiff, **kwargs)
    if (ds == None):
        raise ValueError(
            "The file was not reprojected properly and therefore was not saved. One of your key-worded argument is problematic %s " % kwargs)
    ds = None
    print("Warp completed. See %s " % output_geotiff)
    return
def dxf_to_shapefile(input_dxf, output_file, crs):
    """
    Convert DXF (mineralized ore bodies lets say) to a simplified polygon shapefile.
    Parameters
    ----------
    input_dxf : str
        input dxf path
    output_file : str
        output folder path
    crs : str
        coordinate system string like 'epsg:26921'
    Returns
    -------
    None
    """
    gdf = gpd.read_file(input_dxf)
    original_crs = crs
    gdf = gdf.set_crs(original_crs)
    union = gdf.geometry.unary_union
    s = gpd.GeoSeries(union, crs=original_crs)
    d = {
        "value": 1,
        "geometry": s
    }
    result = gpd.GeoDataFrame(d)
    result = result.explode()
    result.to_file(output_file)
    print("Conversion was successful see folder " + output_file)
    return
def geotiff_to_jpg(input_geotiff, output_path, nb_quantiles=20):
    """
    Convert a raster (geotiff) to jpg and show it in the console.
    Parameters
    ----------
    input_geotiff : str
        path to input geotiff
    output_path : str
        full path to save the jpg.
    nb_quantiles : int, default=20
        number of quantiles to use to bin the data into equals interval for the colorscale (Turbo). Default = 20
    Returns
    -------
    fig : Matplotlib.Figure
        Figure from matplotlib
    ax : Matplotlib.Axe
        Axe object from matplotlib
    Examples
    --------
        Convert a batch of PCs geotiffs from a folder and save them as jpg.
        >>> import glob
        >>> import os
        >>> input_folder = 'my_pcs_rasters/'
        >>> my_output_folder = 'my_jpgs/'
        >>> tifs = glob.glob(input_folder + '*.tif')
        >>> for t in tifs:
        >>>     filename,  extension, directory= utilities.Path_Info(t)
        >>>     utilities.raster_to_jpg(t, os.path.join(my_output_folder, filename + '.jpg'))
    """
    dataset = gdal.Open(input_geotiff, gdal.GA_ReadOnly)
    # Note GetRasterBand() takes band no. starting from 1 not 0
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()
    data = np.array(arr)
    #need to flip the data (y are negative
    data = np.flipud(data)
    increment = 1 / nb_quantiles
    bounds = np.nanquantile(data, np.arange(0, 1, increment))
    cmap = colors.Colormap('turbo')
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    plt.rcParams["figure.figsize"] = (10, 10)
    fig, ax = plt.subplots()
    plt.imshow(data, origin='lower',cmap='turbo', norm=norm, interpolation='bicubic')  # norm=colors.TwoSlopeNorm(vcenter=0))
    plt.axis('off')
    # X-axis tick label
    # plt.xticks(color='w')
    # Y-axis tick label
    # plt.yticks(color='w')
    # plt.tick_params(left=False,bottom=False)
    plt.box(False)
    plt.tight_layout()
    # be sure not to cause crashe
    filename, extension, directory = Path_Info(input_geotiff)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(output_path)
    plt.show()
    return fig, ax
def raster_to_shapefile(input_raster, dst_shapefile, output_type='int'):
    """
    Converts a singleband raster to a vector format (shapefile). Band 1 values will be written in 'value' field of the resulting shapefile.
    Parameters
    ----------
    input_raster : str
        path to the input geotiff raster
    dst_shapefile : str
        path to the output shapefile
    output_type : str, default = 'int'
        can be 'int' or 'float'. For now we use this argumment (Jan-20-2022). There may be a way to detect what is the format in the input raster and write the shapefile automatilcally.
    Returns
    -------
    None
    """
    file, ext, output_directory = Path_Info(dst_shapefile)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    shapefile_only = file + ext
    print(shapefile_only)
    src_ds = gdal.Open(input_raster)
    if src_ds is None:
        print('Unable to open %s' % input_raster)
        sys.exit(1)
    srcband = src_ds.GetRasterBand(1)
    # test = gdal.GetDataTypeName(srcband.DataType)
    # print(test)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    # dst_layername = "PolyFtr"
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_shapefile)
    if (dst_ds == None):
        print('Unable to open %s it is probably opened by another application? Check your QGIS maybe.' % dst_shapefile)
        sys.exit(1)
    dst_layer = dst_ds.CreateLayer(file, srs=srs)
    # there may be a way to detectect the rasters' input and translate that to OGR type
    if (output_type == 'int'):
        output_type = ogr.OFTInteger
    else:
        output_type = ogr.OFTReal
    newField = ogr.FieldDefn('value', output_type)
    dst_layer.CreateField(newField)
    gdal.Polygonize(srcband, srcband, dst_layer, 0, [], callback=None)
    dst_layer = None
    dst_ds = None
    print("Conversion Done. See %s" % dst_shapefile)
    return None
def csv_to_shapefile(input_csv, output_shapefile, x_col, y_col, crs, columns=None):
    df = pd.read_csv(input_csv)
    if(columns != None):
        if(type(columns) == list):
            columns.append(x_col)
            columns.append(y_col)
            df=df[columns]
        else:
            raise TypeError("The type of input argument \"columns\" should be a list.")
    x_empty_sum = df[x_col].isnull().sum()
    y_empty_sum = df[y_col].isnull().sum()
    if(x_empty_sum > 0 or y_empty_sum >0 ):
        print("Your input csv has %s x and %s y empty cells for the coordinates. These records will be dropped for the shapefile." % x_empty_sum, y_empty_sum)
        df.dropna(axis=0, how='any', subset=[x_col, y_col], inplace=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs=crs)
    file, ext, directory = Path_Info(output_shapefile)
    if (os.path.exists(directory) != True):
        os.makedirs(directory)
    gdf.to_file(output_shapefile)
    return gdf
def compare_models(metrics, **kwargs):
    """
    Compare models and create a plot of metrics provided in the nested dictionary
    
    Parameters
    ----------
    metrics : dict
        Dictionary with all models and related metrics
    set_size : OPTIONAL, tuple
        Set the figure size in inches -> Example: (15, 11)
    output_directory : OPTIONAL, str
        The path of the plot
    
    Example:
    
    metrics = {}
    metrics["CatBoost"] = {"f1": 0.78, "prec": 0.79, "acc":0.77}
    metrics["Random Forest"] = {"f1": 0.68, "prec": 0.69, "acc":0.67}
   
    compare_models(metrics, output_directory="./")
    
    """
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.reindex(metrics_df.mean().sort_values().index, axis=1)
    if "set_size" in kwargs:
        ax = metrics_df.transpose().plot.barh(figsize=(kwargs['set_size'][0], kwargs['set_size'][1]), zorder=10)
    else:
        ax = metrics_df.transpose().plot.barh(figsize=(12,8), zorder=10)
    ax.set_title('Model Comparison', fontsize=20)
    ax.set_xlabel("Score", size = 20)
    ax.set_yticklabels(metrics_df.transpose().index.values.tolist(), rotation=45, 
                   rotation_mode="anchor", 
                   size = 15)
    ax.set_xlim([0.0, 1.0])
    ax.set_xticks(np.linspace(0, 1, 11, endpoint = True))
    ax.legend()
    ax.grid(zorder=5)
    leg = ax.legend(loc='upper right', fontsize=15, ncol=1, framealpha=.5)
    leg.set_zorder(100)
    if "output_directory" in kwargs:
        plt.savefig(os.path.join(kwargs['output_directory'], 'model_comparison.png'), dpi=100)
    return metrics_df
def geotiff_to_hist(FullPath, bins=10, masked=True, title='Histogram', figure_size=None, **kwargs):
    from matplotlib.patches import Rectangle
    from rasterio.io import DatasetReader
    """Display a histogram of the given geotiff with percentile
    
    Parameters
    ----------
    source : array or dataset object opened in 'r' mode or Band or tuple(dataset, bidx)
        Input data to display.
        The first three arrays in multi-dimensional
        arrays are plotted as red, green, and blue.
    bins : int, optional
        Compute histogram across N bins.
    masked : bool, optional
        When working with a `rasterio.Band()` object, specifies if the data
        should be masked on read.
    title : str, optional
        Title for the figure.
    **kwargs : optional matplotlib keyword arguments
    
    Examples
    ----------
    output_directory = "./"
    tif_file = "Magnetics_100m_regional_RTP_UC100.tif"
    FullPath = os.path.join(output_directory, tif_file)
    utilities.geotiff_to_hist(FullPath, bins = 20, title=tif_file, figure_size=(10,8), masked=False)
    #utilities.geotiff_to_hist(FullPath, bins = 20, title=tif_file, figure_size=(10,8))
    #utilities.geotiff_to_hist(FullPath, bins = 20, title=tif_file)
    
    """
    source = rasterio.open(FullPath)
    if isinstance(source, DatasetReader):
        arr = source.read(masked=masked)
    elif isinstance(source, (tuple, rasterio.Band)):
        arr = source[0].read(source[1], masked=masked)
    else:
        arr = source
    # The histogram is computed individually for each 'band' in the array
    # so we need the overall min/max to constrain the plot
    rng = np.nanmin(arr), np.nanmax(arr)
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr.flatten(), 0).T
        colors = ['gold']
    else:
        arr = arr.reshape(arr.shape[0], -1).T
        colors = ['red', 'green', 'blue', 'violet', 'gold', 'saddlebrown']
    # The goal is to provide a curated set of colors for working with
    # smaller datasets and let matplotlib define additional colors when
    # working with larger datasets.
    if arr.shape[-1] > len(colors):
        n = arr.shape[-1] - len(colors)
        colors.extend(np.ndarray.tolist(plt.get_cmap('Accent')(np.linspace(0, 1, n))))
    else:
        colors = colors[:arr.shape[-1]]
    # If a rasterio.Band() is given make sure the proper index is displayed
    # in the legend.
    if isinstance(source, (tuple, rasterio.Band)):
        labels = [str(source[1])]
    else:
        labels = (str(i + 1) for i in range(len(arr)))
    ax = plt.gca()
    fig = ax.get_figure()
    # Colours for different percentiles
    perc_25_colour = 'gold'
    perc_50_colour = 'mediumaquamarine'
    perc_75_colour = 'deepskyblue'
    perc_95_colour = 'peachpuff'
    counts, bins, patches = ax.hist(arr,
                            bins=bins,
                            color=colors,
                            label=labels,
                            range=rng,
                            facecolor=perc_50_colour,
                            edgecolor='gray',
                            **kwargs)
    # Change the colors of bars at the edges
    twentyfifth, seventyfifth, ninetyfifth = np.percentile(arr, [25, 75, 95])
    for patch, leftside, rightside in zip(patches, bins[:-1], bins[1:]):
        if rightside < twentyfifth:
            patch.set_facecolor(perc_25_colour)
        elif leftside > ninetyfifth:
            patch.set_facecolor(perc_95_colour)
        elif leftside > seventyfifth:
            patch.set_facecolor(perc_75_colour)
    # Calculate bar centre to display the count of data points and %
    bin_x_centers = 0.5 * np.diff(bins) + bins[:-1]
    bin_y_centers = ax.get_yticks()[1]
    # Display the the count of data points and % for each bar in histogram
    for i in range(len(bins)-1):
        bin_label = "{0:,}".format(counts[i]) + "  ({0:,.2f}%)".format((counts[i]/counts.sum())*100)
        plt.text(bin_x_centers[i], bin_y_centers, bin_label, rotation=90, rotation_mode='anchor')
    #create legend
    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [perc_25_colour, perc_50_colour, perc_75_colour, perc_95_colour]]
    labels= ["0-25 Percentile","25-50 Percentile", "50-75 Percentile", ">95 Percentile"]
    ax.legend(handles, labels, bbox_to_anchor=(0.5, 0., 0.80, 0.99))
    ax.set_title(title, fontweight='bold', y=-0.01)
    ax.grid(True)
    ax.set_xlabel('DN')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    ax.set_ylim([1e0, ax.get_yticks()[-1] * 10])
    dirname = os.path.dirname(FullPath)
    filename = os.path.basename(FullPath).split('.')[0] + '.png'
    if figure_size == None:
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    else:
        plt.rcParams["figure.figsize"] = (figure_size[0], figure_size[1])    
    plt.tight_layout()
    fig.savefig(os.path.join(dirname, filename))  
    return
def __get_xy_properties(xy):
    """Get rasterio transform from a two-column xy coordinates array.
    Parameters
    ----------
    xy : numpy.array((m, 2))
        2-column xy data, x coordinates in first column, y coordinate in second.
    Returns
    -------
    dict :
        Dictionary of coordinates properties.
    """
    try:
        m, n = np.shape(xy)
        assert n == 2
    except:
        print('xy should be an array (m, 2)')
    x0, y0 = np.min(xy, axis=0)
    xf, yf = np.max(xy, axis=0)
    steps = np.abs(xy[:-1] - xy[1:])
    res =np.min(steps[steps != 0])
    width = int(np.abs((xf - x0) / res)) + 1
    height = int(np.abs((yf - y0) / res)) + 1
    print(f'inferred resolution from coordinates is {res}')
    assert len(np.unique(xy % res)), 'your coordinates seem irregularly spaced'
    # Indices
    xy0 = ((xy - np.array([x0, yf])) / np.array([res, -res])).astype('int')
    arr = np.array([xy0[:, 1], xy0[:, 0]])
    indices = np.ravel_multi_index(arr, (height, width))
    # Transform
    trans = rasterio.Affine.translation(
        x0 - res / 2, yf + res / 2) * rasterio.Affine.scale(res, -res)
    properties = {
        'transform': trans,
        'indices': indices,
        'width': width,
        'height': height,
        'n_pts': m
    }
    return properties
def __xydata_to_gtiffs(
        outfolder,
        xy,
        data,
        crs,
        nodata=-9999,
        filenames=None
):
    """From m xy-coordinates and (m, n) data, generate n geotiffs.
    Parameters
    ----------
    outfolder : str
        Path to (existing) basefolder to write all geotiffs.
    xy : numpy.array((m, 2))
        2-column xy data, x coordinates in first column, y coordinate in second.
    data : numpy.array((m, n))
        Column-data to write as geotiffs. Data-points in columns, every column
        becomes a geotiff.
    crs :
        Coordinate reference system of xy data.
    nodata :
        Value to write to geotiff nodata.
    filenames : list of strings, opt.
        List of filenames corresponding to columns.
    Returns
    -------
    list of strings
        List of files written.
    """
    # Shared array properties from xy data
    properties = __get_xy_properties(xy)
    transform = properties['transform']
    indices = properties['indices']
    width = properties['width']
    height = properties['height']
    l = properties['n_pts']
    if data.ndim == 1:
        assert len(data) == l, 'your data size and xy size do not match'
        data = np.array([data]).transpose()
    elif data.ndim == 2:
        m, n = np.shape(data)
        assert m == l, 'your data size and xy size do not match'
    else:
        raise ValueError('data has invalid shape')
    m, n = np.shape(data)
    # Define filenames
    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(filenames, Container):
        assert len(filenames) == n, 'filenames length does not match with data'
        filenames = [str(name) for name in filenames]
    elif filenames == None:
        filenames = [f'data_{i}' for i in range(n)]
    else:
        raise ValueError('Inappropriate filenames value')
    # Iterate to write geotiffs
    outfiles = []
    for i in range(n):
        array = np.ones(height * width) * nodata
        array[indices] = data[:, i]
        array = array.reshape(height, width)
        filename = f'{os.path.join(outfolder, filenames[i])}.tif'
        meta = {
            'driver': 'Gtiff',
            'width': width,
            'height': height,
            'count': 1,
            'dtype': array.dtype,
            'crs': crs,
            'transform': transform,
            'nodata': nodata
        }
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(array, 1)
        outfiles.append(filename)
    return outfiles
def df_to_gtiffs(
        df,
        crs,
        outfolder='.',
        x_col='x',
        y_col='y',
        features=None,
        nodata=-9999,
        verbose=True
):
    """From a dataframe of xy coordinates, write other columns to geotiff.
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing x and y coordinates in their column, and other
        columns being datapoints.
    crs :
        Coordinate reference system of xy data.
    outfolder : str, def = '.'
        Path to (existing) basefolder to write all geotiffs.
    x_col : str, def = 'x'
        Header of column containing x-coordinates.
    y_col : str, def = 'y'
        Header of column containing y-coordinates.
    features : list of strings, opt.
        Subset of data columns to write to geotiff, default is all non-xy.
    nodata :
        Value to write to geotiff nodata.
    verbose : bool, def = True
    Returns
    -------
    list of strings
        List of files written.
    """
    xy = df[[x_col, y_col]].values
    if features is None:
        features = list(df.columns)
        features.remove(x_col)
        features.remove(y_col)
    data = df[features].values
    outfiles = __xydata_to_gtiffs(
        outfolder,
        xy,
        data,
        crs,
        nodata=nodata,
        filenames=features
    )
    if verbose:
        print(f'\n{len(outfiles)} geotiffs written.')
        for file in outfiles:
            print(file)
    return outfiles