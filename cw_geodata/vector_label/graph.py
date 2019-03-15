from __future__ import print_function, division, absolute_import
import time
import numpy as np
from osmnx.utils import log
from osmnx import core
import shapely
from shapely.geometry import Point
import networkx as nx
import geopandas as gpd
import fiona


def geojson_to_graph(vector_file, graph_name=None, retain_all=True,
                     network_type='all_private', valid_road_types=None,
                     road_type_field='type', first_path_idx=0,
                     first_node_idx=0, verbose=False):
    """Convert a geojson of path strings to a network graph.

    Arguments
    ---------
    vector_file : str
        Path to a geojson file (or any other OGR-compatible vector file) to
        load network edges and nodes from.
    graph_name : str, optional
        Name of the graph. If not provided, graph will be named ``'unnamed'`` .
    retain_all : bool, optional
        If ``True`` , the entire graph will be returned even if some parts are
        not connected. Defaults to ``True``.
    valid_road_types : :class:`list` of :class:`int` s, optional
        The road types to permit in the graph. If not provided, it's assumed
        that all road types are permitted. The possible values are integers
        ``1``-``7``, which map as follows::

            1: Motorway
            2: Primary
            3: Secondary
            4: Tertiary
            5: Residential
            6: Unclassified
            7: Cart track

    road_type_field : str, optional
        The name of the property in the vector data that delineates road type.
        Defaults to ``'type'`` .
    first_path_idx : int, optional
        The first index to use for a path. This can be set to a higher value
        so that a graph's path indices don't overlap with existing values in
        another graph.
    first_node_idx : int, optional
        The first index to use for a node. This can be set to a higher value
        so that a graph's node indices don't overlap with existing values in
        another graph.
    verbose : bool, optional
        Verbose print output. Defaults to ``False`` .

    ..deprecated:: 0.1.1
        The `network_type` argument no longer has any effect in
        :func:`osmnx.core.add_paths` and is now ignored.

    Returns
    -------
    G : :class:`networkx.MultiGraph`
        A :class:`networkx.MultiGraph` containing all of the nodes and edges
        from the geojson (or only the largest connected component if
        `retain_all` = ``False``). Edge lengths are weighted based on
        geographic distance.
    """
    log('Creating networkx graph...')
    start_time = time.time()
    # create the graph as a MultiGraph and set the original CRS to EPSG 4326

    # extract nodes and paths
    nodes, paths = get_nodes_paths(vector_file,
                                   valid_road_types=valid_road_types,
                                   first_path_idx=first_path_idx,
                                   first_node_idx=first_node_idx,
                                   road_type_field=road_type_field,
                                   verbose=verbose)
    # nodes is a dict of node_idx: node_params (e.g. location, metadata)
    # pairs.
    # paths is a dict of path dicts. the path key is the path_idx.
    # each path dict has a list of node_idxs as well as properties metadata.

    # initialize the graph object
    G = nx.MultiGraph(name=graph_name, crs={'init': 'epsg:4326'})
    if not nodes:  # if there are no nodes in the graph
        return G
    if verbose:
        print("nodes:", nodes)
        print("paths:", paths)
    # add each osm node to the graph
    for node, data in nodes.items():
        G.add_node(node, **data)
    # add each path to the graph
    G = core.add_paths(G, paths)
    if not retain_all:
        # keep only largest connected component of graph unless retain_all
        G = core.get_largest_component(G)
    # use length (great circle distance between nodes) as weight
    # TODO: set something up here to enable a different edge weighting scheme
    G = core.add_edge_lengths(G)
    log('Created graph with {} nodes and {} edges' +
        ' in {:,.2f} seconds'.format(len(list(G.nodes())),
                                     len(list(G.edges())),
                                     time.time()-start_time))

    return G


def get_nodes_paths(vector_file, first_path_idx=0, first_node_idx=0,
                    node_gdf=gpd.GeoDataFrame(), valid_road_types=None,
                    road_type_field='type', verbose=True):
    """
    Construct dicts of nodes and paths.

    Arguments
    ---------
    vector_file : str
        Path to an OGR-compatible vector file containing line segments (e.g.,
        JSON response from from the Overpass API).
    first_path_idx : int, optional
        The first index to use for a path. This can be set to a higher value
        so that a graph's path indices don't overlap with existing values in
        another graph.
    first_node_idx : int, optional
        The first index to use for a node. This can be set to a higher value
        so that a graph's node indices don't overlap with existing values in
        another graph.
    node_gdf : :class:`geopandas.GeoDataFrame` , optional
        A :class:`geopandas.GeoDataFrame` containing nodes to add to the graph.
        New nodes will be added to this object incrementally during the
        function call.
    valid_road_types : :class:`list` of :class:`int` s, optional
        The road types to permit in the graph. If not provided, it's assumed
        that all road types are permitted. The possible values are integers
        ``1``-``7``, which map as follows::

            1: Motorway
            2: Primary
            3: Secondary
            4: Tertiary
            5: Residential
            6: Unclassified
            7: Cart track

    road_type_field : str, optional
        The name of the attribute containing road type information in
        `vector_file`. Defaults to ``'type'``.
    verbose : bool, optional
        Verbose print output. Defaults to ``False``.

    Returns
    -------
    nodes, paths : `tuple` of `dict` s
        nodes : dict
            A `dict` of ``node_idx: node_parameters`` pairs containing relevant
            data for the node, i.e. geographic location and any metadata
            contained in ``node['properties']`` .
        paths : dict
            A `dict` of ``path_idx: path_parameters`` pairs.
            ``path_parameters`` is a `dict` containing the `node_idx` s for
            each node along `path`, along with any metadata contained in
            ``properties`` (this is the same as ``node[properties]`` ).

    """
    path_idx = first_path_idx
    node_idx = first_node_idx
    if valid_road_types is None:
        valid_road_types = [1, 2, 3, 4, 5, 6, 7]

    with fiona.open(vector_file, 'r') as source:
        nodes = {}
        paths = {}

        for feature in source:
            geom = feature['geometry']
            properties = feature['properties']
            # TODO: create more adjustable filter
            if road_type_field in properties:
                road_type = properties[road_type_field]
            elif 'highway' in properties:
                road_type = properties['highway']
            elif 'road_type' in properties:
                road_type = properties['road_type']
            else:
                road_type = 'None'

            if verbose:
                print("\ngeom:", geom)
                print("   properties:", properties)
                print("   road_type:", road_type)

            # check if road type allowable and a valid road, skip if not
            if geom['type'] == 'LineString' or \
                    geom['type'] == 'MultiLineString':
                if road_type not in valid_road_types or \
                        'LINESTRING EMPTY' in properties.values():
                    if verbose:
                        print("Invalid road type, skipping...")
                    continue

            path_idx += 1

            if geom['type'] == 'LineString':
                linestring = shapely.geometry.shape(geom)
                path, node_dict, node_idx, node_gdf = process_linestring(
                    linestring, path_idx, node_idx, node_gdf,
                    properties=properties
                    )
                node_idx += 1  # increment for next iteration in for loop
                path_idx += 1  # increment for next iteration in for loop
                nodes.update(node_dict)  # keys=node_idxs, values=node params
                paths[path_idx] = path  # see process_linestring() returns

            elif geom['type'] == 'MultiLineString':
                # do the same thing as above, but do it for each piece
                for linestring in shapely.geometry.shape(geom):
                    path, node_dict, node_idx, node_gdf = process_linestring(
                        linestring, path_idx, node_idx, node_gdf,
                        properties=properties
                        )
                    node_idx += 1
                    path_idx += 1
                    nodes.update(node_dict)
                    paths[path_idx] = path

        source.close()

    return nodes, paths


def process_linestring(linestring, path_idx, node_idx,
                       node_gdf=gpd.GeoDataFrame(), properties={},
                       road_type_field='type'):
    """Collect nodes in a linestring and add them all to a path.

    Arguments
    ---------
    linestring : :class:`shapely.geometry.LineString`
        A :class:`shapely.geometry.LineString` object to extract nodes and
        edges from.
    path_idx : int
        The index to assign the path for adding to the graph outside of this
        function. Assigned to ``path['osmid']`` , which is the ID attribute
        for the path `dict`.
    node_idx : int
        The index to assign the first node in the path. Assigned to
        ``node['osmid']`` , which is the ID attribute for the node `dict` .
        Will be incremented for each node in the path.
    node_gdf : :class:`geopandas.GeoDataFrame`, optional
        A :class:`geopandas.GeoDataFrame` of existing nodes already in a graph
        so that nodes at the same location aren't duplicated. The `node_idx`
        will be copied out of `node_gdf` rather than using the current value
        if assigning a node that already exists. An empty gdf is created if
        `node_gdf` isn't provided.
    properties : dict, optional
        A dictionary of properties for the linestring being processed. Loaded
        from the path's ``properties`` attribute from a geographic vector data
        source. Defaults to an empty `dict` .
    road_type_field : str, optional
        The attribute containing road type information in `properties`.
        Defaults to ``'type'`` .

    Returns
    -------
    path, nodes, node_idx, node_gdf : tuple
        path : dict
            A dictionary with the following keys:

            * ``'osmid'`` : the ``path_idx``
            * ``'nodes'`` : a list of ``node_idx`` s for nodes on the path
            * ``'highway'`` : the road type (defined by
              ``properties[road_type_field]`` ).

        nodes : dict
            A dictionary whose ``key, value`` pairs correspond to
            ``node_idx, node_data`` . Node data is a `dict` of:
            * ``'x'`` : x position of the node in the input CRS
            * ``'y'`` : y position of the node in the input CRS
            * ``'osmid'`` : ``node_idx``
        node_idx : int
            The numeric index of the last node in the path. Passed back so that
            `node_idx` s can be incremented so they don't overlap between
            different paths.
        node_gdf : :class:`geopandas.GeoDataFrame`
            A :class:`geopandas.GeoDataFrame` containing all nodes already
            added to the graph so that future iterations through
            `process_linestring` don't add new nodes at the same location.

    """
    path = {}
    path['osmid'] = path_idx  # id attribute for the path

    nodes = {}
    node_list = []
    node_idx += 1  # the last node_idx will have been used already

    for point in linestring.coords:
        point_shp = shapely.geometry.shape(Point(point))
        if node_gdf.size == 0:  # if this is the first node in the graph
            node_id = np.array([])
        else:
            # if it's the same point as an existing one, use the same node ID
            node_id = node_gdf[
                node_gdf.distance(point_shp) == 0.0]['osmid'].values

        if node_id.size == 0:  # if the current node isn't already in the graph
            node_id = node_idx
            # add the new node to the gdf
            node_gdf = node_gdf.append({'geometry': point_shp,
                                        'osmid': node_idx},
                                       ignore_index=True)
            node = {'x': point[0], 'y': point[1], 'osmid': node_id}
            node.update(properties)  # add properties attributes to node
            if road_type_field in properties:
                node['highway'] = properties[road_type_field]
            else:
                # should the next line be the string, or the numeric type ID?
                node['highway'] = 'unclassified'

            nodes[node_id] = node
            node_idx += 1
        else:
            node_id = node_id[0]  # get the existing node from the node_gdf

        node_list.append(node_id)

    path['nodes'] = node_list  # node IDs only
    path.update(properties)
    if road_type_field in properties:
        path['highway'] = properties[road_type_field]
    else:
        path['highway'] = 'unclassified'  # see question about node type

    return path, nodes, node_idx, node_gdf
