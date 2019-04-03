from __future__ import print_function, division, absolute_import
import numpy as np
from ..utils.geo import get_subgraph
import shapely
from shapely.geometry import Point
import networkx as nx
import geopandas as gpd
import fiona


class Node(object):
    """An object to hold node attributes.

    Attributes
    ----------
    idx : int
        The numerical index of the node. Used as a unique identifier
        when the nodes are added to the graph.
    x : `int` or `float`
        Numeric x location of the node, in either a geographic CRS or in pixel
        coordinates.
    y : `int` or `float`
        Numeric y location of the node, in either a geographic CRS or in pixel
        coordinates.

    """

    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Node {} at ({}, {})'.format(self.idx, self.x, self.y)


class Edge(object):
    """An object to hold edge attributes.

    Attributes
    ----------
    nodes : 2-`tuple` of :class:`Node` s
        :class:`Node` instances connected by the edge.
    weight : int or float
        The weight of the edge.

    """

    def __init__(self, nodes, edge_weight=None):
        self.nodes = nodes
        self.weight = edge_weight

    def __repr__(self):
        return 'Edge between {} and {} with weight {}'.format(self.nodes[0],
                                                              self.nodes[1],
                                                              self.weight)

    def set_edge_weight(self, normalize_factor=None, inverse=False):
        """Get the edge weight based on Euclidean distance between nodes.

        Note
        ----
        This method does not account for spherical deformation (i.e. does not
        use the Haversine equation). It is a simple linear distance.

        Arguments
        ---------
        normalize_factor : `int` or `float`, optional
            a number to multiply (or divide, if
            ``inverse=True``) the Euclidean distance by. Defaults to ``None``
            (no normalization)
        inverse : bool, optional
            if ``True``, the Euclidean distance weight will be divided by
            ``normalize_factor`` instead of multiplied by it.
        """
        weight = np.linalg.norm(
            np.array((self.nodes[0].x, self.nodes[0].y)) -
            np.array((self.nodes[1].x, self.nodes[1].y)))

        if normalize_factor is not None:
            if inverse:
                weight = weight/normalize_factor
            else:
                weight = weight*normalize_factor
        self.weight = weight

    def get_node_idxs(self):
        """Return the Node.idx for the nodes in the edge."""
        return (self.nodes[0].idx, self.nodes[1].idx)


class Path(object):
    """An object to hold :class:`Edge` s with common properties.

    Attributes
    ----------
    edges : `list` of :class:`Edge` s
        A `list` of :class:`Edge` s
    properties : dict
        A dictionary of property: value pairs that provide relevant metadata
        about edges along the path (e.g. road type, speed limit, etc.)

    """

    def __init__(self, edges=None, properties=None):
        self.edges = edges
        if properties is None:
            properties = {}
        self.properties = properties

    def __repr__(self):
        return 'Path including {}'.format([e for e in self.edges])

    def add_edge(self, edge):
        """Add an edge to the path."""
        self.edges.append(edge)

    def set_edge_weights(self, data_key=None, inverse=False, overwrite=True):
        """Calculate edge weights for all edges in the Path."""
        for edge in self.edges:
            if not overwrite and edge.weight is not None:
                continue
            if data_key is not None:
                edge.set_edge_weight(
                    normalize_factor=self.properties[data_key],
                    inverse=inverse)
            else:
                edge.set_edge_weight()

    def add_data(self, property, value):
        """Add a property: value pair to the Path.properties attribute."""
        self.properties[property] = value

    def __iter__(self):
        """Iterate through edges in the path."""
        yield from self.edges


def geojson_to_graph(geojson, graph_name=None, retain_all=True,
                     valid_road_types=None, road_type_field='type', edge_idx=0,
                     first_node_idx=0, weight_norm_field=None, inverse=False,
                     verbose=False):
    """Convert a geojson of path strings to a network graph.

    Arguments
    ---------
    geojson : str
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
    edge_idx : int, optional
        The first index to use for an edge. This can be set to a higher value
        so that a graph's edge indices don't overlap with existing values in
        another graph.
    first_node_idx : int, optional
        The first index to use for a node. This can be set to a higher value
        so that a graph's node indices don't overlap with existing values in
        another graph.
    weight_norm_field : str, optional
        The name of a field in `geojson` to pass to argument ``data_key`` in
        :func:`Path.set_edge_weights`. Defaults to ``None``, in which case
        no weighting is performed (weights calculated solely using Euclidean
        distance.)
    verbose : bool, optional
        Verbose print output. Defaults to ``False`` .

    Returns
    -------
    G : :class:`networkx.MultiDiGraph`
        A :class:`networkx.MultiDiGraph` containing all of the nodes and edges
        from the geojson (or only the largest connected component if
        `retain_all` = ``False``). Edge lengths are weighted based on
        geographic distance.

    """
    # due to an annoying feature of loading these graphs, the numeric road
    # type identifiers are presented as string versions. we therefore reformat
    # the valid_road_types list as strings.
    if valid_road_types is not None:
        valid_road_types = [str(i) for i in valid_road_types]

    # create the graph as a MultiGraph and set the original CRS to EPSG 4326

    # extract nodes and paths
    nodes, paths = get_nodes_paths(geojson,
                                   valid_road_types=valid_road_types,
                                   first_node_idx=first_node_idx,
                                   road_type_field=road_type_field,
                                   verbose=verbose)
    # nodes is a dict of node_idx: node_params (e.g. location, metadata)
    # pairs.
    # paths is a dict of path dicts. the path key is the path_idx.
    # each path dict has a list of node_idxs as well as properties metadata.

    # initialize the graph object
    G = nx.MultiDiGraph(name=graph_name, crs={'init': 'epsg:4326'})
    if not nodes:  # if there are no nodes in the graph
        return G
    if verbose:
        print("nodes:", nodes)
        print("paths:", paths)
    # add each osm node to the graph
    for node in nodes:
        G.add_node(node, **{'x': node.x, 'y': node.y})
    # add each path to the graph
    for path in paths:
        # calculate edge length using euclidean distance and a weighting term
        path.set_edge_weights(data_key=weight_norm_field, inverse=inverse)
        edges = [(*edge.nodes, edge.weight) for edge in path]
        G.add_weighted_edges_from(edges)
    if not retain_all:
        # keep only largest connected component of graph unless retain_all
        # code modified from osmnx.core.get_largest_component & induce_subgraph
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = get_subgraph(G, largest_cc)

    return G


def get_nodes_paths(vector_file, first_node_idx=0, node_gdf=gpd.GeoDataFrame(),
                    valid_road_types=None, road_type_field='type',
                    verbose=False):
    """
    Extract nodes and paths from a vector file.

    Arguments
    ---------
    vector_file : str
        Path to an OGR-compatible vector file containing line segments (e.g.,
        JSON response from from the Overpass API, or a SpaceNet GeoJSON).
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
        nodes : list
            A `list` of :class:`Node` s to be added to the graph.
        paths : list
            A list of :class:`Path` s containing the :class:`Edge` s and
            :class:`Node` s to be added to the graph.

    """
    node_idx = first_node_idx
    if valid_road_types is None:
        valid_road_types = ['1', '2', '3', '4', '5', '6', '7']

    with fiona.open(vector_file, 'r') as source:
        paths = []

        for feature in source:
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
                print("\ngeom: {}".format(feature['geometry']))
                print("   properties:", properties)
                print("   road_type:", road_type)
                print("   road_type type: {}".format(type(road_type)))

            # check if road type allowable and a valid road, skip if not
            geom = feature['geometry']
            if geom['type'] == 'LineString' or \
                    geom['type'] == 'MultiLineString':
                if road_type not in valid_road_types or \
                        'LINESTRING EMPTY' in properties.values():
                    if verbose:
                        print("Invalid road type, skipping...")
                    continue

            if geom['type'] == 'LineString':
                linestring = shapely.geometry.shape(geom)
                edges, node_idx, node_gdf = linestring_to_edge(
                    linestring, node_idx, node_gdf)

            elif geom['type'] == 'MultiLineString':
                # do the same thing as above, but do it for each piece
                edges = []
                for linestring in shapely.geometry.shape(geom):
                    edge_set, node_idx, node_gdf = linestring_to_edge(
                        linestring, node_idx, node_gdf)
                    edges.extend(edge_set)

            path = Path(edges=edges, properties=properties)
            paths.append(path)

        nodes = node_gdf['node'].tolist()

    return nodes, paths


def linestring_to_edge(linestring, node_idx, node_gdf=gpd.GeoDataFrame()):
    """Collect nodes in a linestring and add them to an edge.

    Arguments
    ---------
    linestring : :class:`shapely.geometry.LineString`
        A :class:`shapely.geometry.LineString` object to extract nodes and
        edges from.
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
    road_type_field : str, optional
        The attribute containing road type information in `properties`.
        Defaults to ``'type'`` .

    Returns
    -------
    edge, node_idx, node_gdf : tuple
        edge : :class:`Edge`
            An :class:`Edge` instance containing all of the points in
            `linestring` as :class:`Node` s in ``edge.nodes``.
        node_idx : int
            The numeric index of the last node in the path. Passed back so that
            `Node.idx` s can be incremented and therefore won't overlap.
        node_gdf : :class:`geopandas.GeoDataFrame`
            A :class:`geopandas.GeoDataFrame` containing all nodes already
            added to the graph so that future iterations through
            `process_linestring` don't add new nodes at the same location.

    """

    node_list = []
    edges = []

    for point in linestring.coords:
        point_shp = shapely.geometry.shape(Point(point))

        try:
            matching_nodes = node_gdf[
                node_gdf.distance(point_shp) == 0.0]['node'].values
        except AttributeError:
            matching_nodes = []

        if len(matching_nodes) == 1:  # if the current node isn't already there
            node = matching_nodes[0]
        else:
            node = Node(idx=node_idx, x=point[0], y=point[1])
            node_idx += 1
            # add the new node to the gdf
            node_gdf = node_gdf.append({'geometry': point_shp, 'node': node},
                                       ignore_index=True)
        node_list.append(node)
        if len(node_list) > 1:
            edges.append(Edge(nodes=node_list[-2:]))

    return edges, node_idx, node_gdf
