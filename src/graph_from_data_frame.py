import pandas as pd
import networkx as nx
import numpy as np

def graph_from_data_frame(d, directed=True, vertices=None):
    # Convert input to DataFrame
    d = pd.DataFrame(d)
    
    if vertices is not None:
        vertices = pd.DataFrame(vertices)
    
    if d.shape[1] < 2:
        raise ValueError("The data frame should contain at least two columns")
    
    if d.iloc[:, 0:2].isna().any().any():
        print("Warning: In d, NA elements were replaced with string 'NA'.")
        d.iloc[:, 0:2] = d.iloc[:, 0:2].fillna("NA")
    
    if vertices is not None and vertices.iloc[:, 0].isna().any():
        print("Warning: In vertices[, 1], NA elements were replaced with string 'NA'.")
        vertices.iloc[:, 0] = vertices.iloc[:, 0].fillna("NA")
    
    # Combine unique vertex names from both columns of d
    names = pd.unique(d.iloc[:, [0, 1]].values.ravel())
    
    if vertices is not None:
        if vertices.shape[0] == 0:
            raise ValueError("Vertex data frame contains no rows")
        
        vertices = vertices.iloc[:, 0].values
        if any(pd.duplicated(vertices)):
            raise ValueError("Duplicate vertex names")
        
        if not all(np.isin(names, vertices)):
            raise ValueError("Some vertex names in edge list are not listed in vertex data frame")
    
    # Create empty graph
    g = nx.DiGraph() if directed else nx.Graph()
    
    # Add vertices (nodes) to the graph
    g.add_nodes_from(names)
    
    # Add vertex attributes if vertices DataFrame is provided
    if vertices is not None and vertices.shape[1] > 1:
        for col in vertices.columns[1:]:
            g.nodes[vertices.iloc[:, 0]]['attr'] = vertices[col].values
    
    # Add edges to the graph (from, to, and any additional attributes)
    from_nodes = d.iloc[:, 0].astype(str).values
    to_nodes = d.iloc[:, 1].astype(str).values
    edges = list(zip(from_nodes, to_nodes))
    
    edge_attrs = {}
    if d.shape[1] > 2:
        for i in range(2, d.shape[1]):
            newval = d.iloc[:, i].values
            edge_attrs[names[d.columns[i]]] = newval
    
    # Add edges to the graph with attributes
    for edge in edges:
        g.add_edge(edge[0], edge[1], **edge_attrs)
    
    return g

