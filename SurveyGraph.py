import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from src.make_synthetic_data import make_synthetic_data
from src.make_projection import make_projection
import matplotlib.cm as cm
import numpy as np

st.set_page_config(page_title="Synthetic Graph Generator")

# App Title
st.title("🧠 Generate Synthetic Attitude Network")

with st.sidebar:
    st.header("⚙️ Synthetic Data Parameters")

    # Select Layer
    layer = st.selectbox("Select Layer", ["agent", "symbolic"], index=0)

    # Synthetic Data Controls
    nrow = st.slider("Number of rows (individuals)", 10, 500, 10)
    ncol = st.slider("Number of columns (statements)", 5, 50, 10)
    polarisation = st.slider("Polarisation", 0.0, 1.0, 0.22)
    correlation = st.slider("Correlation", 0.0, 1.0, 0.5)

    # Projection Settings
    threshold_method = st.selectbox("Threshold Method", ["target_lcc", "target_ad", "raw_similarity"], index=0)
    method_value = st.slider("Method Value", 0.0, 1.0, 0.99)
    centre = st.checkbox("Centre data", value=True)
    
    st.info("This is the Python implementation for SurveyPackage originally written in R and C++.. To read more about SurveyGraph package: https://surveygraph.ie/")

# Main logic block
try:
    S = make_synthetic_data(nrow=nrow, ncol=ncol, polarisation=polarisation, correlation=correlation)

    names1 = pd.DataFrame({
        'id': list(range(1, len(S['group']) + 1)),
        'group': S['group']
    })

    # Projection based on layer
    if layer == "agent":
        e1 = make_projection(S, "agent", threshold_method=threshold_method, method_value=method_value, centre=centre)
    else:
        e1 = make_projection(S, "symbolic", threshold_method="raw_similarity", method_value=-1, centre=False)

    if e1.empty:
        st.warning("⚠️ No edges were created. Try lowering the threshold or adjusting other settings.")
    else:
        # Create graph
        g1 = nx.from_pandas_edgelist(e1, source='u', target='v', edge_attr=True, create_using=nx.Graph())

        # Add node attributes
        for _, row in names1.iterrows():
            if row['id'] in g1.nodes:
                g1.nodes[row['id']]['group'] = row['group']

        # Layout and plotting
        pos = nx.spring_layout(g1, seed=42)
        
        # Extract group information for each node
        groups = nx.get_node_attributes(g1, 'group')
        group_ids = list(set(groups.values()))
        group_to_color = {group: i for i, group in enumerate(sorted(group_ids))}
        
        # Normalize group values to colormap
        color_map = cm.get_cmap('tab10', len(group_ids))  # You can try 'tab10', 'Accent', etc.
        node_colors = [color_map(group_to_color[groups[node]]) for node in g1.nodes()]
        
        # Optionally vary node size based on degree
        node_sizes = [50 + 30 * g1.degree(node) for node in g1.nodes()]  # scalable sizes
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(
            g1, pos,
            with_labels=False,
            node_size=node_sizes,
            width=0.5,
            edge_color='lightgray',
            node_color=node_colors,
            alpha=0.9,
            ax=ax
        )
        
        st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()
