from src.make_synthetic_data import make_synthetic_data
from src.make_projection import make_projection
from src.graph_from_data_frame import graph_from_data_frame
import pandas as pd
import networkx as nx

S = make_synthetic_data(nrow=10, ncol=10, polarisation=0.22, correlation=0.5)

names1 = pd.DataFrame({
    'id': list(range(1, len(S['group']) + 1)),
    'group': S['group']
})

e1 = make_projection(S, "agent", threshold_method="target_lcc", method_value=0.99, centre=True)
e2 = make_projection(S, "symbolic", threshold_method="raw_similarity", method_value=-1, centre=False)

g1 = nx.from_pandas_edgelist(e1, source='u', target='v', edge_attr=True, create_using=nx.Graph())
for _, row in names1.iterrows():
    g1.nodes[row['id']]['group'] = row['group']

print("Nodes:", g1.nodes(data=True))
print("Edges:", g1.edges(data=True))

import matplotlib.pyplot as plt
import networkx as nx

# Assuming g1 is your graph created using NetworkX

# Create a layout using the spring layout (which is similar to Fruchterman-Reingold)
pos = nx.spring_layout(g1, seed=42)  # You can adjust the seed for reproducibility

# Plot the graph with the specified attributes
plt.figure(figsize=(8, 6))  # Adjust the figure size

# Draw the graph with specified parameters
nx.draw(g1, pos, 
        with_labels=False,  # Do not display labels
        node_size=25,       # Equivalent to vertex.size=2.5 (scaled for better visualization)
        width=0.3,          # Equivalent to edge.width
        edge_color='gray',  # Edge color
        node_color='skyblue',  # Node color
        alpha=0.6)         # Transparency for nodes

# Title of the plot
plt.title("Agent Layer")

# Show the plot
plt.show()
