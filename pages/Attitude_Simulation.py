import streamlit as st
import matplotlib.pyplot as plt
from src.Quayle2020_network_theory import AttitudeNetwork, create_sample_data, create_polarized_sample_data, create_immigration_attitude_data
import random

st.set_page_config(page_title="Attitude Expression Simulation")
st.title("🎭 Attitude Expression & Identity Group Simulation")

# Create and populate network
network = AttitudeNetwork()

# Data selection
st.sidebar.header("Choose Sample Data")
data_option = st.sidebar.selectbox("Select a dataset", ["Basic", "Polarized", "Immigration Attitudes"])

if data_option == "Basic":
    sample_data = create_sample_data()
elif data_option == "Polarized":
    sample_data = create_polarized_sample_data()
else:
    sample_data = create_immigration_attitude_data()

# Build the network
for person, attitudes in sample_data.items():
    network.add_person(person, attitudes)

# Simulate expressions
timestamps = range(10)
for t in timestamps:
    for person in network.people:
        potential = list(network.network.nodes[person]['potential_attitudes'])
        if potential:
            attitude = random.choice(potential)
            network.express_attitude(person, attitude, t)

# Display identity groups
st.subheader("👥 Identity Groups")
identity_groups = network.identify_identity_groups(min_similarity=0.3)
for i, group in enumerate(identity_groups, 1):
    st.write(f"**Group {i}:** {group}")

# Display attitude clusters
st.subheader("💬 Attitude Clusters")
attitude_clusters = network.get_attitude_clusters()
for i, cluster in enumerate(attitude_clusters, 1):
    st.write(f"**Cluster {i}:** {cluster}")

# Visualizations
st.subheader("📊 Visualizations")

with st.expander("Bipartite Network", expanded=True):
    fig1 = network.plot_bipartite_network()
    st.pyplot(fig1)

with st.expander("Agreement Network", expanded=True):
    fig2 = network.plot_agreement_network()
    st.pyplot(fig2)

with st.expander("Attitude Heatmap", expanded=True):
    fig3 = network.plot_attitude_heatmap()
    st.pyplot(fig3)

with st.expander("Expression Timeline", expanded=True):
    fig4 = network.plot_expression_timeline()
    st.pyplot(fig4)

