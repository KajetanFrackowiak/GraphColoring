import numpy as np
import networkx as nx
import plotly.graph_objects as go


def generate_chart(G: nx.Graph, coloring_result):
    pos = {i: np.random.rand(3) for i in G.nodes()}
    edge_x, edge_y, edge_z = [], [], []


    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        line=dict(width=2, color="gray"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, node_z, node_colors = [], [], [], []
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_colors.append(coloring_result[node])

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        marker=dict(size=8, color=node_colors, colorscale="Jet", opacity=0.8),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="3D Interactive Graph Coloring (Sampling Algorithm)",
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )
    fig.write_html("chart.html")
