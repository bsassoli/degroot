""" simulates degroot model in streamlit
"""
import numpy as np
import matplotlib
import streamlit as st
import networkx as nx
from utils import display_latex
from deGroot import DeGroot


def main():
    """runs simulation"""

    st.title("DeGroot Simulator (beta)")

    belief_vector = np.array([1, 0, 0])
    t_matrix = np.array([[0, 0.5, 0.5], [1, 0, 0], [0, 1, 0]])
    model = DeGroot(belief_vector, t_matrix)
    msg_container = st.empty()

    col1, col2, col3 = st.columns((0.33, 0.33, 0.33))
    conv, hist = model.iterate()
    msg = (
        f"Convergence reached at step {len(hist)}"
        if conv
        else "Model does not convergence"
    )

    with msg_container:
        st.write(msg)

    def simulate():
        return DeGroot(belief_vector, t_matrix)

    st.sidebar.header("Choose")
    step = st.sidebar.slider("Simulation step", 0, len(hist) - 1, 0)
    st.sidebar.button("Reset model", on_click=simulate)
    with col1:
        st.subheader("Graph")
        graph = nx.from_numpy_array(model.trust, create_using=nx.MultiDiGraph())
        g_labels = model.trust
        edges = graph.edges(data=True)
        for i, j, edge in edges:
            edge["label"] = g_labels[i][j]
        dot = nx.nx_pydot.to_pydot(graph)
        for i, node in enumerate(dot.get_nodes()):
            node.set_label(f"{hist[step][i]:.3f}")
            node.set_style("filled")
            alpha = round(hist[step][i], 2)
            color = matplotlib.colors.to_hex([0, 0.39, 0.38, alpha], keep_alpha=True)
            node.set_fillcolor(color)
            font_color = alpha
            node.set_fontcolor(matplotlib.colors.to_hex([font_color] * 3))
        st.graphviz_chart(dot.to_string())

    with col2:
        st.subheader("Belief vector")
        st.latex(display_latex(model.trust, hist[step])[0])

    with col3:
        st.subheader("Trust matrix")
        st.latex(display_latex(model.trust, hist[step])[1])
        # st.write(msg)


if __name__ == "__main__":
    main()
