from utils import display_latex
from deGroot import DeGroot
import numpy as np
import streamlit as st
import networkx as nx


def main():
    st.title("DeGroot Simulator (beta)")

    belief_vector = np.array([1, 0, 0])
    t_matrix = np.array([[0, 0.5, 0.5], [1, 0, 0], [0, 1, 0]])
    model = DeGroot(belief_vector, t_matrix)

    col1, col2, col3 = st.columns((0.33, 0.33, 0.33))
    with col1:
        st.subheader("Graph")
        G = nx.from_numpy_array(model._trust, create_using=nx.MultiDiGraph())
        G_labels = model._trust
        edges = G.edges(data=True)
        for i, j, d in edges:
            d["label"] = G_labels[i][j]
        node_attrs = {n: {"color": "blue"} for n in range(len(model.beliefs))}
        dot = nx.nx_pydot.to_pydot(G, node_attr=node_attrs)
        print(dot)
        st.graphviz_chart(dot.to_string())
    conv, hist = model.iterate()
    msg = "Convergence reached" if conv else "No convergence"

    def simulate():
        return DeGroot(belief_vector, t_matrix)

    st.sidebar.header("Choose")
    step = st.sidebar.slider("Simulation step", 0, len(hist)-1, 0)
    st.sidebar.button("Reset model", on_click=simulate)

    with col2:
            st.subheader("Belief vector")
            st.latex(display_latex(model._trust, hist[step])[0])
    with col3:
            st.subheader("Trust matrix")
            st.latex(display_latex(model._trust, hist[step])[1])
            # st.write(msg)


if __name__ == "__main__":
    main()
