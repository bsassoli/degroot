import numpy as np
from deGroot import DeGroot

# belief_vector = np.array([1, 0, 0])
# t_matrix = np.array([[0, 0.5, 0.5], [1, 0, 0], [0, 1, 0]])
# jackson_example = DeGroot(belief_vector, t_matrix)
belief_vector = np.array([1, 0, 0])
t_matrix = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [1, 0, 0]])
model = DeGroot(belief_vector, t_matrix)
# hist = model.iterate()
for _ in range(10):
    model._time_step()
    print(model.beliefs)
