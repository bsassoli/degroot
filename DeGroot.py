from typing import Tuple
from fractions import Fraction as fract  # format decimals as fractions
from sys import float_info  # default value for testing convergence
from numpy.typing import NDArray
import numpy as np

LIMIT_DENOMINATOR = 100000


class DeGroot:
    """
    A class implementing DeGroot social learning model

    Args:
        beliefs(numpy array): a vector representing the initial beliefs
        of the population
        trust_matrix(numpy array): the model's transitions matrix

    Attributes:
        beliefs(numpy array): The current state of the agents beliefs
    """

    def __init__(
        self, beliefs: NDArray[np.float64], trust_matrix: NDArray[np.float64]
    ) -> None:
        """
        Initializes object after checking that inputs are correct.
        """

        if len(beliefs) != len(trust_matrix):
            error_msg = f"incompatible dimensions for beliefs\
                vector and trust matrix.\n\
                Belief vector has shape: {beliefs.shape} and the trust matrix\
                has shape {trust_matrix.shape}"
            raise ValueError(error_msg)
        if trust_matrix.shape[0] != trust_matrix.shape[1]:
            error_msg = f"trust matrix must have shape: \
                {len(beliefs)} x {len(beliefs)}.\n \
                Received shape {trust_matrix.shape}"
            raise ValueError(error_msg)
        if np.any([0 >= belief >= 1 for belief in beliefs]):
            error_msg = "sum of beliefs vector is not equal to 1"
            raise ValueError(error_msg)
        if np.any([sum(row) != 1 for row in trust_matrix]):
            raise ValueError(
                "all rows of the trust matrix should \
                             have a sum of 1"
            )

        self.beliefs = beliefs.T
        self._trust = trust_matrix

    def __str__(self):
        fracts = list(
            map(
                lambda n: str(fract(n).limit_denominator(LIMIT_DENOMINATOR)),
                self.beliefs.tolist(),
            )
        )
        # fracts = "\n".join([f"{n.numerator}/{n.denominator}" for n in fracts])
        return " ".join(fracts)

    def _time_step(self) -> NDArray[np.float64]:
        # updates the model
        self.beliefs = self.beliefs @ self._trust
        return self.beliefs

    def iterate(
        self,
        no_iters: int = 1000,
        tolerance_level: float = float_info.epsilon * 10**3,
    ) -> Tuple[bool, list[NDArray[np.float64]]]:
        """Models Markov chain"""
        if no_iters < 1:
            raise ValueError(
                f"number of iterations must be 1 or greater. Got {no_iters}"
            )
        history = []
        for i in range(no_iters):
            old_vector = self.beliefs
            history.append(old_vector)
            new_vector = self._time_step()
            if np.all(abs(new_vector - old_vector) < tolerance_level):
                print(f"Convergence reached at the {i}th time_step\n")
                return (True, history)
        print("No Convergence\n")
        return (False, history)
