from fractions import Fraction as fract  # format decimals as fractions
from typing import Tuple

LIMIT_DENOMINATOR = 100000


def display_latex(t, b) -> Tuple[str, str]:
    trust = [list(map(lambda n: fract(n).limit_denominator(LIMIT_DENOMINATOR), row)) for row in t]
    fracts = list(map(lambda n: fract(n).limit_denominator(LIMIT_DENOMINATOR), b))
    latex_beliefs = r"\begin{pmatrix}"
    latex_beliefs += "".join(list(map(lambda n: f"{n}" + r"\\", fracts)))
    latex_trust = r"\begin{pmatrix}"
    for row in trust:
        latex_trust += "".join(list(map(lambda n: f"{n}&", row)))
        latex_trust += r"\\"
    latex_trust += r"\end{pmatrix}"
    latex_beliefs += r"\end{pmatrix}"
    return (latex_beliefs, latex_trust)
