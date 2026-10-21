"""Render manuscript equations with mathtext, without importing training code."""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import mathtext, rcParams


def normalized_latex(latex):
    # Standard TeX accepts an unbraced single-letter argument after whitespace;
    # mathtext requires braces for these commands. This preserves the formula.
    return re.sub(r"\\(mathcal|mathbb|mathbf|mathrm|overline|widehat)\s+([A-Za-z])",
                  lambda match: "\\"+match.group(1)+"{"+match.group(2)+"}",
                  latex.strip().strip("$"))


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral",
                     "font.size": 12})
    for equation in request:
        path = Path(equation["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        latex = normalized_latex(equation["latex"])
        mathtext.math_to_image("$" + latex + "$", path, dpi=260, format="png", color="black")
    print(json.dumps({"rendered_equations": len(request)}))


if __name__ == "__main__":
    main()
