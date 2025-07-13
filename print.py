import subprocess
from typing import List
from synthesis import Gate, Sigma1, Sigma2, TGate, FGate, WIGate
import os


def generate_braid_tikz(operations: List[Gate]):
  latex_code = r"""
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{braids}
\begin{document}

\usepgfmodule{nonlineartransformations}
\makeatletter
\def\polartransformation{%
% \pgf@x will contain the radius
% \pgf@y will contain the distance
\pgfmathsincos@{\pgf@sys@tonumber\pgf@x}%
% pgfmathresultx is now the cosine of radius and
% pgfmathresulty is the sine of radius
\pgf@x=\pgfmathresultx\pgf@y%
\pgf@y=\pgfmathresulty\pgf@y%
}
\makeatother

\begin{center}
\begin{tikzpicture}
\begin{scope}
\pgftransformnonlinear{\polartransformation}
\pic[
rotate=90,
braid/.cd,
every strand/.style={line width=6pt,magenta!67},
border height=0cm,
crossing height=15.7pt,
gap=.2,
nudge factor=0.01,
] at (0,5) {braid={"""

  braid_operations = []
  for operation in operations:
    if isinstance(operation, Sigma1):
      print("I")
      braid_operations.append("s_1")
    elif isinstance(operation, Sigma2):
      braid_operations.append("s_2")
    else:
      print(f"Unknown operation: {operation}")

  latex_code += " ".join(braid_operations)

  latex_code += r"""
}}; 
\end{scope}
\end{tikzpicture}
\end{center}
\end{document}
"""

  with open("latex/braid_diagram.tex", "w") as f:
    f.write(latex_code)

  os.chdir("latex")
  subprocess.run(["pdflatex", "braid_diagram.tex"])
  os.chdir("..")


import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np


def draw_braid(word, num_strands=3, spacing=1.0, height=1.0):
  """
  Draws a braid diagram for a given braid word using matplotlib.

  Args:
      word (list of int): A list of braid generators (+i for σᵢ, -i for σᵢ⁻¹).
      num_strands (int): Number of strands in the braid.
      spacing (float): Horizontal spacing between strands.
      height (float): Vertical space per crossing.
  """

  def strand_x(i):
    return i * spacing

  fig, ax = plt.subplots(figsize=(num_strands, len(word) * 0.8))
  ax.set_aspect("equal")
  ax.axis("off")

  # Initialize the strands: list of (x, y) points for each logical strand
  strand_paths = [[] for _ in range(num_strands)]
  positions = list(range(num_strands))  # physical position of each logical strand

  y = 0.0
  for step, gen in enumerate(word):
    y_next = y + height
    i = abs(gen) - 1

    # Add vertical segments to each strand
    for j in range(num_strands):
      strand = positions[j]
      x = strand_x(j)
      strand_paths[strand].append((x, y))
      strand_paths[strand].append((x, y_next))

    # Determine over and under strands
    if gen > 0:
      over_idx, under_idx = i, i + 1
    else:
      over_idx, under_idx = i + 1, i

    over_strand = positions[over_idx]
    under_strand = positions[under_idx]

    # Over arc (Bezier-like)
    x1, x2 = strand_x(over_idx), strand_x(under_idx)
    xm = (x1 + x2) / 2
    yc = y + height / 2

    ax.plot([x1, xm, x2], [y + 0.1 * height, yc, y_next - 0.1 * height], color="black", linewidth=2.5, zorder=2)

    # Hide middle part of under strand
    ax.plot([x2, x2], [y, y_next], color="white", linewidth=3.1, zorder=3)

    # Redraw top and bottom part of under strand to simulate passing under
    ax.plot([x2, x2], [y, y + 0.1 * height], color="black", linewidth=2, zorder=1)
    ax.plot([x2, x2], [y_next - 0.1 * height, y_next], color="black", linewidth=2, zorder=1)

    # Swap strands in the physical position tracker
    positions[i], positions[i + 1] = positions[i + 1], positions[i]

    y = y_next

  # Final segments down to bottom
  y_end = y + height
  for j in range(num_strands):
    strand = positions[j]
    x = strand_x(j)
    strand_paths[strand].append((x, y))
    strand_paths[strand].append((x, y_end))

  # Draw strands
  for path in strand_paths:
    xs, ys = zip(*path)
    ax.plot(xs, ys, color="black", linewidth=2, zorder=0)

  ax.set_xlim(-spacing, (num_strands - 1) * spacing + spacing)
  ax.set_ylim(y_end + height * 0.2, -height * 0.5)
  plt.show()


def plot_braid(braid, num_strands=3):
  fig, ax = plt.subplots(figsize=(2 * len(braid), 4))

  n = len(braid)
  x_spacing = 1
  y_spacing = 1
  strand_positions = np.arange(num_strands)

  # Track the y positions for each braid step
  for i, gen in enumerate(braid):
    x0 = i * x_spacing
    x1 = (i + 1) * x_spacing
    y0 = -i * y_spacing
    y1 = -(i + 1) * y_spacing

    # By default, strands just go straight down
    for s in range(num_strands):
      ax.plot([x0, x1], [y0, y1], color="black", zorder=1)

    if gen == "σ1":
      draw_cross(ax, x0, x1, y0, y1, 0, over=True)
    elif gen == "σ1⁻¹":
      draw_cross(ax, x0, x1, y0, y1, 0, over=False)
    elif gen == "σ2":
      draw_cross(ax, x0, x1, y0, y1, 1, over=True)
    elif gen == "σ2⁻¹":
      draw_cross(ax, x0, x1, y0, y1, 1, over=False)

  ax.axis("equal")
  ax.axis("off")
  plt.show()


def draw_cross(ax, x0, x1, y0, y1, index, over=True):
  # Crossing between strand[index] and strand[index+1]
  midx = (x0 + x1) / 2
  midy = (y0 + y1) / 2

  xs = [x0, x1]
  # Draw under-strand first
  if over:
    # Under strand
    ax.plot(xs, [y0, y1], color="gray", linewidth=2, zorder=1)
    ax.plot(xs, [y0, y1], color="black", linestyle=":", zorder=1)
    # Over strand on top
    ax.plot(xs, [y1, y0], color="black", linewidth=2, zorder=2)
  else:
    # Over strand
    ax.plot(xs, [y1, y0], color="gray", linewidth=2, zorder=1)
    ax.plot(xs, [y1, y0], color="black", linestyle=":", zorder=1)
    # Under strand on top
    ax.plot(xs, [y0, y1], color="black", linewidth=2, zorder=2)


# Example usage:


if __name__ == "__main__":
  braid = ["σ1", "σ2", "σ1⁻¹", "σ2⁻¹", "σ1"]
  plot_braid(braid, num_strands=3)


# if __name__ == "__main__":
def test():
  operations = [Sigma2(), Sigma2(), Sigma1()]
  operations = [
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=10),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=10),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=18),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=8),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=12),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=16),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=10),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=16),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=12),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=8),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=18),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=10),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=10),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=10),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=10),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=10),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=2),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=12),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=8),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=4),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=10),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=4),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=8),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=12),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=2),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=10),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=10),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma2(),
    Sigma1(),
    WIGate(power=4),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
    Sigma1(),
  ]
  generate_braid_tikz(operations[:30])
