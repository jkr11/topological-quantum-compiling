import subprocess
from typing import List
from synthesis import Gate, Sigma1, Sigma2
import os


def generate_braid_tikz(operations: List[Gate]):
  latex_code = """
\\documentclass{article}
\\usepackage{tikz}
\\usetikzlibrary{braids}
\\begin{document}
\\begin{center}
\\begin{tikzpicture}
\\pic[
rotate=90,
braid/.cd,
every strand/.style={ultra thick},
strand 1/.style={red},
strand 2/.style={green},
strand 3/.style={blue},  % You can increase strands if needed
] {braid={"""

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

  latex_code += """
}}; 
\\end{tikzpicture}
\\end{center}
\\end{document}
"""

  with open("latex/braid_diagram.tex", "w") as f:
    f.write(latex_code)

  os.chdir("latex")
  subprocess.run(["pdflatex", "braid_diagram.tex"])
  os.chdir("..")


operations = [Sigma2(), Sigma2(), Sigma1()]
generate_braid_tikz(operations)
