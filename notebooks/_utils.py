import sys
import matplotlib.pyplot as plt


def plt_show():
    """Show plot interactively (Jupyter/IPython) or close silently (CLI script)."""
    if "ipykernel" in sys.modules or hasattr(sys, "ps1"):
        plt.show()
    else:
        plt.close("all")
