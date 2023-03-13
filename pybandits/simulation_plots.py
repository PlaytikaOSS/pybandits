# MIT License
#
# Copyright (c) 2022 Playtika Ltd.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
import pandas as pd
from plotnine import aes, geom_line, ggplot, ylab


def plot_cumulative_proportions(matrix, figsize=None, path="", title=""):
    """Plot simulation."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(matrix)
    ax.set_title(title)
    ax.set_xlabel("number of observations")
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Cumulative proportion")
    ax.legend(matrix.columns)
    fig.savefig(path + title)

    return fig, ax


def plot_regrets(cum_regret):
    """Plot cumulative regrets."""
    regrets = pd.Series(cum_regret).reset_index().rename(columns={"index": "number of observations"})
    return ggplot(regrets, aes(y="cum_regret", x="number of observations")) + geom_line() + ylab("cumulative regret")
