import numpy as np
import pandas as pd
from plotnine import *
from plotnine.themes import *
from plotnine.scales import *
from plotnine.guides import *

def plot_accuracy_compression(accuracy, compression, language, y_label=True, legend=True, output=None):
    N = len(accuracy)
    df = pd.DataFrame({
        'Collisions allowed': 2 * ([0] + list(range(1, N+1))),
        'Metric': ['Accuracy'] * (N+1) + ['Compression'] * (N+1),
        'Value': [1.0] + accuracy + [0.0] + compression,
    })

    fig = (
        ggplot(df, aes('Collisions allowed', 'Value', linetype='Metric', guide=False)) +
        geom_step(show_legend=legend) +
        ggtitle(language) +
        scale_y_continuous(breaks=[0.0, 0.05, 0.1, 0.15, 0.2, 0.95, 1.0]) +
        theme_bw(base_size=16) +
        theme(aspect_ratio=1.2) +
        ylab('Value' if y_label else '')
    )

    if output:
        fig.save(output)

    return fig
