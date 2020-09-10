import numpy as np
import pandas as pd
from plotnine import *
from plotnine.themes import *
from plotnine.scales import *
from plotnine.guides import *
from plotnine.positions import *
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import sklearn

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
        scale_y_continuous(breaks=[0.0, 0.05, 0.1, 0.15, 0.2, 0.90, 0.95, 1.0]) +
        theme_bw(base_size=16) +
        theme(aspect_ratio=1.2) +
        ylab('Value' if y_label else '')
    )

    if output:
        fig.save(output)

    return fig

def plot_accuracy_distributions_fine_tuning(original, before, after, output=None):
    df = pd.DataFrame({
        'Accuracy': original + before + after,
        'Stage': ['1- Original'] * len(original) +
                 ['2- Original + Updated table'] * len(before) +
                 ['3- Original + Updated table + Fine-tuning'] * len(after),
    })

    fig = (
        ggplot(df, aes('Accuracy')) +
        geom_histogram(binwidth=0.01) +
        facet_wrap('Stage', nrow=3, ncol=1, dir='v') +
        scale_x_continuous(breaks=np.arange(0, 1.05, 0.05)) +
        ylab('Frequency') +
        theme_bw()
    )

    if output:
        fig.save(output)

    return fig

def plot_compression_distributions_fine_tuning(before, after, output=None):
    df_before = pd.DataFrame({ 'Compression': before })
    df_after = pd.DataFrame({ 'Compression': after })

    df = pd.DataFrame({
        'Compression': before + after,
        'Stage': ['1- Before fine-tuning'] * len(before) +
                 ['2- After fine-tuning'] * len(after)
    })

    fig = (
        ggplot() +
        geom_histogram(df_before, aes('Compression'), fill='#ffffff', color='#000000', binwidth=0.01) +
        geom_histogram(df_after, aes('Compression'), fill='#aaaaaa', alpha=0.5, binwidth=0.01) +
#        geom_histogram(df_after, aes('Compression'), fill='#555555', binwidth=0.01) +
        #        facet_wrap('Stage', nrow=2, ncol=1, dir='v', as_table=True) +
        # scale_x_continuous(breaks=np.arange(0, 1.05, 0.05)) +
        ylab('Frequency') +
        theme_bw() +
        theme(aspect_ratio=0.3)
    )

    if output:
        fig.save(output)

    return fig

def plot_accuracy_compression_combined(points, languages, output=None):
    for l in points:
        l.insert(0, (1.0, 0.0))

    df = pd.DataFrame({
        'Accuracy': [p[0] for lp in points for p in lp],
        'Compression': [p[1] for lp in points for p in lp],
        'Language': [languages[i] for i, lp in enumerate(points) for p in lp],
        'Collisions': [j for lp in points for j, p in enumerate(lp)],
    })

    fig = (
        ggplot(df) +
        geom_line(aes('Compression', 'Accuracy', linetype='Language')) +
        geom_point(aes('Compression', 'Accuracy')) +
        theme_bw() +
        theme(aspect_ratio=0.5)
    )

    return fig

def plot_user_study_metric(title, settings, values, error_lo, error_hi, output=None):
    df = pd.DataFrame({
        'Autocomplete': settings,
        'Mean': values,
        'ErrorLowerBound': error_lo,
        'ErrorUpperBound': error_hi,
    })

    fig = (
        ggplot(df) +
        geom_col(aes(x='Autocomplete', y='Mean')) +
        ylab(title) +
        geom_errorbar(aes(x='Autocomplete',
                          ymin='ErrorLowerBound', ymax='ErrorUpperBound')) +
        theme_bw()
    )

    if output:
        fig.save(output)

    return fig

def compute_regression_line(s):
    m = sklearn.linear_model.LinearRegression()
    X, y = np.array([[x] for x in range(len(s))]), np.array(s)
    m.fit(X, y)
    return (m.coef_[0], m.intercept_, m.score(X, y))

def plot_user_study_typing_speed(dataset, output=None):
    ts = dataset[['NormalizedTypingSpeed', 'Autocomplete', 'Snippet']]
    ts = ts[ts['Autocomplete'].map(lambda s: s in ('VSCode', 'VSCode+Pragmatic', 'Pragmatic'))]
    ts['Snippet'] = ts['Snippet'] + 1 # 1-indexing for display.
    means = ts.groupby(['Autocomplete', 'Snippet']).mean('NormalizedTypingSpeed').reset_index()
    errors = ts.groupby(['Autocomplete', 'Snippet']).agg(
            lambda s: bs.bootstrap(s.to_numpy(), alpha=0.05, stat_func=bs_stats.mean).error_width() / 2)

    means['ErrorWidth'] = errors.reset_index()['NormalizedTypingSpeed']

    rlines = pd.DataFrame(
            means
            .groupby('Autocomplete')
            .agg(compute_regression_line)
            )

    rlines[['A', 'B', 'R2']] = pd.DataFrame(rlines['NormalizedTypingSpeed'].to_list(),
                                            index=rlines.index)
    rlines = rlines.reset_index()[['Autocomplete', 'A', 'B', 'R2']] # .set_index('Autocomplete')
    print('R^2:', rlines[['Autocomplete', 'R2']])

    fig = (
        ggplot(means) +
        geom_point(data=means, mapping=aes(x='Snippet', y='NormalizedTypingSpeed')) +
        geom_errorbar(data=means,
                      mapping=aes(x='Snippet', 
                                  ymin='NormalizedTypingSpeed-ErrorWidth',
                                  ymax='NormalizedTypingSpeed+ErrorWidth')) +
        geom_line(data=means, mapping=aes(x='Snippet', y='NormalizedTypingSpeed')) +
        geom_abline(data=rlines, mapping=aes(slope='A', intercept='B'),
                    linetype='dashed') +
        ylab('Normalized Typing Speed') +
        facet_wrap('Autocomplete') +
        xlab('Snippet') +
        theme_bw() +
        theme(aspect_ratio=0.75)
    )

    if output:
        fig.save(output)

    return fig

def plot_user_study_keystrokes(dataset, output=None):
    ks = dataset[['NormalizedKeystrokes', 'Autocomplete', 'Snippet']]

    ks = ks[ks['Autocomplete'].map(lambda s: s in ('VSCode', 'VSCode+Pragmatic', 'Pragmatic'))]
    ks['Snippet'] = ks['Snippet'] + 1 # 1-indexing for display.


    ks_last_4 = ks[(2 <= ks['Snippet']) & (ks['Snippet'] <= 3)]
    ks_last_3 = ks[(3 <= ks['Snippet']) & (ks['Snippet'] <= 4)]
    ks_last_2 = ks[(4 <= ks['Snippet']) & (ks['Snippet'] <= 5)]

    compute_mean_error = lambda ks: (
            ks.groupby(['Autocomplete'])
              .mean('NormalizedKeystrokes')
              .reset_index(),
            ks.groupby(['Autocomplete']).agg(
                    lambda s: bs.bootstrap(s.to_numpy(), alpha=0.05, stat_func=bs_stats.mean).error_width() / 2).reset_index()['NormalizedKeystrokes']
            )

    means_last_4, error_last_4 = compute_mean_error(ks_last_4)
    means_last_3, error_last_3 = compute_mean_error(ks_last_3)
    means_last_2, error_last_2 = compute_mean_error(ks_last_2)

    means_last_4['Dropped'] = 1
    means_last_4['ErrorWidth'] = error_last_4
    means_last_3['Dropped'] = 2
    means_last_3['ErrorWidth'] = error_last_3
    means_last_2['Dropped'] = 3
    means_last_2['ErrorWidth'] = error_last_2

    ds = pd.concat([means_last_4, means_last_3, means_last_2])

    fig = (
        ggplot(aes(x='Dropped', y='NormalizedKeystrokes')) +
        geom_col(data=ds, mapping=aes(fill='Autocomplete'), position='dodge2') +
        geom_errorbar(data=ds, 
                      position=position_dodge2(padding=0.5),
                      width=0.9,
                      mapping=aes(ymin='NormalizedKeystrokes-ErrorWidth',
                                  ymax='NormalizedKeystrokes+ErrorWidth')) +
        xlab('') +
        ylab('Normalized keystrokes') +
        scale_fill_gray(0.4, 0.8) +
        coord_cartesian(ylim=(0.75, 1.02)) +
        theme_bw() +
        theme(aspect_ratio=0.3)
    )

    if output:
        fig.save(output)

    return fig
