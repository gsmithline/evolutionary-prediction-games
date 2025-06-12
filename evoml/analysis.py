import base64
import contextlib
import functools
import io
import json
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import scipy.interpolate

__all__ = [
    'plot_classification_2d_scatter',
    'plot_pdf_flat_support',
    'plot_population_distribution',
    'plot_population_game',
    'group_colors',
    'sample_to_meshgrid',
    'meshgrid_to_2d',
    'meshgrid_to_extent',
    'classification_y_colors',
    'classification_y_cmaps',
    'background_line_style',
    'create_figure',
    'ParamTracker',
]

# Colors

classification_y_colors = ['tab:blue','tab:red']

classification_y_cmaps = [
    matplotlib.colors.LinearSegmentedColormap.from_list(
        name=f'class{i}',
        colors=[
            [1,1,1,0],
            classification_y_colors[i],
        ],
    )
    for i in [0,1]
]

background_line_style = {
    'color': 'lightgray',
    'linestyle': ':',
    'zorder': -100,
}

# Common labels

composition_tickpos = [0,0.5,1]
composition_ticklabels = [
    'Only $B$',
    '50-50',
    'Only $A$',
]
composition_label = r'Population state ($\bm{p}$)'

group_cmap = matplotlib.colormaps['PiYG']
group_colors = [
    group_cmap(0.8),
    group_cmap(0.2),
]

# Plotting

def plot_classification_2d_scatter(X,y,g=None,ax=None,**kwargs):
    n=len(X)
    block_size=10
    colors = np.array(kwargs.pop('colors',None) or ['tab:blue','tab:red','tab:green'])
    markers = np.array(kwargs.pop('markers',None) or ['o','s','^'])
    g = g if g is not None else np.zeros(len(X))
    g = g.round().astype(int)
    if ax is None:
        fig,ax = plt.subplots()
    for block in range(0,n,block_size):
        for i in np.unique(g):
            ax.scatter(
                *(X[g==i][block:block+block_size].T),
                **kwargs,
                color=colors[y[g==i][block:block+block_size].round().astype(int)],
                marker=markers[i],
            )
    return ax

def plot_pdf_flat_support(X, Y, pdf, support_level, ax, color, **kwargs):
    pdf_flat = pdf.ravel()
    pdf_argsort = np.argsort(pdf_flat)[::-1]
    argsort_cdf = (pdf_flat/pdf.sum())[pdf_argsort].cumsum()
    argsort_ind = (argsort_cdf<support_level).argmin()
    v = pdf_flat[pdf_argsort[argsort_ind]]
    return ax.contourf(
        X,
        Y,
        pdf,
        levels=[0,v,pdf.max()],
        colors=[[1,1,1,0],color,color],
        antialiased=True,
        **kwargs,
    )

def plot_population_distribution(pop, ax, *, random_state, mixture_kwargs={}, extent=None):
    markers = ['o','s','^']
    p = mixture_kwargs.pop('p',np.ones(pop.n_groups)/pop.n_groups)
    mixture_data = pop.sample_from_mixture(
        **(
            dict(
                n=200,
                p=p,
                random_state=random_state,
            ) | mixture_kwargs
        ),
    )
    plot_classification_2d_scatter(
        *mixture_data,
        ax=ax,
        alpha=0.7,
        markers=markers,
        s=0.5,
    )
    ax.axis('equal')
    ax.set(
        title='Data distribution',
        xlabel='$x_1$',
        ylabel='$x_2$',
    )
    if extent is None:
        X = pop.sample_from_uniform_mixture(n=1000, random_state=random_state)[0]
        X_grid = sample_to_meshgrid(X,200,padding=0.0)
    else:
        X_grid = np.meshgrid(
            np.linspace(extent[0], extent[1], 200),
            np.linspace(extent[2], extent[3], 200),
        )
    for i,g in enumerate(pop.groups):
        pdf = g.pdf(np.dstack(X_grid))
        for y in [0,1]:
            plot_pdf_flat_support(
                *X_grid,
                pdf=pdf[y],
                support_level=0.9,
                ax=ax,
                color=(
                    classification_y_colors[y],
                    0.3*(p[i]/max(p)),
                ),
                # alpha=0.5,
                zorder=-1,
            )
    return ax

def plot_2d_clf(pop, clf, random_state, **kwargs):
    return sklearn.inspection.DecisionBoundaryDisplay.from_estimator(
        estimator=clf,
        X=pop.sample_from_mixture(n=1000,p=[0.5,0.5],random_state=random_state)[0],
        **kwargs,
    )

def plot_population_game(game_df, ax, *, legend=False, equilibria={}, equilibria_kwargs={}, plot_kwargs={}, legend_kwargs={}, ax_kwargs={}, annotations={}, annotation_kwargs={}):
    if 'rep' in game_df:
        game_df = game_df[['acc_A','acc_B','acc_p']].groupby(level=0).mean()
    (
        game_df
        [['acc_A','acc_B']]
        .plot.line(
            ax=ax,
            color=group_colors,
            legend=legend,
            **plot_kwargs,
        )
    )
    (
        game_df
        ['acc_p']
        .plot.line(
            ax=ax,
            color='tab:cyan',
            zorder=-1,
            legend=legend,
            linestyle=':',
        )
    )
    if legend:
        ax.legend(
            [
                r'$\mathrm{acc}_A$',
                r'$\mathrm{acc}_B$',
                r'$\mathrm{acc}_\vp$',
            ],
            **legend_kwargs,
        )
    ax.set(
        # title='Prediction game',
        xlabel=composition_label,
        xticks=composition_tickpos,
        xticklabels=composition_ticklabels[::-1],
        ylabel='Accuracy',
        **ax_kwargs,
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1,decimals=1))
    if equilibria:
        plot_population_game_xphase(
            ax=ax,
            equilibria=equilibria,
            **equilibria_kwargs,
        )
        for x,is_stable in equilibria:
            y = scipy.interpolate.interp1d(*game_df['acc_p'].reset_index().to_numpy().T)(x).item()
            color = 'tab:blue'
            ax.plot(
                x,y,'o',
                color=color if is_stable else 'white',
                markeredgecolor=color,
                markersize=3.5,
            )
    if annotations:
        annotation_cfg = {
            'acc_A': dict(
                color=group_cmap(1.0),
                text='$\\mathrm{acc}_A$',
            ),
            'acc_B': dict(
                color=group_cmap(0.0),
                text='$\\mathrm{acc}_B$',
            ),
            'acc_p': dict(
                color='tab:blue',
                text='$\\mathrm{acc}_{\\bm{p}}$',
            ),
        }
        for metric, xy in annotations.items():
            ax.annotate(
                xy=xy,
                **annotation_cfg[metric],
                **annotation_kwargs,
            )
    return ax

@contextlib.contextmanager
def autoscale_turned_off(ax=None):
  ax = ax or plt.gca()
  lims = [ax.get_xlim(), ax.get_ylim()]
  yield
  ax.set_xlim(*lims[0])
  ax.set_ylim(*lims[1])

def plot_population_game_xphase(ax, equilibria, d=0.05, markersize=3.5, additional_markers={}):
    with autoscale_turned_off(ax=ax):
        y = ax.get_ylim()[0]
        for i,(x,is_stable) in enumerate(equilibria):
            ax.plot(
                x,y,
                'o',
                clip_on=False,
                zorder=100,
                color='black' if is_stable else 'white',
                markeredgecolor='black',
                markersize=markersize,
            )
            marker_points = [x-d,x+d]
            markers = (
                {x_marker: ['>','<'][is_stable ^ (x_marker<x)] for x_marker in marker_points}
                | additional_markers
            )
            for x_marker, symbol in markers.items():
                if x_marker>=1 or x_marker<=0:
                    continue
                ax.plot(
                    x_marker,y,
                    marker=symbol,
                    clip_on=False,
                    zorder=100,
                    color='black',
                    # color='white' if abs(x-0.5)<0.2 else 'black',
                    # markeredgecolor='black',
                    markersize=markersize,
                )
    return ax

# Meshgrid utilities

def sample_to_meshgrid(X, n_grid_points=50, padding=0.2):
    xmax = np.max(X,axis=0)
    xmin = np.min(X,axis=0)
    d = np.abs(xmax-xmin)
    assert len(d)==2
    pad = d*padding
    linspaces = [
        np.linspace(xmin[i]-pad[i],xmax[i]+pad[i],num=n_grid_points)
        for i in [0,1]
    ]
    return np.meshgrid(*linspaces)


def meshgrid_to_2d(X_grid):
    return np.array(X_grid).reshape((2,-1),order='F').T

def meshgrid_to_extent(*grid):
    return [grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()]

simplex_projection = np.array([
    [0,1/np.sqrt(3),-1/np.sqrt(3)],
    [1,0,0],
])

# Figures

class DownloadableIO(io.BytesIO):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._download_filename = filename

    def _repr_html_(self):
        buf = self.getbuffer()
        buf_enc = base64.b64encode(buf).decode('ascii')
        return f'<a href="data:text/plain;base64,{buf_enc}" download="{self._download_filename}">Download {self._download_filename}</a>'

def save_fig(fig, fname, **savefig_kwargs):
    return fig.savefig(
        fname=fname,
        # bbox_inches='tight',
        # pad_inches=0,
         **savefig_kwargs,
    )

def download_fig(fig, fname, **savefig_kwargs):
    fig_out = DownloadableIO(filename=os.path.basename(fname))
    save_fig(
        fig,
        fig_out,
        format=fname.split('.')[-1],
        **savefig_kwargs,
    )
    display(fig_out)

def save_and_download_fig(fig, fname, **savefig_kwargs):
    save_fig(fig, fname, **savefig_kwargs)
    print(f'Figure saved as {fname}')
    download_fig(fig, fname, **savefig_kwargs)

@functools.wraps(plt.subplots)
def create_figure(*args, **kwargs):
    kwargs['figsize'] = kwargs.pop('figsize',(10,3))
    kwargs['constrained_layout'] = kwargs.pop('constrained_layout',{'w_pad':0.15})
    # kwargs['layout'] = kwargs.pop('layout','constrained')
    fig, ax = plt.subplots(*args, **kwargs)
    fig.download = lambda filename: download_fig(fig, filename)
    fig.save_and_download = lambda filename: save_and_download_fig(fig, filename)
    return fig, ax


# Param tracker

class NpEncoder(json.JSONEncoder):
    # https://stackoverflow.com/questions/50916422
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class ParamTracker:
    def __init__(self):
        self.data = {}
        self.data_str = {}

    def store(self, x, param_name, encoding=str):
        if type(encoding) is str:
            encoding = encoding.format
        if param_name in self.data and x!=self.data[param_name]:
            warnings.warn(
                'Incosistent parameter value. '
                f'existing: {param_name}={self.data[param_name]!r}, '
                f'new: {param_name}={x!r}.'
            )
        self.data[param_name] = x
        self.data_str[param_name] = encoding(x)
        print(f'{param_name}={encoding(x)}')
        return x

    def get(self, param_name):
        return self.data[param_name]

    def save(self, fname, strings=True):
        json.dump(
            obj=self.data_str if strings else self.data,
            fp=open(fname,'w'),
            cls=NpEncoder,
        )
        print(f'Parameters saved as: {fname}')
