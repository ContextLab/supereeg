

import seaborn as sb
import pandas as pd
from nilearn import plotting as ni_plt
from loadnii import BrainData, loadnii
import numpy as np
from matplotlib import pyplot as plt
#plt.switch_backend('agg')
from scipy.spatial.distance import squareform as squareform
import os

def compare_matrices(x, y, outfile=None, titles=(None, None)):
    """
    Compares subject's inter-electrode correlation matrix with their full correlation matrix, indexed by their electrode locations - this  is a sanity check- they should be the same with a radius of 1

    Parameters
    ----------
    x : ndarray
        Subject's correlation matrix

    y : array
        Full correlation matrix by subject's electrodes

    outfile : string
        File name

    titles : strings
        Plot titles for x and y

    Returns
    ----------
    results : plotted matrices
         If outfile in arguments, the plot is saved.  Otherwise, it the plot is shown.

    """
    plt.subplot(1, 2, 1)
    sb.heatmap(x)

    plt.subplot(1, 2, 2)
    sb.heatmap(y)

    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()


def reverse_squareform(Matrix_sqfm):
    """
    Expands squareform matrix

    Parameters
    ----------
    Matrix_half : ndarray
        Squareform matrix

    Returns
    ----------
    Matrix_full : ndarray
         If outfile in arguments, the plot is saved.  Otherwise, it the plot is shown.

    """
    Matrix_full = squareform(Matrix_sqfm, checks=False)
    Matrix_full = Matrix_full + np.eye(np.shape(Matrix_full)[0])
    return Matrix_full

def plot_cov(x, outfile=None, title=None):
    """
    Plot full correlation matrix

    Parameters
    ----------
    x : ndarray
        Full correlation matrix


    outfile : string
        File name

    title : strings
        Plot titles for x and y

    Returns
    ----------
    results : plotted matrices
         If outfile in arguments, the plot is saved.  Otherwise, it the plot is shown.

    """
    sb.set(font_scale=1.5)
    ax = sb.heatmap(x, annot_kws={"size": 16})
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if not title is None:
        plt.title(title)
    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()


def plotter(plotfunc, niifile=None, idx=0, mask_strategy='background', colorbar=True, *args):
    """
    Plotting function for nilearn plots

    Parameters
    ----------
    plotfunc : nifti file or Braindata
        Full correlation matrix

    niifile : string
        Nifti file name

    Returns
    ----------
    results : brainplots

    """
    if type(niifile) is str:
        data = loadnii(niifile, mask_strategy)
    elif type(niifile) is BrainData:
        data = niifile
    else:
        data = niifile

    img = data.mask.inverse_transform(np.mean(np.array(data.Y[idx, :], ndmin=2), axis=0))
    plotfunc(img, *args)
    plt.show()


def plot_glass_brain(niifile=None, idx=0, mask_strategy='background', colorbar=True, *args):
    plotter(ni_plt.plot_glass_brain, niifile, idx, mask_strategy, colorbar, *args)


def plot_anat(niifile=None, idx=0, mask_strategy='background', **kwargs):
    plotter(ni_plt.plot_anat, niifile, idx, mask_strategy, **kwargs)


def plot_epi(niifile=None, idx=0, mask_strategy='background', **kwargs):
    plotter(ni_plt.plot_epi, niifile, idx, mask_strategy, **kwargs)


def plot_stat_map(niifile=None, idx=0, mask_strategy='background', **kwargs):
    plotter(ni_plt.plot_stat_map, niifile, idx, mask_strategy, **kwargs)


def brain_net_viewer_format(n_by_3_locs, subj_inds, electrode_ind):
    """
    This function returns a 6 column matirx for brain net viewer: first 3 columns are x, y, z coordinates, 4th coloring, 5th size of node

    Parameters
    ----------
    n_by_3_locs : ndarray
        Full list of electrode locations

    sub_inds : list
        Subject's electrodes indices

    electrode_ind : int
        Electrode index

    Returns
    ----------
    results : ndarray
        6 column matirx for brain net viewer: first 3 columns are x, y, z coordinates, 4th coloring, 5th size of node

    """
    R_add_cols = np.append(n_by_3_locs, np.zeros(np.shape(n_by_3_locs)), 1)
    R_add_cols[subj_inds, 3] = 1
    R_add_cols[electrode_ind, 3] = 2
    R_add_cols[subj_inds, 4] = 1
    R_add_cols[electrode_ind, 4] = 2
    return np.round(R_add_cols, 2)


def brain_net_viewer_format_color(n_by_3_locs, color_by):
    """
    This function returns a 6 column matirx for brain net viewer: first 3 are x, y, z coordinates, and 4th coloring

    Parameters
    ----------
    n_by_3_locs : ndarray
        Full list of electrode locations

    color_by : list
        What the electrodes should be colored by

    Returns
    ----------
    results : ndarray
        6 column matirx (first 3 are x, y, z coordinates) and 4th coloring

    """
    R_add_cols = np.append(n_by_3_locs, np.zeros(np.shape(n_by_3_locs)), 1)
    R_add_cols[:,3] = color_by
    return np.round(R_add_cols, 2)


def plot_times_series(time_data_df, lower_bound, upper_bound, outfile = None):
    """
    Plot reconstructed timeseries

    Parameters
    ----------
    time_data_df : pandas dataframe
        Dataframe with reconstructed and actual timeseries

    lower_bound : int
        Lower bound for timeseries

    upper_bound : int
        Upper bound for timeseries

    outfile : string
        File name

    Returns
    ----------
    results : plotted timeseries
         If outfile in arguments, the plot is saved.  Otherwise, it the plot is shown.

    """
    fig, ax = plt.subplots()
    sb.set_style("white")
    time_data_df[(time_data_df['time'] > lower_bound) & (time_data_df['time'] < upper_bound)]['actual'].plot(ax=ax, title='observed activity', color='k', lw=2, fontsize=21)
    time_data_df[(time_data_df['time'] > lower_bound) & (time_data_df['time'] < upper_bound)]['predicted'].plot(ax=ax, color='r', lw=3)
    ax.legend(['actual', 'predicted'], fontsize=21)
    ax.set_xlabel("time", fontsize=21)
    ax.set_ylabel("voltage", fontsize=21)
    ax.set_axis_bgcolor('w')
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()


def scatter_plot(dataframe, x, y, outfile = None, *args, **kwargs):
    """
    Plot reconstructed timeseries

    Parameters
    ----------
    dataframe : pandas dataframe
        Dataframe with reconstructed and actual timeseries

    x : string
        Column header from dataframe - for x variable

    y : string
        Column header from dataframe - for y variable

    outfile : string
        File name


    Returns
    ----------
    results : scatter plot
         If outfile in arguments, the plot is saved.  Otherwise, it the plot is shown.

    """
    fig, ax = plt.subplots()
    if 'color' in kwargs:
        color = kwargs.get('color')
    if 'alpha' in kwargs:
        alpha = kwargs.get('alpha')
    if 'x_scale' in kwargs:
        x_scale = kwargs.get('x_scale')
        ax.set_xscale(x_scale)
        del kwargs['x_scale']
    if 'y_scale' in kwargs:
        y_scale = kwargs.get('y_scale')
        ax.set_yscale(y_scale)
        del kwargs['y_scale']
    if 'xlim' in kwargs:
        xlim = kwargs.get('xlim')
        ax.set_xlim(xlim)
        del kwargs['xlim']
    if 'ylim' in kwargs:
        ylim = kwargs.get('ylim')
        ax.set_ylim(ylim)
        del kwargs['ylim']
    if 'electrode' in kwargs:
        electrode = kwargs.get('electrode')
        subject = kwargs.get('subject')
        ax.plot(dataframe[x][(dataframe.Subject == subject)& (dataframe.Electrode == electrode)], dataframe[y][(dataframe.Subject == subject)& (dataframe.Electrode == electrode)], marker='o',
                color='r')
        ax.set_title(y + ' vs ' + x + ' Subject = ' + subject + ' Electrode = ' + str(electrode))
        del kwargs['subject']
        del kwargs['electrode']
    else:
        if 'subject' in kwargs:
            subject = kwargs.get('subject')
            ax.plot(dataframe[x][dataframe.Subject == subject], dataframe[y][dataframe.Subject == subject], marker='o',
                color='r')
            ax.set_title(y + ' vs ' + x + ' Subject = ' + subject)
            del kwargs['subject']
    if 'title' in kwargs:
        title = kwargs.get('title')
        ax.set_title(title)
        del title
    if 'plot_type' in kwargs:
        plot_type = kwargs.get('plot_type')
        del kwargs['plot_type']
        if plot_type == 'scatter':
            ax.scatter(dataframe[x], dataframe[y], *args, **kwargs)
        elif plot_type == 'bar':
            ax.bar(dataframe[x], dataframe[y], *args, **kwargs)
        elif plot_type == '2d_hist':
            ax2 = ax.twinx()
            sb.jointplot(dataframe[x], dataframe[y], kind="kde", space=0, ax = ax2, **kwargs)
    else:
        ax.plot(dataframe[x], dataframe[y], *args, **kwargs)
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()


def pca_plot(npz_data, outfile=None, normalize=True):
    """
    Plot PCA

    Parameters
    ----------
    npz_data : str
        Path to npz data

    outfile : string
        File name

    Returns
    ----------
    results : PCA plot
         If outfile in arguments, the plot is saved.  Otherwise, it the plot is shown.

    """
    def extract_pca(Corr_timeseries):
        corr_data = np.load(Corr_timeseries, mmap_mode='r')
        PCA = corr_data['PCA']
        return pd.DataFrame({'sessions': PCA[:, 0], 'Components': PCA[:, 1], 'Var': PCA[:, 2]})

    # def normalize_pca(pca_dataframe):
    #     Y = np.atleast_2d(pca_dataframe)
    #     if np.shape(Y)[0] == 1:
    #         Y = Y.T
    #     Y = np.atleast_2d(pca_dataframe).T
    #     m = mat.repmat(np.min(Y, axis=0), Y.shape[0], 1)
    #     Y = Y - m
    #     m = mat.repmat(np.max(Y, axis=0), Y.shape[0], 1)
    #     return pd.DataFrame(np.divide(Y, m))

    def parse_path_name(path_name):
        if os.path.basename(path_name).count('_') == 1:
            f_name = os.path.splitext(os.path.basename(path_name))[0].split("_", 1)[0]
            return f_name
        elif os.path.basename(path_name).count('_') == 2:
            f_name = '_'.join(os.path.splitext(os.path.basename(path_name))[0].split("_", 2)[0:2])
            return f_name
        else:
            return "error"

    full_pca = pd.DataFrame()
    fig, ax = plt.subplots()
    ax.set_axis_bgcolor('w')
    ax.set_ylabel('Variance Explained')
    ax.set_ylim(0,1)
    for i in npz_data:
        compile_temp = extract_pca(i)
        file_name = parse_path_name(i)
        if full_pca.empty:
            temp_pca = compile_temp.groupby('Components')['Var'].mean().rename(file_name)
            summed_pca = np.cumsum(temp_pca)
            full_pca = full_pca.append(summed_pca.T)
            summed_pca.plot(alpha=.3, color='k', ax=ax, legend=False)
        else:
            temp_pca = compile_temp.groupby('Components')['Var'].mean().rename(file_name)
            summed_pca = np.cumsum(temp_pca)
            full_pca = full_pca.append(summed_pca.T)
            summed_pca.plot(alpha=.3, color='k', ax=ax, legend=False)

    mean_full = full_pca.mean(axis=0)
    mean_full.plot(color='r')
    plt.axhline(y=0.95, xmin=0, xmax=1, hold=None, color = 'k')
    plt.axvline(x = mean_full[mean_full>.95].idxmin(), ymin=0, ymax=1, hold=None, color = 'k')
    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()
    return full_pca
