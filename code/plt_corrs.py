
import glob
import os
import sys
import scipy.io
from scipy.stats import zscore
import numpy.matlib as mat
from scipy import stats, linalg
import numpy as np
from stats import z2r, r2z, compile_corrs, density, tal2mni, normalize_Y, round_it, good_channels_loc, compile_nn_corrs, good_chans
from plot import  scatter_plot, brain_net_viewer_format, plot_times_series, pca_plot
from bookkeeping import get_parent_dir, get_grand_parent_dir, get_rows, known_unknown
import matplotlib as mpl

import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import pandas as pd
import re
import seaborn as sb




### input is now: path_to_npz_subject_data electrode_number radius_for_RBF kurtosis_threshold
def main(fname, electrode, r, k_thresh):
    ###################################################################################
    ### subject_plots = True if you want subject plots
    subject_plots = True
    ### nearest neighbors - change to 11
    nearest_ns = 3
    ### radius for brain plots:
    plot_r = .5
    ### for timeseries interval
    lower_time = 100
    upper_time = 102
    lower_time_gif = 0
    upper_time_gif = 10
    ###################################################################################
    ################### compile paths to directories: #################################
    path_to_npz =os.path.splitext(os.path.dirname(fname))[0]
    file_name = os.path.splitext(os.path.basename(fname))[0]
    fig_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'figs/paper')
    compare_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'correlations')
    time_dir = os.path.join(compare_dir, 'timeseries_recon')
    average_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'ave_matrices')
    full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices')
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')
    pca_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'pca')
    nn_corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'nn_corr')

    gif_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'gif')
    if not os.path.isdir(pca_dir):
        os.mkdir(pca_dir)
    loc_name = 'R_full_MNI.npy'
    k_loc_name = 'R_full_k_' + str(k_thresh) + '_MNI.npy'


    ################## compile paths to data for this subject: #########################
    npz_data = np.load(fname, mmap_mode='r')
    R_subj = tal2mni(npz_data['R'])
    R_full = np.load(os.path.join(get_parent_dir(os.getcwd()), k_loc_name))
    sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'), mmap_mode='r')
    full_data = np.load(os.path.join(full_dir, 'full_matrix_' + file_name + '_k' + str(k_thresh) + '_r' + str(r) + '.npz'), mmap_mode='r')
    corr_data = np.load(os.path.join(compare_dir, 'corr_timeseries_' + file_name + '_' + electrode + "_" + str(k_thresh) + '_' + str(r) + '.npz'), mmap_mode='r')
    ave_data = np.load(os.path.join(average_dir, 'average_full_matrix' + "_k_" + str(k_thresh) + '_r_' + str(r) + '.npz'), mmap_mode='r')
    R_std = np.load(os.path.join(get_parent_dir(os.getcwd()), 'locs.npz'))['R']

    # files = glob.glob(os.path.join(compare_dir, '*.npz'))
    # #all_corrs = pd.read_csv(os.path.join(fig_dir, 'all_corrs.csv')).dropna()
    # electrode = int(electrode)
    # all_corrs = pd.DataFrame()
    # ## this file_name split is particular to this data.  Resolved issue of multiple subjects with common basename TJO018_1, TJ018_2, TJ018
    # for i in files:
    #     matrix_variables = str('_' + str(k_thresh) + '_' + str(r)+'.npz')
    #     match = re.search(matrix_variables, i)
    #     if match:
    #         compile_temp = compile_corrs(path_to_npz, i)
    #         if all_corrs.empty:
    #             all_corrs = compile_temp
    #         else:
    #             all_corrs = all_corrs.append(compile_temp)
    #             all_corrs.to_csv(os.path.join(fig_dir, 'all_corrs.csv'))
    #     else:
    #         pass
    # all_corrs.to_csv(os.path.join(fig_dir, 'all_corrs.csv'))
    # ######## Find density and add to dataframe ########
    # all_corrs['Density'] = density(all_corrs['R'].tolist(), nearest_ns)
    #
    # ##### to create the .mat file for matlab analysis ##########
    # # data = pd.read_csv('all_corrs.csv')
    # data = all_corrs
    # a_dict = {col_name : data[col_name].values for col_name in data.columns.values}
    # scipy.io.savemat('all_corrs_to_mat.mat', {'struct':a_dict})
    # ######## Create CSV with data #################
    # all_corrs.to_csv(os.path.join(fig_dir, 'all_corrs.csv'))
    # ## issues with Nan
    #
    # #################### retrieve all pca vales and plot #######
    # files = glob.glob(os.path.join(pca_dir, '*.npz'))
    # outfile = os.path.join(fig_dir, 'PCA.pdf')
    # pca_full = pca_plot(files, outfile=outfile, normalize=True)
    # plt.clf()
    # min_comps = []
    # subject_number = []
    # for index, row in pca_full[pca_full > .95].iterrows():
    #     min_comps.append(row.idxmin(skipna=True))
    #     subject_number.append(index)
    # components = pd.DataFrame()
    # components['min_comps'] = min_comps
    # components['Subject'] = subject_number
    # ###### save for J ######
    # data = components
    # a_dict = {col_name: data[col_name].values for col_name in data.columns.values}
    # outfile = os.path.join(fig_dir, 'components_to_mat.mat')
    # scipy.io.savemat(outfile, {'struct': a_dict})
    # ######
    # ############## ratios from J #####################################
    # rats_comps = pd.DataFrame()
    # rats = scipy.io.loadmat(os.path.join(fig_dir, 'ratios_for_L.mat'))
    # for rats in rats['r']:
    #     rats_comps = rats_comps.append(pd.DataFrame({'Ratio': rats}))
    # n_count = len(rats_comps)
    # mpl.rcParams['axes.facecolor'] = 'white'
    # bin_values = np.arange(start=0, stop=1, step=.1)
    # ax = rats_comps.plot(kind='hist', bins=bin_values, color='k', title='# components that explain 95% of the variance', legend = False)
    # ax.set_xlim(0, 1)
    # vals = ax.get_yticks()
    # ax.set_yticklabels([round_it(x / n_count, 3) for x in vals])
    # ax.set_ylabel('Proportion of patients')
    # ax.set_xlabel('Number of components that explains 95% of the variance')
    # outfile = os.path.join(fig_dir, 'PCA_hist.pdf')
    # plt.savefig(outfile)
    # plt.clf()
    ########### Create timeseries for all electrodes ##############################
    R_K_subj, k_flat = good_chans(sub_data['K_subj'], R_subj, k_thresh)
    known_inds, unknown_inds = known_unknown(R_full, R_subj)
    unknown_timeseries = np.squeeze(np.dot(np.dot(np.float32(ave_data['average_matrix'][unknown_inds, :][:, known_inds]),
                                              np.linalg.pinv(
                                                  np.float32(
                                                      ave_data['average_matrix'][known_inds, :][:, known_inds]))),
                                       zscore(np.float32(npz_data['Y'][range(lower_time_gif, upper_time_gif), :])[:,
                                              k_flat]).T).T)
    known_timeseries = zscore(np.float32(npz_data['Y'][range(lower_time_gif, upper_time_gif), :][:,
                                              k_flat]))
    unknown_df = pd.DataFrame(unknown_timeseries.T, index=unknown_inds)
    known_df = pd.DataFrame(known_timeseries.T, index=known_inds)
    gif_df = pd.concat([unknown_df, known_df]).sort_index()
    gif_df.rename(columns=lambda x: 't_' + str(x).rjust(2,'0'), inplace=True)
    R=[]
    for row in R_full:
        R = np.append(R, str(row))
    gif_df['R'] = R
    gif_df.to_csv(os.path.join(gif_dir, 'gif_df.csv'))
    outfile = os.path.join(gif_dir, 'gif_df.mat')
    a_dict = {col_name : gif_df[col_name].values for col_name in gif_df.columns.values}
    scipy.io.savemat(outfile, {'struct': a_dict})


    # ###### compile nearest neighbors correlations into all_corrs, by electrode location and subject #######
    # files = glob.glob(os.path.join(nn_corr_dir, '*.npz'))
    # nn_corrs = pd.DataFrame()
    # for i in files:
    #     compile_temp = compile_nn_corrs(i)
    #     if nn_corrs.empty:
    #         nn_corrs = compile_temp
    #     else:
    #         nn_corrs = nn_corrs.append(compile_temp)
    # data = nn_corrs.dropna()
    # a_dict = {col_name: data[col_name].values for col_name in data.columns.values}
    # outfile = os.path.join(fig_dir, 'nn_corrs_to_mat.mat')
    # scipy.io.savemat(outfile, {'struct': a_dict})

    # ###### try to figure out how to run proportion kurtosis passed ####
    # corr_files = glob.glob(os.path.join(corr_dir, '*.npz'))
    # proportions = 0
    # n_count = 0
    # for i in corr_files:
    #     sub_data = np.load(i, mmap_mode='r')
    #     K_subj = sub_data['K_subj']
    #     k_flat = np.squeeze(np.where(K_subj < int(k_thresh)))
    #     if not k_flat == []:
    #         if np.shape(k_flat)[0] > 1:
    #             proportions += np.true_divide(np.shape(k_flat)[0],np.shape(K_subj)[0])
    #             n_count += 1
    #         else:
    #             print('noo')
    #     else:
    #         print('noo')
    #
    # print('total proportion = ' + str(np.true_divide(proportions,n_count)))
    #
    # k_flat = []
    # ###### find number of samples
    # DF_samples = pd.DataFrame()
    # files = glob.glob(os.path.join(corr_dir, '*.npz'))
    # for i in files:
    #
    #     def parse_path_name(path_name):
    #         if os.path.basename(path_name).count('_') == 2:
    #             f_name = os.path.splitext(os.path.basename(path_name))[0].split("_", 2)[2]
    #             return f_name
    #         elif os.path.basename(path_name).count('_') == 3:
    #             f_name = '_'.join(os.path.splitext(os.path.basename(path_name))[0].split("_", 3)[2:4])
    #             return f_name
    #         else:
    #             return "error"
    #
    #     f_name = parse_path_name(i)
    #     sub_data = np.load(i, mmap_mode='r')
    #     npz_data = np.load(os.path.join(path_to_npz, f_name + '.npz'), mmap_mode='r')
    #     K_subj = sub_data['K_subj']
    #     k_flat = np.squeeze(np.where(K_subj < int(k_thresh)))
    #     temp_pd = pd.DataFrame()
    #     if not k_flat == []:
    #         if np.shape(k_flat)[0] > 1:
    #             npz_data = np.load(os.path.join(path_to_npz, file_name + '.npz'), mmap_mode='r')
    #             tempsamples = np.shape(npz_data['Y'])[0]
    #             tempsamplerate = np.mean(npz_data['samplerate'])
    #             temp_pd = pd.DataFrame({'Subject': [f_name],'Sample rate': [tempsamplerate], 'Samples': [tempsamples]})
    #         else:
    #             print('noo')
    #     if DF_samples.empty:
    #         DF_samples = temp_pd
    #     else:
    #         DF_samples = DF_samples.append(temp_pd)
    # data = DF_samples
    # a_dict = {col_name: data[col_name].values for col_name in data.columns.values}
    # outfile = os.path.join(fig_dir, 'samples.mat')
    # scipy.io.savemat(outfile, {'struct': a_dict})
    # #################### retrieve all correlation values and compile in dataframe with subject number, electrode, and sample rate   #######
    # ### getting sample rate is tricky - requires opening individual's npz file in compile_corrs function
    #

    # ####### Create histogram of correlations #########
    # mpl.rcParams['axes.facecolor'] = 'white'
    # df_corrs = pd.DataFrame()
    # df_corrs['Correlations'] = all_corrs['Correlation'].values
    # n_count = len(df_corrs)
    # bin_values = np.arange(start=-1, stop=1, step=.025)
    # ax = df_corrs.plot(kind='hist', bins=bin_values, color='k', title = 'Correlations with RBF radius = ' + str(r), legend = False)
    # vals = ax.get_yticks()
    # ax.set_yticklabels([round_it(x/n_count,3) for x in vals])
    # ax.set_ylabel('Proportion of electrodes')
    # ax.set_xlabel('Correlation')
    # ax.set_xlim(-1, 1)
    # outfile = os.path.join(fig_dir, 'hist_'+ str(k_thresh) + '_' + str(r) + '.pdf')
    # plt.savefig(outfile)
    # plt.clf()
    #
    #
    # ####### Create density vs accuracy scatterplot ################
    # mpl.rcParams['axes.facecolor'] = 'white'
    # all_corrs['chosen_one'] = (all_corrs.Subject == file_name) & (all_corrs.Electrode == electrode)
    # # electrode = int(electrode)
    # outfile = os.path.join(fig_dir, 'density_acc_scatter_' + file_name + '_electrode_' + str(electrode) + '_' + str(k_thresh) + '_' + str(r) + '.pdf')
    # scatter_plot(all_corrs, 'Density', 'Correlation', plot_type = 'scatter', outfile = outfile, color = 'k', alpha=.3, subject=file_name, electrode=electrode, x_scale = 'log', xlim = (-4,2), ylim = (-1,1))
    # plt.clf()
    # outfile = os.path.join(fig_dir, 'density_acc_2dhist_' + str(k_thresh) + '_' + str(r) + '.pdf')
    # g = sb.jointplot(all_corrs['Density'], all_corrs['Correlation'], kind="kde", space=0, color='k')
    # g.ax_joint.plot(all_corrs['Density'][all_corrs['chosen_one']],
    #                 all_corrs['Correlation'][all_corrs['chosen_one']], marker='o',
    #                 color='r')
    # plt.savefig(outfile)
    # plt.clf()
    #
    # # ################### log plot ####################################
    # # ### figure out why theres an nan in correlation column
    # outfile = os.path.join(fig_dir, 'density_acc_scatter_log_' + file_name + '_electrode_' + str(electrode) + '_' + str(k_thresh) + '_' + str(r) + '.pdf')
    # mybins = 10 ** np.linspace(np.log10(1e-9), np.log10(1), 200)
    # g = sb.JointGrid('Density', 'Correlation', all_corrs, xlim=[1e-9,1],ylim=[-1,1])
    # g.ax_marg_x.hist(all_corrs['Density'], bins=mybins, color = 'k', alpha = .3)
    # g.ax_marg_y.hist(all_corrs['Correlation'], bins=np.arange(-1, 1, .01), orientation='horizontal', color = 'k', alpha = .3)
    # g.ax_marg_x.set_xscale('log')
    # g.ax_marg_y.set_yscale('linear')
    # g.plot_joint(plt.scatter, color='black', edgecolor='black', alpha = .6)
    # g.ax_joint.plot(all_corrs['Density'][all_corrs['chosen_one']],
    #                 all_corrs['Correlation'][all_corrs['chosen_one']], marker='o',
    #                 color='r')
    # ax = g.ax_joint
    # left, width = .05, .5
    # bottom, height = .05, .5
    # rstat = stats.pearsonr(all_corrs['Density'], r2z(all_corrs['Correlation']))
    # ax.text(left, bottom, 'r = ' + str(round_it(rstat[0],2)) + ' p = ' + str(rstat[1]),
    #         horizontalalignment='left',
    #         verticalalignment='top',
    #         transform=ax.transAxes)
    # ax.set_xscale('log')
    # ax.set_yscale('linear')
    #
    # plt.savefig(outfile, bbox_inches="tight")
    # plt.clf()
    #
    #
    # data = nn_corrs
    # a_dict = {col_name: data[col_name].values for col_name in data.columns.values}
    # scipy.io.savemat('nn_corrs_to_mat.mat', {'struct': a_dict})
    # ###### t-tests #########################################
    # true_mu = 0
    # ## all electrodes
    # onesample_results = scipy.stats.ttest_1samp(r2z(all_corrs['Correlation']), true_mu)
    # print('t-test for all electrodes = ' + str(onesample_results))
    # ## grouped by subject
    # onesample_results = scipy.stats.ttest_1samp(r2z(all_corrs.groupby('Subject')['Correlation'].mean()), true_mu)
    # print('t-test grouped by subject = ' + str(onesample_results))
    #
    # ################## Subject Plots : #######################################
    # if subject_plots:
    #     ########## correlation and electrode location ######################################
    #     corr_electrode = corr_data['corrs']
    #     R_electrode = corr_data['coord']
    #
    #     ########## create text file with electrode locations for brain net viewer #########
    #     electrode_ind = get_rows(R_full, R_electrode)
    #     sub_inds = get_rows(R_full, R_subj)
    #     node_matrix = brain_net_viewer_format(R_full, sub_inds, electrode_ind)
    #
    #     ########## need to save an ascii file for brain net viewer ########################
    #     np.savetxt(os.path.join(fig_dir, file_name + '.node'), node_matrix)
    #
    #     ################# Plot normed Y matrix ###################################
    #     sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'), mmap_mode='r')
    #     R_K_subj, k_flat_removed = good_channels_loc(sub_data['K_subj'], R_subj, k_thresh, electrode)
    #     Y = normalize_Y(npz_data['Y'])
    #     Y.reset_index()
    #     Y['time'] = Y.index / npz_data['samplerate'].mean()
    #     ##### create time mask ######
    #     mask = (Y['time'] > lower_time) & (Y['time'] < upper_time)
    #     ax = Y[Y.columns[k_flat_removed]][mask].plot(legend=False, title='electrode activity for ' + file_name, color='k', lw=.6)
    #     Y[Y.columns[int(electrode)]][mask].plot(color='r', lw=.8)
    #     ax.set_axis_bgcolor('w')
    #     ax.set_xlabel("time")
    #     ax.set_ylabel("electrode")
    #     ax.set_ylim([0,len(Y.columns) + 1])
    #     vals = ax.get_xticks()
    #     ax.set_xticklabels([round_it(x/npz_data['samplerate'].mean(),3) for x in vals])
    #     plt.savefig(os.path.join(fig_dir,'y_matrix_'+ file_name+ '.pdf'))
    #     ##### may still need to plot kurtosis removed electrodes
    #     plt.clf()
    #     ############## plot zscored Y matrix ###############################
    #     Y = stats.zscore(npz_data['Y'])
    #     added = mat.repmat(1 + np.arange(Y.shape[1])*3, Y.shape[0], 1)
    #     Y = pd.DataFrame(Y+ added)
    #     ax = Y[Y.columns[k_flat_removed]][mask].plot(legend=False, title='electrode activity for ' + file_name, color='k', lw=.6)
    #     Y[Y.columns[int(electrode)]][mask].plot(color='r', lw=.8)
    #     yvals = ax.get_yticks()
    #     ax.set_yticklabels([round_it(x / 3, 3) for x in yvals])
    #     xvals = ax.get_xticks()
    #     ax.set_xticklabels([round_it(x / npz_data['samplerate'].mean(), 3) for x in xvals])
    #     ax.set_axis_bgcolor('w')
    #     ax.set_xlabel("time")
    #     ax.set_ylabel("electrode")
    #     ax.set_ylim([-2, len(Y.columns) * 3 + 1])
    #     plt.savefig(os.path.join(fig_dir, 'y_matrix_zscore_' + file_name + '.pdf'))
    #     plt.clf()
    #
    #     ############# subject summary ###########################################
    #     total_files = glob.glob(os.path.join(corr_dir, '*.npz'))
    #     R_full_beforek = np.load(os.path.join(get_parent_dir(os.getcwd()), loc_name))
    #     print('total # electrodes = ' + str(len(R_full_beforek)))
    #     print('total # electrodes that pass kurtosis = ' + str(len(all_corrs['Electrode'])))
    #     print('total # subjects in dataset = ' + str(len(total_files)))
    #     print('total # subjects that contributed = ' + str(len(all_corrs.groupby('Subject'))))
    #     print('total # electrodes for sample subject = ' + str(len(R_subj)))
    #     print('total # electrdoes for sample subject that pass kurtosis threshold = ' + str(len(R_K_subj)))

        ##################  Covariance Matrix ####################################
        # outfile = os.path.join(fig_dir, file_name + '_full_cov.png')
        # plot_cov(reverse_squareform(full_data['C_est']), outfile, file_name + 'Covariance Matrix')
        # plt.clf()

        ################## Average Matrix ########################################
        # outfile = os.path.join(fig_dir, 'average_full_cov.png')
        # plot_cov(ave_data['average_matrix'], outfile, 'Average Covariance Matrix')
        # plt.clf()


        # ################## plots for actual vs predicted timeseries #####################
        # if os.path.isfile(os.path.join(time_dir, 'corr_timeseries_T_' + file_name + '_' + electrode + '_' + str(k_thresh) + '_' + str(r) + '.npz')):
        #     time_data = np.load(os.path.join(time_dir, 'corr_timeseries_T_' + file_name + '_' + electrode + '_' + str(
        #         k_thresh) + '_' + str(r) + '.npz'), mmap_mode='r')
        #     corrs = time_data['corrs']
        #     df_subj_corrs = pd.DataFrame({'actual':corrs[:,0],'predicted':corrs[:,1],'session':corrs[:,2],'time':corrs[:,3]/npz_data['samplerate'].mean()})
        #     print('sample rate is = ' + str(npz_data['samplerate'].mean()))
        #     print('total samples = '+ np.max(corrs[:,3]))
        #     print('total amount of time in seconds = ' + str(df_subj_corrs['time'].max()))
        #     print('total amount of time in hours = ' + str(df_subj_corrs['time'].max()/3600))
        #     print('total number of sessions = ' + str(df_subj_corrs['session'].max()))
        #
        #     ########## view reconstructed timeseries ##########################################
        #     outfile = os.path.join(fig_dir, file_name + '_' + electrode + '_interval_' + str(lower_time) + '_' + str(upper_time) + '_timeseries_recon.pdf')
        #     plot_times_series(df_subj_corrs, lower_time, upper_time, outfile)
        #     plt.clf()
        #     ########## view 2D histogram of predicted vs actual times ##########################
        #     # sb.set_style("white")
        #     # sb.jointplot(x = 'actual', y = 'predicted', data=df_subj_corrs, kind="kde", color='k', xlim = (-3,3.5), ylim = (-2,2))
        #     # plt.savefig(os.path.join(fig_dir, file_name + '_2d_hist.pdf'))
        #     # plt.clf()
        #
        # else:
        #     print('timeseries reconstruction does not exist for' + file_name + '_electrode_' + electrode)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


