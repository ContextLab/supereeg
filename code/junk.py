# fig, ax = plt.subplots()
# mpl.rcParams['axes.facecolor'] = 'white'
# df_corrs = all_corrs.groupby('Subject')['Correlation']
# bin_values = np.arange(start=-1, stop=1, step=.025)
# ax = df_corrs.plot(kind='hist', bins=bin_values, alpha=.5, lw=3, title = 'Correlations with RBF radius = ' + str(r))
# vals = ax.get_yticks()
# ax.set_yticklabels([round_it(x / n_count, 3) for x in vals])
# ax.set_ylabel('Proportion of electrodes, grouped by subject')
# ax.set_xlabel('Correlation')
# ax.set_xlim(-1, 1)
# outfile = os.path.join(fig_dir, 'hist_color_' + str(k_thresh) + '_' + str(r) + '.pdf')
# plt.savefig(outfile)
# plt.clf()

######### Plot Density BNV ####################################
# node_matrix = brain_net_viewer_format_color(np.asarray(all_corrs['R'].tolist()), np.asarray(all_corrs['density']))
# np.savetxt(os.path.join(fig_dir, 'electrode_density.node'), node_matrix)

#
# ######## Create accuracy vs samplerate ####################
# outfile = os.path.join(fig_dir, 'samplerate_acc_scatter_' + file_name + '_electrode_' + str(electrode) + '_' +  str(k_thresh) + '_' + str(r) + '.pdf')
# df_sr = all_corrs.groupby(['Subject']).mean().reset_index()
# df_sr['chosen_one'] = df_sr.Subject == file_name
# scatter_plot(df_sr, 'Sample rate', 'Correlation', plot_type = 'scatter', outfile = outfile,  color = 'k', alpha = .6, subject = file_name)
# plt.clf()
# outfile = os.path.join(fig_dir, 'samplerate_acc_swarm_' + file_name + '_electrode_' + str(electrode) + '_' + str(k_thresh) + '_' + str(r) + '.pdf')
# ax = sb.swarmplot(df_sr['Sample rate'], df_sr['Correlation'], hue=df_sr['chosen_one'], palette = {'k', 'r'})
# ax.legend_.remove()
# plt.savefig(outfile)

# ######## Create vector for brain data - with new val_interp function instead of RBF ################
#
# # R_corrs = val_interp(R_std, 'Correlation', all_corrs, tau = .5, fishersZ=True)
# # flattened_corr = flatten_arrays(R_corrs.Correlation.values.flatten())
# # scipy.io.savemat(os.path.join(fig_dir, 'mean_corr_k_' + str(k_thresh) + '_r_' + str(r) + '_pltr_' + str(plot_r) + '.mat'), dict(R=R_std, V=np.atleast_2d(flattened_corr)))
# # R_density = val_interp(R_std, 'Correlation', all_corrs, tau=.5, fishersZ=False)
# # flattened_density = flatten_arrays(R_density.Correlation.values.flatten())
# # scipy.io.savemat(os.path.join(fig_dir, 'mean_corr_k_' + str(k_thresh) + '_r_' + str(r) + '_pltr_' + str(plot_r) + '.mat'), dict(R=R_std, V=np.atleast_2d(flattened_density)))
#
# # ######## try parallel processing ##############################################
# # inputs = [(a_row) for a_row in R_std[0:np.shape(R_std)[0]:1, :]]
# # num_cores = multiprocessing.cpu_count()
# # print(num_cores)
# # # ## can't have the delayed(processInput) in a function
# # R_corrs = Parallel(n_jobs=num_cores)(
# #     delayed(val_interp_parallel)(a_row, 'Correlation', all_corrs, tau = 1, fishersZ=True) for a_row in inputs)
# # flattened_corr = flatten_arrays(R_corrs)
# # scipy.io.savemat(
# #     os.path.join(fig_dir, 'mean_corr_k_' + str(k_thresh) + '_r_' + str(r) + '_tau_' + str(1) + '.mat'),
# #     dict(R=R_std, V=np.atleast_2d(flattened_corr)))
# # R_density = Parallel(n_jobs=num_cores)(
# #     delayed(val_interp_parallel)(a_row, 'Density', all_corrs, tau = 1, fishersZ=False) for a_row in inputs)
# # flattened_density = flatten_arrays(R_density)
# # scipy.io.savemat(
# #     os.path.join(fig_dir, 'mean_dense_k_' + str(k_thresh) + '_r_' + str(r) + '_tau_' + str(1) + '.mat'),
# #     dict(R=R_std, V=np.atleast_2d(flattened_density)))
# ############### RBF plot ################################################
# if not os.path.isfile(os.path.join(fig_dir, 'RBF_r' + str(r) + '.png')):
#     outfile = os.path.join(fig_dir, 'RBF.png')
#     center = np.atleast_2d([0, 0])
#     coords = []
#     fun_map = np.empty((1000,1000))
#     for x in np.linspace(-2,2,1000):
#         for y in np.linspace(-2,2,1000):
#             coordinate = (x, y)
#             coords.append(coordinate)
#
#     weight = rbf(coords, center, 1)
#     weights = weight.reshape(1000, 1000)
#     sb.heatmap(weights, cmap="RdBu_r")
#     plt.savefig(outfile)
#     plt.clf()

# b = BrainData(os.path.join(get_parent_dir(os.getcwd()), 'avg152T1_gray_3mm.nii.gz'))


