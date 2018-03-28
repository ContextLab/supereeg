import supereeg as se

bo = se.load('example_data')
mo = se.load('example_model')

bo_recon = mo.predict(bo, nearest_neighbor=True)

nii = bo_recon.to_nii()
nii.plot_glass_brain()