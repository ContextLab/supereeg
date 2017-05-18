import superEEG
import seaborn as sb

# load example data
bo = superEEG.load_example_data()

# remove elecs that exceed some electrode
bo = bo.remove_elecs()

# load example model
model = superEEG.load_example_model()

# debug predict.py
sub_corrmat = superEEG.predict(bo, model=model)

sb.heatmap(sub_corrmat)
sb.plt.show()
