import superEEG

# load example data
bo = superEEG.load_example_data()

# load example model
model = superEEG.load_example_model()
print(model.locs)

# debug predict.py
sub_corrmat = superEEG.predict(bo, model=model)
