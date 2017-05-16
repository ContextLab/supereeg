import superEEG

# load example data
bo = superEEG.load_example_data()

# debug predict.py
sub_corrmat = superEEG.predict(bo)

print(sub_corrmat)
