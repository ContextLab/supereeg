import superEEG

# print default context
print(superEEG.context)

# change context to preset cluster default
superEEG.set_context('cluster')

# print updated context
print(superEEG.context)

# change context to custom dict
google_cluster = {
    'environment' : 'google-cluster',
    'nodes' : 1000000000,
    'memory' : 3000000000,
}
superEEG.set_context(google_cluster)
print(superEEG.context)
