import superEEG

# initialize default contexts
single = {
    'environment' : 'single',
    'nodes' : 1,
    'memory' : 8,
}
cluster = {
    'environment' : 'cluster',
    'nodes' : 4,
    'memory' : 30,
}

def set_context(context='single'):
    """
    Sets the context of the analyses: single machine, cluster, etc.
    """
    if context is 'single':
        superEEG.context = single
    elif context is 'cluster':
        superEEG.context = cluster
    elif type(context) is dict:
        superEEG.context = context
