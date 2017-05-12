import superEEG

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
        superEEG.__context__ = single
    elif context is 'cluster':
        superEEG.__context__ = cluster
    elif type(context) is dict:
        superEEG.__context__ = context
