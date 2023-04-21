import selberai.data.load_data as load_data

def analyse(config: dict, name: str, subtask: str):
    """
    """
    print('Analysing {} for subtask {}'.format(name, subtask))
    
    ###
    # Load data
    ###
    
    # set the path to data and token we want to load
    path_to_data = config['general']['path_to_data']
    path_to_token = config['dataverse']['path_to_token']
    
    # load data
    dataset = load_data.load(name, subtask, sample_only=True, tabular=True, path_to_data=path_to_data, path_to_token=path_to_token, )
    print(type(dataset))
    
    ###
    # Analyse st-ood score
    ###
    
    ###
    # Analyse 
    ###
    
    
    # 1) JS DIVERGENCE
    # 2) PCA -> POIT2POINT DERIVATIVE (shifting the features and labels) -> MEAN

    # 3) OOD SCORE -> JS DIVERGENCE BTW train and val/test
