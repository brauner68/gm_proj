def get_config():
    '''
    Get the config dict used for trainer and generator
    '''
    cfg = {
        'data_path': '/content/nsynth_data/nsynth-valid',
        'output_dir': '/content/results/run_01',
        'T_target': 160,
        'max_samples': None,
        'selected_families': ['guitar', 'mallet', 'brass'],
        'epochs': 15,
        'batch_size': 32,
        'lr': 1e-4,
        'save_interval': 5,
        'cfg_prob': 0.1,

        'num_train_timesteps':1000,
        'beta_start':0.0001,
        'beta_end':0.02,
        'beta_schedule':"linear"
    }
    return cfg