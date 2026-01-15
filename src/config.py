def get_config():
    '''
    Get the config dict used for trainer and generator
    '''
    cfg = {
        'data_path': '/content/nsynth_data/nsynth-valid',
        'output_dir': '/content/results/run_01',
        'script': False,

        'conditioning': 'time', # 'time' or 'concat'
        'class_emb_size': 32,
        'use_pitch': False,
        'T_target': 160,
        'max_samples': None,
        'equalize_data': False,
        'selected_families': ['guitar', 'mallet', 'brass'],
        'epochs': 15,
        'batch_size': 32,
        'lr': 1e-4,
        'change_lr': None,
        'save_interval': 5,
        'cfg_prob': 0.1,

        'num_train_timesteps':1000,
        'beta_start':0.0001,
        'beta_end':0.02,
        'beta_schedule':"linear",

        'denoise_method': None, # 'spectral', 'nlm', or None
        'denoise_strength': 0.0,
        'custom_kernel': [0.15, 0.7, 0.15]
    }
    return cfg