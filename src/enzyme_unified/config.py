TASK_CONFIGS = {
    "kcat_km": {
        "csv_path": "dataset/kcat-km_processed_0.4_data_10fold.csv",
        "label_col": "kcat_km",
        "folds": 10,
        "log_target": True,
        "default_lr": 1e-5,
        "default_batch_size": 512,
    },
    "ph": {
        "csv_path": "dataset/ph_largeset_data_clustered_0.4_5fold.csv",
        "label_col": "pH",
        "folds": 5,
        "log_target": False,
        "default_lr": 5e-4,
        "default_batch_size": 128,
    },
    "topt": {
        "csv_path": "dataset/topt_data_clustered_0.4_5fold.csv",
        "label_col": "temperature",
        "folds": 5,
        "log_target": False,
        "default_lr": 1e-3,
        "default_batch_size": 64,
    },
}

VARIANTS = {
    "hybrid": {"use_prostt5": False, "use_physchem": False},
    "hybrid_prostt5": {"use_prostt5": True, "use_physchem": False},
    "hybrid_pp": {"use_prostt5": False, "use_physchem": True},
}

