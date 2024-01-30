class Config_Shanghai_2020(object):
    model_name = ''
    ini_params_mode = 'orthogonal'

    dBZ_threshold = 10

    dataset = ''
    dataset_root = ''
    in_seq_len_11 = 11
    in_seq_len = 10
    out_seq_len = 10
    seq_len = in_seq_len + out_seq_len
    seq_interval = None

    use_gpu = True
    device_ids = [0,1]
    num_workers = 8
    train_batch_size = 8
    valid_batch_size = 8
    test_batch_size = 8
    train_max_epochs = 50
    learning_rate = 0.001#1e-4
    optim_betas = (0.5, 0.999)
    scheduler_gamma = 0.5
    cross_learning_rate = 0.001
    cross_optim_betas = (0.5, 0.999)
    cross_scheduler_gamma = 0.5
    # adv train config
    model_train_fre = 1
    log_dir = './logdir_shanghai_2020'
    loss_log_iters = 100

    model_save_fre = 1
    checkpoints_dir = './checkpoints'




config_sh = Config_Shanghai_2020()
