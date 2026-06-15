
import yaml


class ConfigDict(dict):
    """ 
    A dictionary that allows access to its keys as attributes where unset elements return None instead of raising errors.
    """
    def __getitem__(self, key):
        return self.get(key, None)

    def __getattr__(self, key):
        return self.get(key, None)

    def __setattr__(self, key, value):
        self[key] = value

    # merge the two configs, if the key is not in the config file, use the default value
    def update(self, u):
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(self.get(k), dict):
                self[k].update(v)
            else:
                self[k] = v


def parse_config(config_file, default_config_file):
    """
    This function parses a YAML configuration file and merges it with default configuration values.
    Where values of default_config_file are overwritten by those in config_file.

    Args:
        config_file (str): Path to the YAML configuration file.
        default_config_file (str): Path to the default configuration YAML file.
    Returns:
        ConfigDict: A dictionary-like object containing the merged configuration.
    """
    with open(default_config_file, 'r') as file:
        default_config = yaml.safe_load(file)

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # loop all the properties and if they are dict with a value key use that value
    for k, v in config.items():
        if isinstance(v, dict) and 'value' in v:
            config[k] = v['value']

    merged_config = ConfigDict(default_config)
    merged_config.update(config)

    # Is more convenient to hardcode the frequency, patch size or orther important parameter for your model here,
    # so for each model we do not have to set in the config file, making it cleaner
    if merged_config.use_ecg_jepa:
        merged_config.sampling_freq = 250
        merged_config.patch_size = 50
        # jepa uses 8 leads
        merged_config.leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    elif merged_config.use_st_mem:
        merged_config.sampling_freq = 250
        merged_config.patch_size = 75
        merged_config.low_pass_filter = 40
        merged_config.high_pass_filter = 0.67
        merged_config.standardize = True

    elif merged_config.use_ecg_founder:
        merged_config.sampling_freq = 500
        merged_config.low_pass_filter = 50
        merged_config.high_pass_filter = 0.5
        merged_config.layerwise_lr_decay = 1.
        merged_config.drop_path_prob = 0.
        # merged_config.normalize = True

    elif merged_config.use_ecg_cpc:
        merged_config.sampling_freq = 240
        merged_config.layerwise_lr_decay = 1.
        merged_config.drop_path_prob = 0.
        merged_config.discriminative_lr_factor = 0.1
        merged_config.patch_size = 2

    if merged_config.linear_probing:
        merged_config.layerwise_lr_decay = 1.
        merged_config.drop_path_prob = 0.

    if merged_config.r_peaks_detection:
        merged_config.num_classes = merged_config.patch_size
    
    return merged_config


def set_num_classes_r_peaks(config):
    if config.use_ecg_founder:
        config.num_classes = 5000
    else:
        # ensure that num_classes is equal to patch_size
        config.num_classes = config.patch_size

    return config