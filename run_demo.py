import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


# - (DTM) **rosy-morning-93:** data/model/dcrnn_DR_2_h_12_64-64_lr_0.01_bs_64_1016200818/models-1.0771-33060
# - (BASE) **ancient-terrain-91**: data/model/dcrnn_DR_2_h_12_64-64_lr_0.01_bs_64_1016132112/models-1.6182-30210

DTM_MODEL = 'data/model/dcrnn_DR_2_h_12_64-64_lr_0.01_bs_64_1016200818/config_57.yaml'  # models-1.0771-33060
BASE_MODEL = 'data/model/dcrnn_DR_2_h_12_64-64_lr_0.01_bs_64_1016132112/config_52.yaml' # models-1.6182-30210


def run_dcrnn(args):
    with open(args.config_filename) as f:
        config = yaml.load(f)
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    graph_pkl_filename = config['data']['graph_pkl_filename']
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
        supervisor.load(sess, config['train']['model_filename'])
        outputs = supervisor.evaluate(sess)
        np.savez_compressed(args.output_filename, **outputs)
        print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default=DTM_MODEL, type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()
    run_dcrnn(args)
