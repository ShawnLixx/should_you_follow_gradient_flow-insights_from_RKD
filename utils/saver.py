import os
from os.path import join
from os import makedirs
from shutil import copyfile
import json

import tensorflow as tf

class Saver:
    def __init__(self, args):
        self.args = args
        if args.model_name is None:
            config_identifier = \
                f"lr-denom_{args.lr_denom}_momen_{args.momentum:.2f}" \
                f"_seed_{args.random_seed}"
            if args.optim == 'SGD':
                config_identifier += f"_bs_{args.batch_size}"
            if args.optim == 'RMSprop':
                config_identifier += f"_alpha_{args.rms_alpha:.2f}"
            if args.nesterov:
                config_identifier += f"_Nesterov"
            if 'ResNet' in args.model and args.fix_up:
                config_identifier += f"_fix-up"
        else:
            config_identifier = args.model_name
        self.save_dir = join(
                args.save_dir, args.dataset, args.model,
                args.optim, args.loss, config_identifier)
        # makedirs(self.save_dir, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(self.save_dir)


        # save all configurations
        with self.summary_writer.as_default():
            tf.summary.text("config",
                    [[k, str(w)] for k, w in sorted(vars(args).items())],
                    step=0)

    def write(self, kind, iteration, item):
        with self.summary_writer.as_default():
            tf.summary.scalar(kind, item,
                    step=iteration)

