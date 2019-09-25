# -*- coding: utf-8 -*-
import os
from config_util import Convertible, immutables, learning, structures, setups
from utils import line_count, current_datetime


class Configuration(Convertible):
    def __init__(self, name, work_dir, data_dir, output_root):
        super(Configuration, self).__init__()
        # file related params
        self.work_dir = work_dir
        self.data_dir = data_dir
        self.vocab_path = os.path.join(self.data_dir, 'vocabulary')
        self.embed_path = os.path.join(self.data_dir, 'glove_embedding.npy')
        self.vocab_size = line_count(self.vocab_path, skip_empty=True)

        with setups(self):
            with immutables(self):
                self.start_time = current_datetime()
                self.name = name
                self.output_dir = os.path.join(output_root, name)
                self.train_path = os.path.join(self.data_dir, 'trainyiseg.csv')
                self.train_eval_path = os.path.join(self.data_dir, 'trainyiseg_eval.csv')
                self.valid_path = os.path.join(self.data_dir, 'validyiseg.csv')
                self.test_path = os.path.join(self.data_dir, 'testayiseg.csv')
                self.model_path = os.path.join(self.output_dir, 'model')
                self.elmo_path = os.path.join(self.data_dir, 'elmo', 'model')
                self.num_aspects = 20
                self.visible_gpus = '0'  # visible GPUs
                self.num_gpus = len(self.visible_gpus.split(','))
                os.environ['CUDA_VISIBLE_DEVICES'] = self.visible_gpus

        with structures(self):
            self.hidden_size = 512
            self.embed_size = 300
            self.atn_units = 300
            self.num_layers = 1
            self.use_elmo = False

        with learning(self):
            # training process params
            self.load_embed = True
            self.keep_prob = 0.65
            self.rnn_kernel_keep_prob = 0.8
            self.max_epoch = 50
            self.grad_clip_max_norm = 5.0
            self.early_stop_epoch = 10

            # input params
            self.batch_size = 64
            self.eval_batch_size = 64
