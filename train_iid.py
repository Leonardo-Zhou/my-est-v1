from __future__ import absolute_import, division, print_function

from iid_trainer import IIDTrainer
from iid_options import IIDOptions

options = IIDOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = IIDTrainer(opts)
    trainer.train()