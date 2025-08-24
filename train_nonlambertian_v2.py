from __future__ import absolute_import, division, print_function

from nonlambertian_trainer_v2 import NonLambertianTrainer
from nonlambertian_options_v2 import NonLambertianOptions

options = NonLambertianOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = NonLambertianTrainer(opts)
    trainer.train()