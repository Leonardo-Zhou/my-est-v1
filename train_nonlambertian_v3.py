from __future__ import absolute_import, division, print_function

from nonlambertian_options_v3 import NonLambertianOptions
from nonlambertian_trainer_v3 import NonLambertianTrainer

options = NonLambertianOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = NonLambertianTrainer(opts)
    trainer.train()