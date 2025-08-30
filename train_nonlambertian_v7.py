from __future__ import absolute_import, division, print_function

from nonlambertian_trainer_v7 import NonLambertianTrainerV7
from nonlambertian_options_v7 import NonLambertianOptions


options = NonLambertianOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = NonLambertianTrainerV7(opts)
    trainer.train()