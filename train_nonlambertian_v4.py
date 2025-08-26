from __future__ import absolute_import, division, print_function

from nonlambertian_options_v4 import NonLambertianOptionsV4
from nonlambertian_trainer_v4 import NonLambertianTrainerV4

options = NonLambertianOptionsV4()
opts = options.parse()

if __name__ == "__main__":
    trainer = NonLambertianTrainerV4(opts)
    trainer.train()