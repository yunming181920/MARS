import torch
from attacks.attack import Attack

class CerP(Attack):

    def __init__(self, params, synthesizer):
        super().__init__(params, synthesizer)
        self.loss_tasks.append('cs_constraint')
        self.fixed_scales = {'normal':0.3,
                            'backdoor':0.3,
                            'cs_constraint':0.4}

    