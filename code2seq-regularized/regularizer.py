from enum import Enum
import pickle
import torch
import numpy as np


class Geometry(Enum):
    l1 = 1
    l2 = 2


class EncoderDecoderRegularizer:
    def __init__(self, geometry=Geometry.l1, model=[]):
        self.geometry = geometry
        self.model = model

    def __call__(self, loss, iter):
        return loss


class StandardRegularizer(EncoderDecoderRegularizer):
    def __init__(self, coeff, geometry=Geometry.l1, model=[], only_penalty=False):
        super().__init__(geometry, model)
        self.only_penalty = only_penalty
        self.coeff = coeff

    def __call__(self, loss, _iter):
        params = [torch.cat([x.view(-1) for x in part.parameters()])
                  for part in self.model]

        penalty = 0.0
        for param in params:
            val = torch.norm(param, self.geometry.value) ** self.geometry.value
            penalty += val
        if self.only_penalty:
            penalty = penalty ** (1./self.geometry.value)

        return loss + self.coeff * penalty


class ProximalRegularizer(EncoderDecoderRegularizer):
    def __init__(self, coeff, geometry=Geometry.l1, model=[], only_penalty=False, prox_period=100):
        super().__init__(geometry, model)
        self.only_penalty = only_penalty
        self.coeff = coeff
        self.prox_point = [torch.cat([torch.zeros_like(x.view(-1)) for x in part.parameters()])
                           for part in self.model]

    def __call__(self, loss, _iter):
        params = [torch.cat([x.view(-1) - self.prox_point[i] for i, x in enumerate(part.parameters())])
                  for part in self.model]

        penalty = 0.0
        for param in params:
            val = torch.norm(param, self.geometry.value) ** self.geometry.value
            penalty += val
        if self.only_penalty:
            penalty = penalty ** (1./self.geometry.value)

        if _iter % self.prox_period == 0:
            self.prox_point = [torch.cat([x.view(-1) for x in part.parameters()])
                               for part in self.model]

        return loss + self.coeff * penalty


class CatalystRegularizer(EncoderDecoderRegularizer):
    def __init__(self, coeff, geometry=Geometry.l1, model=[], only_penalty=False, prox_period=100, q=0.1):
        super().__init__(geometry, model)
        self.only_penalty = only_penalty
        self.coeff = coeff
        self.prox_point = [torch.cat([x.view(-1) for x in part.parameters()])
                           for part in self.model]
        self.prev_point = [torch.cat([x.view(-1) for x in part.parameters()])
                           for part in self.model]

        self.q = q
        self.alpha = np.sqrt(self.q)

    def __call__(self, loss, _iter):
        params = [torch.cat([x.view(-1) - self.prox_point[i] for i, x in enumerate(part.parameters())])
                  for part in self.model]

        penalty = 0.0
        for param in params:
            val = torch.norm(param, self.geometry.value) ** self.geometry.value
            penalty += val
        if self.only_penalty:
            penalty = penalty ** (1./self.geometry.value)

        if _iter % self.prox_period == 0:
            prev_point = self.prev_point
            self.prev_point = [torch.cat([x.view(-1) for x in part.parameters()])
                               for part in self.model]

            alpha_new = np.min(
                np.roots([1, self.alpha**2 - self.q, -self.alpha**2])).real
            beta = self.alpha * (1 - self.alpha) / (self.alpha**2 + alpha_new)
            self.alpha = alpha_new
            self.prox_point = [torch.cat([x.view(-1) + beta * (x.view(-1) - prev_point[i]) for i, x in enumerate(part.parameters())])
                               for part in self.model]

        return loss + self.coeff * penalty
