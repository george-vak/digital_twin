import torch
from torch import nn

from model import GRUModel


class Transform(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        config = checkpoint["config"]
        columns = [f"CV[{i}]" for i in range(41)] + [f"MV[{i}]" for i in range(12)]
        norm = checkpoint["normalization"]
        self.window = config["window"]
        self.frequency = config["frequency_seconds"]
        self.register_buffer("mean", torch.tensor([norm["mean"][c] for c in columns], dtype=torch.float32))
        self.register_buffer("scale", torch.tensor([norm["scale"][c] for c in columns], dtype=torch.float32))

    def history(self, levels):
        return (levels[..., 1:, :] - levels[..., :-1, :] - self.mean) / self.scale

    def cv_target(self, last, following):
        return (following[..., :41] - last[..., :41] - self.mean[:41]) / self.scale[:41]

    def cv_levels(self, last, delta):
        return last[..., :41] + delta * self.scale[:41] + self.mean[:41]


def source_model(checkpoint):
    model = GRUModel(53, checkpoint["config"]["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


class Actor(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        source = source_model(checkpoint)
        self.gru = source.gru
        self.head = nn.Linear(source.head.in_features, 12)
        with torch.no_grad():
            self.head.weight.copy_(source.head.weight[41:])
            self.head.bias.copy_(source.head.bias[41:])

    def forward(self, history, last, transform):
        _, hidden = self.gru(history)
        delta = self.head(hidden[-1])
        return (last[..., 41:] + delta * transform.scale[41:] + transform.mean[41:]).clamp(0, 100)


class Dynamics(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        source = source_model(checkpoint)
        self.gru = source.gru
        self.head = nn.Linear(source.head.in_features + 12, 41)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[:, :-12].copy_(source.head.weight[:41])
            self.head.bias.copy_(source.head.bias[:41])

    def forward(self, history, last, action, transform):
        _, hidden = self.gru(history)
        action_delta = (action - last[..., 41:] - transform.mean[41:]) / transform.scale[41:]
        return self.head(torch.cat((hidden[-1], action_delta), dim=-1))


def cost(cv, frequency_seconds=60):
    component = cv.new_tensor([2.206, 0, 6.177, 22.06, 14.56, 17.89, 30.44, 22.94])
    purge = (cv[..., 28:36] / 100 * component).sum(-1) * 44.79 * cv[..., 9]
    product = (cv[..., 36:39] / 100 * component[3:6]).sum(-1) * 9.20697 * cv[..., 16]
    return (purge + product + .0536 * cv[..., 19] + .0318 * cv[..., 18]) * (frequency_seconds / 60)


def actor_update(actor, dynamics, transform, optimizer, history, last):

    optimizer.zero_grad(set_to_none=True)
    dynamics.requires_grad_(False)
    try:
        action = actor(history, last, transform)
        prediction = transform.cv_levels(last, dynamics(history, last, action, transform))
        loss = cost(prediction, transform.frequency).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite actor loss")
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    finally:
        dynamics.requires_grad_(True)
    return float(loss.detach())
