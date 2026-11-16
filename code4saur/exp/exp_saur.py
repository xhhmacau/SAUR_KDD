import torch
from exp.exp_online import Exp_Online


class SAURState:

    def __init__(self, channels, horizon, device, alpha="xxx", beta="xxx",
                 gamma="xxx", kappa="xxx", a="xxx", q="xxx", delta="xxx",
                 epsilon="xxx"):
        controller_values = dict(alpha=alpha, beta=beta, gamma=gamma, kappa=kappa,
                                 a=a, q=q, delta=delta, epsilon=epsilon)
        unset = [name for name, value in controller_values.items()
                 if isinstance(value, str) and value == "xxx"]
        if unset:
            raise ValueError("SAUR controller values must be set explicitly; replace 'xxx' for: "
                             + ", ".join(unset))
        if channels < 1 or horizon < 1:
            raise ValueError("channels and horizon must be positive")
        if not 0 < beta < alpha < 1:
            raise ValueError("SAUR requires 0 < beta < alpha < 1")
        if not 0 < gamma < 1 or not 0 < q < 1:
            raise ValueError("gamma and q must lie in (0, 1)")
        if kappa <= 0 or a <= 0 or delta <= 0 or epsilon <= 0:
            raise ValueError("kappa, a, delta and epsilon must be positive")
        self.channels = channels
        self.horizon = horizon
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.q = q

        self.delta = delta
        self.epsilon = epsilon
        positions = torch.arange(1, horizon + 1, dtype=torch.float32, device=device)
        self.omega = torch.softmax(positions / kappa, dim=0)
        self.reset()

    def reset(self):
        self.fast = torch.zeros(self.channels, device=self.device)
        self.slow = torch.zeros(self.channels, device=self.device)
        self.fast_signed = torch.zeros(self.channels, device=self.device)
        self.directional = torch.zeros(self.channels, device=self.device)
        self.persistence = torch.zeros(self.channels, device=self.device)
        self.initialized = False

    @torch.no_grad()
    def update(self, residual):
        residual = residual.detach()
        if residual.ndim == 3 and residual.shape[0] == 1:
            residual = residual[0]
        if residual.shape != (self.horizon, self.channels):
            raise ValueError("residual must have shape [H, N] or [1, H, N]")
        residual = residual.to(device=self.device, dtype=self.fast.dtype)
        magnitude = (residual.abs() * self.omega[:, None]).sum(0)
        signed = (residual * self.omega[:, None]).sum(0)
        if not self.initialized:
            self.fast.copy_(magnitude)
            self.slow.copy_(magnitude)
            self.fast_signed.copy_(signed)
            self.initialized = True
            return torch.ones(self.channels, device=self.device)
        self.fast.mul_(1 - self.alpha).add_(magnitude, alpha=self.alpha)
        self.slow.mul_(1 - self.beta).add_(magnitude, alpha=self.beta)
        self.fast_signed.mul_(1 - self.alpha).add_(signed, alpha=self.alpha)
        severity = (self.fast + self.epsilon) / (self.slow + self.epsilon)
        directional = (self.fast_signed.abs() / (self.fast + self.epsilon)).clamp(0, 1)
        persistence = 1 - torch.exp(-torch.clamp(severity - 1, min=0))
        self.directional.mul_(1 - self.gamma).add_(directional, alpha=self.gamma)
        self.persistence.mul_(1 - self.gamma).add_(persistence, alpha=self.gamma)
        structure = torch.maximum(self.directional, self.persistence)
        routing = torch.tanh(self.a * (structure - self.q)) * torch.log(torch.clamp(severity, min=1))
        raw_weights = routing.clamp(-self.delta, self.delta).exp()

        return (raw_weights / raw_weights.mean()).detach()


class Exp_SAUR(Exp_Online):
    def __init__(self, args):
        super().__init__(args)
        self.state = SAURState(
            args.c_out, args.pred_len, self.device,
            alpha=args.saur_alpha, beta=args.saur_beta, gamma=args.saur_gamma,
            kappa=args.saur_kappa, a=args.saur_a, q=args.saur_q,
            delta=args.saur_delta, epsilon=args.saur_epsilon,
        )

    def _reset_online_state(self):
        self.state.reset()

    def _replay(self, batch, stored_prediction, optimizer):
        target = batch[self.label_position].to(self.device)
        residual = target.detach() - stored_prediction.detach()
        weights = self.state.update(residual)
        optimizer.zero_grad(set_to_none=True)
        prediction = self.forward(batch)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        squared = (prediction - target).square().squeeze(0)
        channel_loss = (squared * self.state.omega[:, None]).sum(0)
        loss = (weights * channel_loss).mean()
        loss.backward()
        optimizer.step()
        return loss.detach()
