import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    raise ImportError("Please install torchdiffeq using: pip install torchdiffeq")


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation to inject Lead context."""
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.fc_gamma = nn.Linear(context_dim, hidden_dim)
        self.fc_beta = nn.Linear(context_dim, hidden_dim)

    def forward(self, h, context):
        gamma = self.fc_gamma(context)
        beta = self.fc_beta(context)
        return (1.0 + gamma) * h + beta


class BaselineODEFunc(nn.Module):
    """Differential Vector Function of pure ODE with FiLM integration."""
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.film = FiLMLayer(context_dim, hidden_dim)
        self._current_context = None

    def set_context(self, context):
        self._current_context = context

    def forward(self, t, h):
        if self._current_context is not None:
            h = self.film(h, self._current_context)
        return self.net(h)


class ContinuousTimeFocalLossWithTimeDecay(nn.Module):
    """Continuous-Time Focal Loss with Time-Decay Weighting for AI CRM."""
    def __init__(self, alpha=0.75, gamma=2.0, decay_lambda=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.decay_lambda = decay_lambda

    def forward(self, probs, targets, event_times, current_time):
        """
        probs: Predictive probability of the model [batch_size]
        targets: Current label (0 = Churn, 1 = Conversion) [batch_size]
        event_times: Time of lead interaction [batch_size]
        current_time: Reference time during evaluation [scalar]
        """
        time_diff = torch.clamp(current_time - event_times, min=0.0)
        time_weights = torch.exp(-self.decay_lambda * time_diff)

        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma

        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        focal_loss = alpha_factor * modulating_factor * bce_loss
        
        weighted_focal_loss = time_weights * focal_loss
        return weighted_focal_loss.mean()


class RedstoneBaselineCRM(nn.Module):
    """Redstone CRM Baseline Architecture"""
    def __init__(self, input_dim, hidden_dim, context_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.ode_func = BaselineODEFunc(hidden_dim, context_dim)
        self.film = FiLMLayer(context_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x0, context, evaluation_times):
        """
        x0: Initial lead embedding [batch_size, input_dim]
        context: Lead metadata context [batch_size, context_dim]
        evaluation_times: The time tensor for which the probability is to be calculated [num_steps]
        """
        h0 = self.encoder(x0)
        h0 = self.film(h0, context)

        self.ode_func.set_context(context)

        h_t_trajectory = odeint(
            self.ode_func, 
            h0, 
            evaluation_times, 
            method='dopri5'
        )

        logits = self.classifier(h_t_trajectory)
        probs = torch.sigmoid(logits).squeeze(-1)
        
        return probs
