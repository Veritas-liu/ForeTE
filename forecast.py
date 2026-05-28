import torch
import torch.nn as nn


class ExpSmoothingForecast(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, tm_hist: torch.Tensor) -> torch.Tensor:
        if tm_hist.ndim != 3:
            raise ValueError("tm_hist must be a 3D tensor of shape [batch, hist_len, num_paths]")
        hist = tm_hist.flip(dims=[1])
        hist_len = hist.size(1)
        weights = self.alpha * (1.0 - self.alpha) ** torch.arange(hist_len, device=hist.device, dtype=hist.dtype)
        weights = weights / weights.sum()
        return torch.sum(hist * weights.view(1, hist_len, 1), dim=1)


class TrainableExpSmoothingForecast(nn.Module):
    def __init__(self, hist_len: int, tm_shape, alpha: float = 0.5):
        super().__init__()
        self.hist_len = hist_len
        self.tm_shape = tuple(tm_shape)
        init_weights = alpha * (1.0 - alpha) ** torch.arange(hist_len, dtype=torch.float32)
        init_weights = init_weights / init_weights.sum()
        self.weights = nn.Parameter(init_weights)
        self.bias = nn.Parameter(torch.zeros(self.tm_shape, dtype=torch.float32))

    def forward(self, tm_hist: torch.Tensor) -> torch.Tensor:
        if tm_hist.ndim != 3:
            raise ValueError("tm_hist must be a 3D tensor of shape [batch, hist_len, num_paths]")
        hist = tm_hist.flip(dims=[1])
        weights = self.weights.to(tm_hist.dtype)
        weights = weights / weights.sum()
        tm_pred = torch.sum(hist * weights.view(1, self.hist_len, 1), dim=1)
        bias = torch.sigmoid(self.bias.to(tm_pred.dtype)) * 2
        return tm_pred * bias


class ForecastNet(nn.Module):
    def __init__(self, forecast_type: str = "exp", hist_len: int = None, tm_shape=None, num_paths: int = None, hidden_dim: int = 512, hidden_layers: int = 2, alpha: float = 0.5):
        super().__init__()
        forecast_type = forecast_type.lower()
        if forecast_type == "exp":
            self.model = ExpSmoothingForecast(alpha=alpha)
        elif forecast_type == "dnn":
            if hist_len is None or tm_shape is None:
                raise ValueError("DNN forecast requires hist_len and tm_shape")
            self.model = TrainableExpSmoothingForecast(hist_len, tm_shape, alpha=alpha)
        else:
            raise ValueError(f"Unsupported forecast_type: {forecast_type}")

    def forward(self, tm_hist: torch.Tensor) -> torch.Tensor:
        return self.model(tm_hist)
