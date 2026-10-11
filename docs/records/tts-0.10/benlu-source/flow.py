import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelLayerNorm(nn.Module):
    """LayerNorm for tensors with shape [batch, channels, time]."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        x = self.norm(x)

        # [B, T, C] -> [B, C, T]
        return x.transpose(1, 2)

class ResidualConvLayer(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        dilation,
        dropout=0.0
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.norm = ChannelLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x

        x = self.conv(x * mask)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Residual connection
        return (residual + x) * mask

class CouplingNetwork(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size=5,
        dilation_rate=1,
        n_layers=4,
        gin_channels=0
    ):
        super().__init__()

        self.input_projection = nn.Conv1d(
            input_channels,
            hidden_channels,
            kernel_size=1
        )

        self.residual_layers = nn.ModuleList([
            ResidualConvLayer(
                channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation_rate ** layer_index
            )
            for layer_index in range(n_layers)
        ])

        self.output_projection = nn.Conv1d(
            hidden_channels,
            input_channels,
            kernel_size=1
        )

        # Speaker embedding projection
        if gin_channels > 0:
            self.condition_projection = nn.Conv1d(
                gin_channels,
                hidden_channels,
                kernel_size=1
            )
        else:
            self.condition_projection = None

        # Make the coupling layer start as an identity function
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x, mask, g=None):
        x = self.input_projection(x * mask)

        if g is not None:
            if self.condition_projection is None:
                raise ValueError(
                    "gin_channels must be greater than 0 when using g."
                )

            # g is normally [B, gin_channels, 1]
            # It broadcasts across the time dimension
            x = x + self.condition_projection(g)

        x = x * mask

        for layer in self.residual_layers:
            x = layer(x, mask)

        shift = self.output_projection(x)

        return shift * mask

class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size=5,
        dilation_rate=1,
        n_layers=4,
        gin_channels=0
    ):
        super().__init__()

        if channels % 2 != 0:
            raise ValueError("channels must be an even number.")

        half_channels = channels // 2

        self.coupling_network = CouplingNetwork(
            input_channels=half_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            n_layers=n_layers,
            gin_channels=gin_channels
        )

    def forward(self, x, mask, g=None, reverse=False):
        # Split channels into two equal parts
        x_keep, x_change = torch.chunk(x, chunks=2, dim=1)

        # Predict the shift using only x_keep
        shift = self.coupling_network(x_keep, mask, g)

        if reverse:
            # Inverse transformation
            x_change = x_change - shift

            x = torch.cat([x_keep, x_change], dim=1)
            return x * mask

        # Forward transformation
        x_change = x_change + shift

        x = torch.cat([x_keep, x_change], dim=1)
        x = x * mask

        # An additive coupling layer has log determinant = 0
        log_determinant = torch.zeros(
            x.size(0),
            device=x.device,
            dtype=x.dtype
        )

        return x, log_determinant

class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size=5,
        dilation_rate=1,
        n_layers=4,
        n_flows=4,
        gin_channels=0
    ):
        super().__init__()

        self.coupling_layers = nn.ModuleList([
            ResidualCouplingLayer(
                channels=channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                n_layers=n_layers,
                gin_channels=gin_channels
            )
            for _ in range(n_flows)
        ])

    def forward(self, x, mask, g=None, reverse=False):
        if not reverse:
            for coupling_layer in self.coupling_layers:
                x, _ = coupling_layer(
                    x,
                    mask,
                    g=g,
                    reverse=False
                )

                # Replace modules.Flip()
                x = torch.flip(x, dims=[1])

        else:
            # Undo operations in the opposite order
            for coupling_layer in reversed(self.coupling_layers):
                x = torch.flip(x, dims=[1])

                x = coupling_layer(
                    x,
                    mask,
                    g=g,
                    reverse=True
                )
        return x