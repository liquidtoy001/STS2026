import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        dilations=(1, 3, 5),
        negative_slope=0.1,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd."
            )
        self.negative_slope = negative_slope

        # First convolution in each residual pair:
        # uses different dilation values.
        self.dilated_convs = nn.ModuleList()

        # Second convolution:
        # uses dilation=1 to refine local details.
        self.refinement_convs = nn.ModuleList()

        for dilation in dilations:
            dilated_padding = (dilation * (kernel_size - 1) // 2)

            self.dilated_convs.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=dilated_padding,
                ) 
            )

            self.refinement_convs.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=1,
                    padding=(
                        kernel_size - 1
                    ) // 2,
                )
            )

    def forward(self, x):
        for dilated_conv, refinement_conv in zip(
            self.dilated_convs,
            self.refinement_convs,
        ):
            residual = x
            hidden = F.leaky_relu(x, negative_slope=self.negative_slope)
            hidden = dilated_conv(hidden)

            hidden = F.leaky_relu(hidden, negative_slope=self.negative_slope)
            hidden = refinement_conv(hidden)
            
            x = residual + hidden
        return x
    
class GatedResidualBlock(nn.Module):
    """
    WaveNet-style non-causal residual block.

    Input
    -----
    x:
        [batch, channels, time]

    mask:
        [batch, 1, time]

    condition:
        Optional speaker condition:
        [batch, condition_channels, 1]
        or
        [batch, condition_channels, time]
    """

    def __init__(
        self,
        channels,
        kernel_size,
        dilation,
        dropout=0.1,
        condition_channels=0,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd."
            )

        padding = (
            dilation
            * (kernel_size - 1)
            // 2
        )

        # Produce 2 * channels:
        # one half for filter values,
        # one half for gate values.
        self.temporal_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        
        # Optional speaker conditioning
        if condition_channels > 0:
            self.condition_projection = nn.Conv1d(
                in_channels=condition_channels,
                out_channels=2 * channels,
                kernel_size=1,
            )
            
        else:
            self.condition_projection = None

        # Produce residual and skip outputs
        self.residual_skip_projection = nn.Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, condition=None):
        gate_input = self.temporal_conv(
            x * mask
        )

        # Add optional speaker condition
        if condition is not None:
            if self.condition_projection is None:
                raise ValueError(
                    "A condition was provided, but "
                    "condition_channels=0."
                )

            gate_input = (
                gate_input
                + self.condition_projection(
                    condition
                )
            )

        # Gated activation
        filter_values, gate_values = gate_input.chunk(2, dim=1)

        hidden = (
            torch.tanh(filter_values)
            * torch.sigmoid(gate_values)
        )

        hidden = self.dropout(hidden)
        hidden = hidden * mask

        # Residual and skip connections
        residual, skip = (
            self.residual_skip_projection(
                hidden
            ).chunk(2, dim=1)
        )
        x = (x + residual) * mask
        skip = skip * mask
        return x, skip


class ConvTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_channels=192,
        feedforward_channels=448,
        num_heads=2,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.conv1 = nn.Conv1d(
            hidden_channels,
            feedforward_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            feedforward_channels,
            hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

        self.activation = nn.ReLU()
        self.ffn_dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

    def forward(self, x, valid_mask):
        # x: [B, T, H]
        # valid_mask: [B, T], True = valid token
        padding_mask = ~valid_mask
        padding_bth = padding_mask.unsqueeze(-1)

        x = x.masked_fill(padding_bth, 0.0)

        attention_output, _ = self.attention(
            x, x, x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        x = self.norm1(
            x + self.attention_dropout(attention_output)
        )
        x = x.masked_fill(padding_bth, 0.0)

        # Convolutional FFN: [B, H, T]
        padding_bct = padding_mask.unsqueeze(1)
        y = x.transpose(1, 2)

        y = self.conv1(y)
        y = self.activation(y)
        y = self.ffn_dropout(y)

        # Remove padding activations before the next convolution.
        y = y.masked_fill(padding_bct, 0.0)
        y = self.conv2(y)
        y = y.masked_fill(padding_bct, 0.0)

        y = y.transpose(1, 2)

        x = self.norm2(x + self.output_dropout(y))
        return x.masked_fill(padding_bth, 0.0)

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
    
class DurationConvBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        dropout
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=kernel_size // 2
        )

        self.norm = ChannelLayerNorm(output_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.conv(x * mask)
        x = F.relu(x)
        x = self.norm(x)
        x = self.dropout(x)

        return x * mask