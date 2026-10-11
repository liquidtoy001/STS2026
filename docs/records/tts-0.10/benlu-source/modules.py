import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import *
from util import sequence_mask

class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_channels=192,
        latent_channels=192,
        feedforward_channels=448,
        num_heads=2,
        num_layers=6,
        dropout=0.1,
        max_text_length=512,
        padding_id=None,
        kernel_size=3,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.max_text_length = max_text_length

        # Use None if token 0 is also a valid VITS blank token.
        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_channels,
            padding_idx=padding_id,
        )
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=hidden_channels ** -0.5,
        )

        if padding_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[padding_id].zero_()

        self.position_embedding = nn.Embedding(
            max_text_length,
            hidden_channels,
        )
        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=hidden_channels ** -0.5,
        )

        # Each block is initialized independently.
        self.layers = nn.ModuleList([
            ConvTransformerBlock(
                hidden_channels=hidden_channels,
                feedforward_channels=feedforward_channels,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.statistics_projection = nn.Linear(
            hidden_channels,
            latent_channels * 2,
        )

    def forward(self, token_ids, text_lengths):
        # token_ids: [B, T], right-padded
        # text_lengths: [B]
        batch_size, text_length = token_ids.shape

        if text_length > self.max_text_length:
            raise ValueError(
                f"Text length {text_length} exceeds "
                f"max_text_length={self.max_text_length}"
            )

        text_lengths = text_lengths.to(token_ids.device)

        if text_lengths.shape != (batch_size,):
            raise ValueError("text_lengths must have shape [batch]")

        if ((text_lengths < 1) | (text_lengths > text_length)).any():
            raise ValueError("Each text length must be in [1, T]")

        positions = torch.arange(
            text_length,
            device=token_ids.device,
        )

        valid_mask = (
            positions.unsqueeze(0) < text_lengths.unsqueeze(1)
        )
        padding_bth = ~valid_mask.unsqueeze(-1)

        x = self.token_embedding(token_ids)
        x = x * math.sqrt(self.hidden_channels)
        x = x + self.position_embedding(positions).unsqueeze(0)
        x = x.masked_fill(padding_bth, 0.0)

        for layer in self.layers:
            x = layer(x, valid_mask)

        statistics = self.statistics_projection(x)
        statistics = statistics.masked_fill(padding_bth, 0.0)

        prior_mean, prior_log_scale = statistics.chunk(2, dim=-1)

        encoded_text = x.transpose(1, 2)
        prior_mean = prior_mean.transpose(1, 2)
        prior_log_scale = prior_log_scale.transpose(1, 2)
        text_mask = valid_mask.unsqueeze(1).to(x.dtype)

        return encoded_text, prior_mean, prior_log_scale, text_mask

class PosteriorEncoder(nn.Module):
    """
    Convert a spectrogram into a posterior latent variable.

    Input
    -----
    spectrogram:
        [batch, spec_channels, time]

    spectrogram_lengths:
        [batch]

    speaker_condition:
        Optional speaker embedding:
        [batch, condition_channels, 1]

    Output
    ------
    z:
        [batch, latent_channels, time]

    posterior_mean:
        [batch, latent_channels, time]

    posterior_log_scale:
        Log standard deviation:
        [batch, latent_channels, time]

    spectrogram_mask:
        [batch, 1, time]
    """

    def __init__(
        self,
        spec_channels,
        latent_channels=192,
        hidden_channels=192,
        stem_channels=256,
        kernel_size=3,
        dilations=(
            1, 2, 4, 8,
            1, 2, 4, 8,
        ),
        dropout=0.1,
        condition_channels=0,
        minimum_log_scale=-7.0,
        maximum_log_scale=2.0,
    ):
        super().__init__()

        if len(dilations) == 0:
            raise ValueError(
                "At least one dilation is required."
            )

        self.minimum_log_scale = minimum_log_scale
        self.maximum_log_scale = maximum_log_scale

        # --------------------------------------------------
        # Progressive input stem
        #
        # For a 513-channel linear spectrogram:
        # 513 -> 256 -> 192
        # --------------------------------------------------
        self.input_projection = nn.Conv1d(
            spec_channels,
            stem_channels,
            kernel_size=1,
        )

        self.input_refinement = nn.Conv1d(
            stem_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )

        # WaveNet-style residual encoder
        self.blocks = nn.ModuleList([
            GatedResidualBlock(
                channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                condition_channels=condition_channels,
            )
            for dilation in dilations
        ])

        # Original VITS predicts mean and log scale
        # with one projection and then splits them.
        self.statistics_projection = nn.Conv1d(
            hidden_channels,
            2 * latent_channels,
            kernel_size=1,
        )

    def forward(
        self,
        spectrogram,
        spectrogram_lengths,
        speaker_condition=None,
        noise_scale=1.0,
        sample=True,
    ):
        time_length = spectrogram.size(2)

        valid_mask = sequence_mask(
            spectrogram_lengths,
            max_length=time_length,
        )

        spectrogram_mask = (
            valid_mask
            .unsqueeze(1)
            .to(
                device=spectrogram.device,
                dtype=spectrogram.dtype,
            )
        )

        # Progressive feature extraction
        hidden = self.input_projection(spectrogram)
        hidden = F.silu(hidden) * spectrogram_mask

        hidden = self.input_refinement(hidden)
        hidden = F.silu(hidden) * spectrogram_mask
        
        # Residual blocks and skip aggregation
        skip_sum = torch.zeros_like(hidden)

        for block in self.blocks:
            hidden, skip = block(
                hidden,
                spectrogram_mask,
                speaker_condition,
            )
            skip_sum = skip_sum + skip

        # Keep activation magnitude reasonably stable
        hidden = (skip_sum / math.sqrt(len(self.blocks)))
        hidden = F.silu(hidden) * spectrogram_mask
        
        # Posterior distribution
        statistics = (
            self.statistics_projection(hidden)
            * spectrogram_mask
        )
        posterior_mean, posterior_log_scale = statistics.chunk(2, dim=1)

        # Prevent exp(log_scale) from overflowing,
        # particularly during early AMP training.
        posterior_log_scale = (
            posterior_log_scale.clamp(
                min=self.minimum_log_scale,
                max=self.maximum_log_scale,
            )
            * spectrogram_mask
        )

        # Reparameterization
        if sample:
            noise = torch.randn_like(posterior_mean)
            posterior_std = torch.exp(posterior_log_scale)

            z = (
                posterior_mean
                + noise_scale
                * noise
                * posterior_std
            )
        else:
            # Useful for deterministic reconstruction
            z = posterior_mean
        z = z * spectrogram_mask

        return (
            z,
            posterior_mean,
            posterior_log_scale,
            spectrogram_mask,
        )

class Decoder(nn.Module):
    """
    Convert a latent representation into a waveform.

    Input
    -----
    z:
        [batch, latent_channels, latent_time]

    speaker_condition:
        Optional speaker embedding:
        [batch, condition_channels, 1]

    Output
    ------
    waveform:
        [batch, 1, latent_time * upsample_factor]
    """

    def __init__(
        self,
        latent_channels=192,
        initial_channels=512,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7),
        resblock_dilations=(1, 3, 5),
        condition_channels=0,
        negative_slope=0.1,
    ):
        super().__init__()
        number_of_stages = len(upsample_rates)

        if (initial_channels % (2 ** number_of_stages) != 0):
            raise ValueError(
                "initial_channels must be divisible "
                "by 2 ** number_of_stages."
            )

        self.negative_slope = negative_slope
        self.upsample_factor = math.prod(upsample_rates)

        # --------------------------------------------------
        # Input projection
        #
        # [B, 192, T] -> [B, 512, T]
        # --------------------------------------------------
        self.input_projection = nn.Conv1d(
            in_channels=latent_channels,
            out_channels=initial_channels,
            kernel_size=7,
            padding=3,
        )

        # Optional speaker conditioning
        if condition_channels > 0:
            self.condition_projection = nn.Conv1d(
                in_channels=condition_channels,
                out_channels=initial_channels,
                kernel_size=1,
            )
        else:
            self.condition_projection = None

        # Upsampling layers and residual blocks
        self.upsample_layers = nn.ModuleList()
        self.resblock_stages = nn.ModuleList()

        current_channels = initial_channels

        for (upsample_rate, upsample_kernel_size) in zip(upsample_rates, upsample_kernel_sizes):
            next_channels = current_channels // 2
    
            # This padding makes the output length
            # exactly input_length * upsample_rate.
            padding_difference = upsample_kernel_size - upsample_rate
    
            if padding_difference % 2 != 0:
                raise ValueError(
                    "upsample_kernel_size - "
                    "upsample_rate must be even."
                )
            upsample_padding = padding_difference // 2
            
            self.upsample_layers.append(
                nn.ConvTranspose1d(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    kernel_size=(
                        upsample_kernel_size
                    ),
                    stride=upsample_rate,
                    padding=upsample_padding,
                )
            )

            # Multiple receptive fields:
            # kernel sizes 3, 7 and 11 (11 is not useing) operate
            # on the same upsampled features.
            stage_resblocks = nn.ModuleList([
                ResBlock(
                    channels=next_channels,
                    kernel_size=kernel_size,
                    dilations=resblock_dilations,
                    negative_slope=negative_slope,
                )
                for kernel_size
                in resblock_kernel_sizes
            ])

            self.resblock_stages.append(
                stage_resblocks
            )
            current_channels = next_channels

        # --------------------------------------------------
        # Output projection
        #
        # [B, 32, T * 256] -> [B, 1, T * 256]
        # --------------------------------------------------
        self.output_projection = nn.Conv1d(
            in_channels=current_channels,
            out_channels=1,
            kernel_size=7,
            padding=3,
            bias=False,
        )

    def forward(self, z, speaker_condition=None):
        x = self.input_projection(z)

        # Add target speaker information.
        if speaker_condition is not None:
            if self.condition_projection is None:
                raise ValueError(
                    "A speaker condition was provided, "
                    "but condition_channels=0."
                )

            x = x + self.condition_projection(speaker_condition)

        # Progressive waveform upsampling
        for (upsample_layer, stage_resblocks) in zip(self.upsample_layers, self.resblock_stages):
            x = F.leaky_relu(x, negative_slope=(self.negative_slope))
            x = upsample_layer(x)

            # Multi-receptive-field fusion
            stage_output = None

            for resblock in stage_resblocks:
                block_output = resblock(x)

                if stage_output is None:
                    stage_output = block_output
                else:
                    stage_output = stage_output + block_output 

            x = stage_output / len(stage_resblocks) 

        # Generate waveform
        x = F.leaky_relu(
            x,
            negative_slope=self.negative_slope,
        )
        x = self.output_projection(x)
        waveform = torch.tanh(x)
        return waveform



class DurationPredictor(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels=256,
        kernel_size=3,
        dropout=0.5,
        gin_channels=0
    ):
        super().__init__()

        self.conv_block_1 = DurationConvBlock(
            input_channels,
            hidden_channels,
            kernel_size,
            dropout
        )

        self.conv_block_2 = DurationConvBlock(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dropout
        )

        # Produce one duration value for each text token
        self.output_projection = nn.Conv1d(
            hidden_channels,
            out_channels=1,
            kernel_size=1
        )

        if gin_channels > 0:
            self.condition_projection = nn.Conv1d(
                gin_channels,
                input_channels,
                kernel_size=1
            )
        else:
            self.condition_projection = None

    def forward(self, x, mask, g=None):
        # Prevent duration loss from updating the text encoder
        x = x.detach()

        if g is not None:
            if self.condition_projection is None:
                raise ValueError(
                    "gin_channels must be greater than 0 when using g."
                )

            g = g.detach()
            x = x + self.condition_projection(g)

        x = self.conv_block_1(x, mask)
        x = self.conv_block_2(x, mask)

        log_duration = self.output_projection(x)

        return log_duration * mask