import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FusionGate(nn.Module):
    def __init__(self, input_dim):
        super(FusionGate, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, features]
        gate = self.sigmoid(self.linear(x))
        return x * gate

class MultimodalTCN(nn.Module):
    def __init__(self, price_input_size, macro_input_size, text_input_size, 
                 num_channels, kernel_size=2, dropout=0.2, output_quantiles=[0.1, 0.5, 0.9]):
        super(MultimodalTCN, self).__init__()
        
        self.price_encoder = TemporalConvNet(price_input_size, num_channels, kernel_size, dropout)
        
        # Macro and Text encoders (Simple MLPs or Linear projections)
        # Assuming we feed the last time step's macro/text or a window?
        # If we feed a window, we can use TCN for them too.
        # For simplicity, let's assume we project them to a common dimension and concat.
        
        # We'll use TCN for all if they are time-series.
        # But text embeddings are high dimensional (768).
        # Let's project text down first.
        self.text_projection = nn.Linear(text_input_size, 64)
        self.macro_projection = nn.Linear(macro_input_size, 32)
        
        # After projection, we might want to process them temporally or just use the current step.
        # Let's assume the input x contains [Price, Macro, Text] concatenated along feature dim?
        # Or we pass them separately.
        # The user asked for "Three encoders".
        
        # Let's assume we pass 3 tensors: Price [B, C_p, L], Macro [B, C_m, L], Text [B, C_t, L]
        
        self.macro_encoder = TemporalConvNet(32, [32]*len(num_channels), kernel_size, dropout)
        self.text_encoder = TemporalConvNet(64, [64]*len(num_channels), kernel_size, dropout)
        
        # Fusion
        # We take the last time step output from each encoder
        tcn_out_dim = num_channels[-1]
        macro_out_dim = 32
        text_out_dim = 64
        
        fusion_input_dim = tcn_out_dim + macro_out_dim + text_out_dim
        self.fusion_gate = FusionGate(fusion_input_dim)
        
        # Heads
        self.quantile_head = nn.Linear(fusion_input_dim, len(output_quantiles))
        self.prob_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, price, macro, text):
        # price: [B, C_p, L]
        # macro: [B, C_m, L]
        # text: [B, C_t, L]
        
        # Project text and macro
        # Need to transpose for Linear: [B, L, C] -> [B, L, New_C] -> [B, New_C, L]
        text_proj = self.text_projection(text.transpose(1, 2)).transpose(1, 2)
        macro_proj = self.macro_projection(macro.transpose(1, 2)).transpose(1, 2)
        
        # Encode
        p_out = self.price_encoder(price) # [B, C_out, L]
        m_out = self.macro_encoder(macro_proj) # [B, C_out_m, L]
        t_out = self.text_encoder(text_proj) # [B, C_out_t, L]
        
        # Take last time step
        p_last = p_out[:, :, -1]
        m_last = m_out[:, :, -1]
        t_last = t_out[:, :, -1]
        
        # Concat
        combined = torch.cat([p_last, m_last, t_last], dim=1)
        
        # Fusion
        fused = self.fusion_gate(combined)
        
        # Heads
        quantiles = self.quantile_head(fused)
        prob = self.prob_head(fused)
        
        return quantiles, prob
