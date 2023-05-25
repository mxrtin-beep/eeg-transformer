import torch
import torch.nn as nn


# CONV BLOCK
IN_T = 1155

KERNEL_CB_1 = 64
PADDING_CB_1 = 31
KERNEL_CB_2 = 22

HIDDEN_SIZE_CB = 16

TIME_DIVIDER = 140

OUT_T = 20


WINDOW_LEN_CT = 19
CONTINUOUS_TRANSFORMER_LEN = OUT_T - WINDOW_LEN_CT + 1


n_classes = 4


# in channels: 1
# out channels: 8
# (1, 1, 22, 1125)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, KERNEL_CB_1), stride=1, padding=(0, PADDING_CB_1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.depthwise_conv = nn.Conv2d(out_channels, HIDDEN_SIZE_CB, kernel_size=(KERNEL_CB_2, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(HIDDEN_SIZE_CB)
        self.elu = nn.ELU()

        kernel_cb_3 = int(IN_T / TIME_DIVIDER)
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 8), stride=8)
        self.dropout1 = nn.Dropout(dropout)

        # (1, 16, 1, 140)

        self.conv2 = nn.Conv2d(HIDDEN_SIZE_CB, HIDDEN_SIZE_CB, kernel_size=(1, 8), padding=(0,4))

        out_kernel = int((TIME_DIVIDER + 1)/OUT_T)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, out_kernel))
        
    def forward(self, x):
        #print(x.shape)              # Should be (96, 1, 25, 1125)
        x = self.conv(x)
        #print(x.shape)  # Should be (1, 8, 25, 1124)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        #print(x.shape)  # Should be (1, 16, 1, 1124) or (1, 16, 4, 1124) 
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avg_pool(x)    # Divides by 8
        x = self.dropout1(x)
        #print(x.shape)  # Should be (1, 16, 1, 140)

        x = self.conv2(x)
        #print(x.shape)  # Should be (1, 16, 1, 141)

        x = self.avg_pool2(x)
        #print(x.shape)  # Should be (1, 16, 1, 20)

        # Break it up into  (1, 16, 1, 1) x 20

        return x


class TemporalConvNet(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=1, dilation=1, depth=2, dropout=0.1):
        super(TemporalConvNet, self).__init__()

        layer_1 = []
        in_channels = input_size

        # in: 16
        # out: 16
        for i in range(depth-1): # 0, 1
            #dilation_size = dilation ** i
            #padding = dilation_size * (kernel_size - 1) // 2
            # kernel_size: 19-4+1 = 16; 18-4+1 = 15
            # 19 = WINDOW_LEN
            # WINDOW_LEN - KERNEL_SIZE + 1 = 16
            # KERNEL_SIZE = WINDOW_LEN + 1 - 16
            kernel_size_tcn = WINDOW_LEN_CT + 1 - 16
            layer_1.append(nn.Conv1d(in_channels, output_size, kernel_size=kernel_size_tcn, stride=1, padding=0,
                                 dilation=1))

            layer_1.append(nn.BatchNorm1d(output_size))
            layer_1.append(nn.ELU())
            layer_1.append(nn.Dropout(dropout))
            in_channels = output_size
            #print('OS', output_size)

        self.conv1 = nn.Sequential(*layer_1)

        self.residual1 = nn.Sequential(
            nn.BatchNorm1d(output_size),
            nn.ELU()
        )



        layer_2 = []
        layer_2.append(nn.Conv1d(in_channels, output_size, kernel_size=4, stride=4, padding=0,
                                 dilation=1))
        layer_2.append(nn.ELU())
        layer_2.append(nn.Dropout(dropout))

        self.conv2 = nn.Sequential(*layer_2)

        self.residual2 = nn.Sequential(
            nn.AvgPool1d(4, 4),
            nn.BatchNorm1d(output_size),
            nn.ELU()
        )

        layer_3 = []
        layer_3.append(nn.Conv1d(in_channels, output_size, kernel_size=4, stride=4, padding=0))
        layer_3.append(nn.ELU())

        self.conv3 = nn.Sequential(*layer_3)

        self.residual3 = nn.Sequential(
            nn.AvgPool1d(4, 4),
            #nn.BatchNorm1d(output_size),
            nn.ELU()
        )
        

    def forward(self, x):
        #print('x', x.shape) # (1, 16, 19)
        B, N_ch, T = x.shape
        diff = T - 16
        if diff <= 0:
            residual_1 = self.residual1(x[:, :, :]) # not sure if the right side
        else:
            residual_1 = self.residual1(x[:, :, :-diff])

        out_1 = self.conv1(x)
        #out2 = self.network2(out1)

        #print('residual_1', residual_1.shape)   # (1, 16, 16)
        #print('out_1', out_1.shape)           # (1, 16, 15), should be (1, 16, 16)

        combined_1 = torch.add(residual_1, out_1)

        out_2 = self.conv2(combined_1)
        #print('out_2: ', out_2.shape)

        residual_2 = self.residual2(combined_1)
        #print('residual_2', residual_2.shape)

        combined_2 = torch.add(out_2, residual_2)
        #print(combined_2.shape)  # (1, 16, 4)

        out_3 = self.conv3(combined_2)

        residual_3 = self.residual3(combined_2)

        #print('out3', out_3.shape)
        #print('rout3', residual_3.shape)
        combined_3 = torch.add(out_3, residual_3)

        #print('combined_3', combined_3.shape)

        return combined_3




class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.layer_norm1(x)
        linear_output = self.linear2(self.dropout2(nn.functional.relu(self.linear1(x))))
        x = x + self.dropout1(linear_output)
        x = self.layer_norm2(x)
        return x

class ContinuousTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, d_model, nhead, dim_feedforward, dropout, n_classes=4):
        super(ContinuousTransformer, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout)
        self.transformer_block1 = TransformerBlock(d_model, nhead, dim_feedforward, dropout)
        self.transformer_block2 = TransformerBlock(d_model, nhead, dim_feedforward, dropout)
        #self.tcn_block = TCN_Block(16, 1)
        self.temporal_conv_net = TemporalConvNet(16, 16)
        #self.temp_conv = nn.Conv1d(in_channels=d_model, out_channels=out_channels, kernel_size=3, padding=1)
        self.fc_out = nn.Linear(16*CONTINUOUS_TRANSFORMER_LEN, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x = x.squeeze(1)
        #print('a', x.shape) # (96, 1, 25, 1125)
        x = self.conv_block(x)
        #print('b', x.shape) # (96, 16, 1, 20)

        b, F2, N_Ch, T = x.shape

        seq = []
        for i in range(T):
            seq.append(x[:, :, :, i]) # (1, 16, 1) (B, F2, N_ch)

        #print('seq', len(seq))

        #print('c', seq[0].shape)
        # 20 items in seq, each (1, 16, 1) (B, F2, N_ch)

        # OUT_T = 20

        # wl 8, ss: (1, 11, 16); wl 16, ss: (1, 19, 16)
        output = []


        for i in range(len(seq) - WINDOW_LEN_CT + 1):
            sub_section = torch.stack(seq[i:i+WINDOW_LEN_CT]).squeeze(-1).permute(1, 0, 2)
            
            #print('ss', sub_section.shape) # (1, 16, 16) (B, T, F2)

            attention_block = self.transformer_block1(sub_section)
            attention_block = self.transformer_block2(attention_block)
            #print('d', attention_block.shape) # (1, 16, 16)

            attention_block = attention_block.permute(0, 2, 1)
            #print('e', attention_block.shape)

            # (1, 16, 16) (B, F2, T) #(B, 16, 19)
            # In: 19, out: 1
            x = self.temporal_conv_net(attention_block)

            output.append(x)
            #print('f', x.shape) # (B, 16, 1)
        

        #print('output', len(output)) # 2
        #print(output[0].shape) # (3 by (96, 16, 1))

        all_out = torch.cat(output, 1).squeeze(-1) # (1, 32), (B, N_features)

        #print('ao', all_out.shape) # (96, 48) instead of (96, 32)
        
        logits = self.fc_out(all_out) # (1, 4)

        #print('l', logits.shape)
        predictions = self.softmax(logits)
        #print(predictions.shape)
        return predictions





