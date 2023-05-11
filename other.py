import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 64), stride=1, padding=(0, 31), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.depthwise_conv = nn.Conv2d(8, 16, kernel_size=(22, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.elu = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 8), stride=8)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=(1, 8), padding=(0,4))
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, 7))
        
    def forward(self, x):
                        # Should be (1, 1, 22, 1125)
        x = self.conv(x)
        print(x.shape)  # Should be (1, 8, 22, 1124)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        print(x.shape)  # Should be (1, 16, 1, 1124)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avg_pool(x)
        x = self.dropout1(x)
        print(x.shape)  # Should be (1, 16, 1, 140)

        x = self.conv2(x)
        print(x.shape)  # Should be (1, 16, 1, 141)

        x = self.avg_pool2(x)
        print(x.shape)  # Should be (1, 16, 1, 20)

        # Break it up into  (1, 16, 1, 1) x 20

        return x


class TCN_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.padding1 = nn.ConstantPad1d((3, 0), 0.0)
        self.conv1 = nn.Conv1d(16, 16, kernel_size=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.1)

        #self.conv2 = nn.Conv1d()
        #self.bn2 = nn.BatchNorm1d()
        #self.elu2 = nn.ELU()
        #self.dropout2 = nn.Dropout()

    def forward(self, x):

        # In: (1, 16, 16) (B, F2, T)
        x = self.padding1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.dropout1(x)
        print('f', x.shape)
        quit()

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=1, dilation=1, depth=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()

        layers1 = []
        in_channels = input_size

        for i in range(depth-1): # 0, 1
            #dilation_size = dilation ** i
            #padding = dilation_size * (kernel_size - 1) // 2
            layers1.append(nn.Conv1d(in_channels, output_size, kernel_size=4, stride=1, padding=0,
                                 dilation=1))

            layers1.append(nn.BatchNorm1d(output_size))
            layers1.append(nn.ReLU())
            layers1.append(nn.Dropout(dropout))
            in_channels = output_size

        self.network1 = nn.Sequential(*layers1)
        layers2 = []
        layers2.append(nn.Conv1d(in_channels, output_size, kernel_size=4, stride=4, padding=0,
                                 dilation=1))
        layers2.append(nn.ReLU())
        layers2.append(nn.Dropout(dropout))

        self.network2 = nn.Sequential(*layers2)

        self.avg_pool = nn.AvgPool1d(4, 4)

        self.last = nn.Conv1d(in_channels, output_size, kernel_size=4, stride=4, padding=0)

        self.avg_pool2 = nn.AvgPool1d(4, 4)
        

    def forward(self, x):
        print('x', x.shape) # (1, 16, 19)
        residual = x[:, :, :-3] # not sure if the right side
        out1 = self.network1(x)
        #out2 = self.network2(out1)

        print('residual', residual.shape)   # (1, 16, 16)
        print('out1', out1.shape)           # (1, 16, 16)
        #print('out2', out2.shape)           # (1, 16, 13)
        residual_out = torch.add(residual, out1)
        print(residual_out.shape)           # (1, 16, 16)
        out2 = self.network2(residual_out)
        print(out2.shape)

        residual2 = self.avg_pool(residual_out)
        print('residual2', residual2.shape)

        residual_out2 = torch.add(out2, residual2)
        print(residual_out2.shape)  # (1, 16, 4)

        out3 = self.last(residual_out2)

        residual_out3 = self.avg_pool2(residual_out2)

        print('out3', out3.shape)
        print('rout3', residual_out3.shape)

        final_out = torch.add(out3, residual_out3)
        quit()
        return out


class TemporalConvNet2(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=1, dilation=1, depth=2, dropout=0.0):
        super(TemporalConvNet2, self).__init__()

        layer_1 = []
        in_channels = input_size

        for i in range(depth-1): # 0, 1
            #dilation_size = dilation ** i
            #padding = dilation_size * (kernel_size - 1) // 2
            layer_1.append(nn.Conv1d(in_channels, output_size, kernel_size=4, stride=1, padding=0,
                                 dilation=1))

            layer_1.append(nn.BatchNorm1d(output_size))
            layer_1.append(nn.ELU())
            layer_1.append(nn.Dropout(dropout))
            in_channels = output_size

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
        print('x', x.shape) # (1, 16, 19)
        residual_1 = self.residual1(x[:, :, :-3]) # not sure if the right side
        out_1 = self.conv1(x)

        #out2 = self.network2(out1)

        print('residual_1', residual_1.shape)   # (1, 16, 16)
        print('out_1', out_1.shape)           # (1, 16, 16)

        combined_1 = torch.add(residual_1, out_1)

        out_2 = self.conv2(combined_1)
        print('out_2: ', out_2.shape)

        residual_2 = self.residual2(combined_1)
        print('residual_2', residual_2.shape)

        combined_2 = torch.add(out_2, residual_2)
        print(combined_2.shape)  # (1, 16, 4)

        out_3 = self.conv3(combined_2)

        residual_3 = self.residual3(combined_2)

        print('out3', out_3.shape)
        print('rout3', residual_3.shape)
        combined_3 = torch.add(out_3, residual_3)

        print('combined_3', combined_3.shape)

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

n_classes = 4
class ContinuousTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, d_model, nhead, dim_feedforward, dropout):
        super(ContinuousTransformer, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.transformer_block1 = TransformerBlock(d_model, nhead, dim_feedforward, dropout)
        self.transformer_block2 = TransformerBlock(d_model, nhead, dim_feedforward, dropout)
        #self.tcn_block = TCN_Block(16, 1)
        self.temporal_conv_net = TemporalConvNet2(16, 16)
        #self.temp_conv = nn.Conv1d(in_channels=d_model, out_channels=out_channels, kernel_size=3, padding=1)
        self.fc_out = nn.Linear(32, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x = x.squeeze(1)
        print('a', x.shape)
        x = self.conv_block(x)
        print('b', x.shape) # (1, 16, 1, 20)

        b, F2, N_Ch, T = x.shape

        seq = []
        for i in range(T):
            seq.append(x[:, :, :, i]) # (1, 16, 1) (B, F2, N_ch)

        print('c', seq[0].shape)
        # 20 items in seq, each (1, 16, 1) (B, F2, N_ch)
        window_len = 16
        output = []
        for i in range(len(seq) - window_len - 3 + 1):
            sub_section = torch.stack(seq[i:i+window_len+3]).squeeze(-1).permute(1, 0, 2)
            
            print(sub_section.shape) # (1, 16, 16) (B, T, F2)

            attention_block = self.transformer_block1(sub_section)
            print('d', attention_block.shape)

            attention_block = attention_block.permute(0, 2, 1)
            print('e', attention_block.shape)

            # (1, 16, 16) (B, F2, T)
            x = self.temporal_conv_net(attention_block)

            output.append(x)
            print('f', x.shape)

        # Length is 2
        print(len(output))
        print(output[0].shape)

        all_out = torch.cat(output, 1).squeeze(-1) # (1, 32), (B, N_features)
        
        logits = self.fc_out(all_out) # (1, 4)
        predictions = self.softmax(logits)
        print(predictions.shape)
        return predictions



model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=4, dim_feedforward=512, dropout=0.1)
input_data = torch.randn(1, 1, 22, 1125) 

out = model(input_data)

print('Out: ', out.shape)
print(out)

