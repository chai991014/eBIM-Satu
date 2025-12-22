import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = torch.relu(self.fc1(x[:, -1, :]))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.output_layer(x)
        return x


class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_matrix):
        super(STGCNLayer, self).__init__()
        self.A = nn.Parameter(torch.from_numpy(adjacency_matrix).float(), requires_grad=False)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        x = self.gcn(x)
        x = self.tcn(x)
        return x


class STGCNModel(nn.Module):
    def __init__(self, num_classes, adjacency_matrix):
        super(STGCNModel, self).__init__()
        self.layer1 = STGCNLayer(3, 64, adjacency_matrix)
        self.layer2 = STGCNLayer(64, 128, adjacency_matrix)
        self.layer3 = STGCNLayer(128, 256, adjacency_matrix)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rel_channels = max(1, in_channels // rel_reduction)

        self.conv_1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv_3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv_4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, A):
        # x: (N, C, T, V), A: (V, V)
        x1 = self.conv_1(x).mean(-2)  # (N, C_rel, V)
        x2 = self.conv_2(x).mean(-2)  # (N, C_rel, V)

        # Channel-wise topology refinement
        x1 = x1.unsqueeze(3)  # (N, C_rel, V, 1)
        x2 = x2.unsqueeze(2)  # (N, C_rel, 1, V)
        a1 = self.tanh(x1 - x2)  # (N, C_rel, V, V)
        a2 = self.conv_4(a1)  # (N, C_out, V, V)

        # Combine with static adjacency matrix
        alpha = a2 + A.unsqueeze(0).unsqueeze(0)  # (N, C_out, V, V)
        x = self.conv_3(x)  # (N, C_out, T, V)

        # alpha is (N, C, V, W), unsqueeze it to (N, C, 1, V, W)
        alpha = alpha.unsqueeze(2)

        # Final aggregation
        # Now einsum equation 'nctvw,nctw->nctv' matches:
        # alpha: (n)batch, (c)channel, (t)1, (v)node, (w)node
        # x:     (n)batch, (c)channel, (t)time, (w)node
        out = torch.einsum('nctvw,nctw->nctv', alpha, x)
        return out


class CTRGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_matrix):
        super(CTRGCNLayer, self).__init__()
        self.A = nn.Parameter(torch.from_numpy(adjacency_matrix).float(), requires_grad=False)
        self.gc = CTRGC(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.gc(x, self.A)
        x = self.tcn(x)
        return x


class CTRGCNModel(nn.Module):
    def __init__(self, num_classes, adjacency_matrix):
        super(CTRGCNModel, self).__init__()
        self.layer1 = CTRGCNLayer(3, 64, adjacency_matrix)
        self.layer2 = CTRGCNLayer(64, 128, adjacency_matrix)
        self.layer3 = CTRGCNLayer(128, 256, adjacency_matrix)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return self.fc(x)
