class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvNextBlock, self).__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pwconv1 = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1)
        self.pwconv2 = nn.Conv2d(4 * out_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.pwconv2d(out)
        out += residual
        return out

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.block1 = ConvNextBlock(32, 64)
        self.block2 = ConvNextBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc(x)
        return x

model = ConvNeXt(num_classes=10)

        