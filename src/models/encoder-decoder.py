class EncoderBlock(nn.Module):
    # Consists of Conv -> ReLU -> MaxPool
    def __init__(self, in_channels, out_channels, layers=2, sampling_factor=2, padding="same"):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Conv2d(in_channels, out_channels, 3, 1, padding=padding))
        self.encoder.append(nn.ReLU())
        for _ in range(layers - 1):
            self.encoder.append(nn.Conv2d(out_channels, out_channels, 3, 1, padding=padding))
            self.encoder.append(nn.ReLU())
        self.encoder.append(nn.MaxPool2d(sampling_factor))
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        mp_out = self.mp(x)
        return mp_out, x
    
class DecoderBlock(nn.Module):
    # Consists of 2x2 transposed convolution -> ReLU
    def __init__(self, in_chans, out_chans, layers=2, skip_connection=True, sampling_factor=2, padding="same"):
        super().__init__()
        skip_factor = 1 if skip_connection else 2
        self.decoder = nn.ModuleList()
        self.tconv = nn.ConvTranspose2d(in_chans, in_chans//2, sampling_factor, sampling_factor)

        self.decoder.append(nn.Conv2d(out_chans, out_chans, 3, 1, padding=padding))
        self.decoder.append(nn.ReLU())

        self.skip_connection = skip_connection
        self.padding = padding

    def forward(self, x, enc_features=None):
        x = self.tconv(x)
        if self.skip_connection:
            if self.padding != "same":
                # Crop the enc_features to the same size as input
                w = x.size(-1)
                c = (enc_features.size(-1) - w) // 2
                enc_features = enc_features[:,:,c:c+w,c:c+w]
            x = torch.cat((enc_features, x), dim=1)
        for dec in self.decoder:
            x = dec(x)
        return x