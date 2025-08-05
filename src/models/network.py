import torch
import torch.nn as nn



# Добавим модификации к оригинальной архитектуре. 
# К примеру - padding = 1(в оригинале от паддинга отказались за счет *более правдивой сегментации*)
# Также добавим батчнорм

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        conv_path = self.conv_block(x)
        residual = self.shortcut(x)

        conv_path += residual

        conv_path = self.activation(conv_path)

        return conv_path
        

class RRCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RRCBlock, self).__init__()
    
        self.t = t
    
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            
        self.main_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        recurrent = residual
        
        for i in range(self.t):
            recurrent = self.main_conv(residual + recurrent)
            recurrent = self.bn(recurrent)
            recurrent = self.relu(recurrent)

        return recurrent
    
class AttentionBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, out_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels_x, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        
        psi = self.psi(psi)
        
        return x * psi


class R2UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(R2UNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.down_conv1 = RRCBlock(in_channels, 64)
        self.down_conv2 = RRCBlock(64, 128)
        self.down_conv3 = RRCBlock(128, 256)
        self.down_conv4 = RRCBlock(256, 512)

        # bottleneck
        self.bottleneck = RRCBlock(512, 1024)

        # Decoder        
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_double_conv1 = RRCBlock(1024, 512)
        
        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_double_conv2 = RRCBlock(512, 256)
        
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_double_conv3 = RRCBlock(256, 128)
        
        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_double_conv4 = RRCBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.down_conv1(x)
        p1 = self.pool(e1)

        e2 = self.down_conv2(p1)
        p2 = self.pool(e2)

        e3 = self.down_conv3(p2)
        p3 = self.pool(e3)

        e4 = self.down_conv4(p3)
        p4 = self.pool(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.up_conv1(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.up_double_conv1(d4)

        d3 = self.up_conv2(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.up_double_conv2(d3)

        d2 = self.up_conv3(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.up_double_conv3(d2)

        d1 = self.up_conv4(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.up_double_conv4(d1)

        return self.out_conv(d1)
        

class ResAttU_Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResAttU_Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.down_conv1 = ResidualBlock(in_channels, 64)
        self.down_conv2 = ResidualBlock(64, 128)
        self.down_conv3 = ResidualBlock(128, 256)
        self.down_conv4 = ResidualBlock(256, 512)

        # Bottleneck
        self.bottleneck = ResidualBlock(512, 1024)

        # Decoder        
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.Att1 = AttentionBlock(512, 512, 256)
        self.up_double_conv1 = ResidualBlock(1024, 512)
        
        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.Att2 = AttentionBlock(256, 256, 128)
        self.up_double_conv2 = ResidualBlock(512, 256)
        
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Att3 = AttentionBlock(128, 128, 64)
        self.up_double_conv3 = ResidualBlock(256, 128)
        
        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Att4 = AttentionBlock(64, 64, 32)
        self.up_double_conv4 = ResidualBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down_conv1(x)
        p1 = self.pool(x1)

        x2 = self.down_conv2(p1)
        p2 = self.pool(x2)

        x3 = self.down_conv3(p2)
        p3 = self.pool(x3)

        x4 = self.down_conv4(p3)
        p4 = self.pool(x4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.up_conv1(b)
        x4_att = self.Att1(x=x4, g=d4)
        d4 = torch.cat([d4, x4_att], dim=1)
        d4 = self.up_double_conv1(d4)

        d3 = self.up_conv2(d4)
        x3_att = self.Att2(x=x3, g=d3)
        d3 = torch.cat([d3, x3_att], dim=1)
        d3 = self.up_double_conv2(d3)

        d2 = self.up_conv3(d3)
        x2_att = self.Att3(x=x2, g=d2)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.up_double_conv3(d2)

        d1 = self.up_conv4(d2)
        x1_att = self.Att4(x=x1, g=d1)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.up_double_conv4(d1)

        return self.out_conv(d1)