class AE(nn.Module):
    def __init__(self, d_in: int, d_l: int):
        super(AE, self).__init__()

        self.d_in = d_in
        self.d_l = d_l

        # encoder
        self.en_fc1 = nn.Linear(self.d_in, d_l)

        # decoder
        self.dec_fc1 = nn.Linear(d_l, d_in)

    def encode(self, x):
        return F.relu(self.en_fc1(x))

    def decode(self, z):
        return torch.sigmoid(self.dec_fc1(z))

    def forward(self, x):
        z = self.encode(x.view(-1, self.d_in))

        return self.decode(z)

    def save_parameters(self, filename: str):
        torch.save(self.state_dict(), filename)

    def load_parameters(self, filename: str, device: torch.device):
        self.load_state_dict(torch.load(filename, map_location=device))