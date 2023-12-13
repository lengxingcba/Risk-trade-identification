import fsspec

from model.layer import *


def model_v1(num_classes):
    return nn.Sequential(Block(inp=1, oup=8, k=3, shortcut=False, stride=2),
                         Block(8, 8, 3, True, stride=1),
                         Block(8, 16, 3, False, stride=1),
                         Block(16, 16, 3, True, stride=1),
                         Block(16, 8, 3, False, stride=1),
                         Block(8, 8, 3, True, stride=1),
                         Decoder(num_classes))


class MLP(nn.Module):
    def __init__(self, inp=1, num_classes=2):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(inp, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        b,c,f=x.shape
        x=x.view(b,c*f)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)

        return self.softmax(x)


def model_v2(num_classes,d_model):

    return nn.Sequential(
                         encoder(d_model=d_model,n_head=4,num_layers=2),
                         Decoder_softmax(num_classes)
    )

# x=torch.randn([16,1,32])
#
# model=model_v1()
# y=model.forward(x)
#
# print(y.shape)
