import torch
import torch.nn as nn
import sys
import os

basic_channel = 32


class MLP(nn.Module):
    def __init__(self, input_channel=1, output_classes=5):
        super(MLP, self).__init__()
        self.input_channel = input_channel
        self.output_classes = output_classes

        self.FC1 = nn.Sequential(
            nn.Linear(self.input_channel, basic_channel*4),
            nn.BatchNorm1d(basic_channel*4),
            nn.Sigmoid()
        )

        self.FC2 = nn.Sequential(
            nn.Linear(basic_channel*4, basic_channel*4),
            nn.BatchNorm1d(basic_channel*4),
            nn.Sigmoid()
        )

        self.FC3 = nn.Sequential(
            nn.Linear(basic_channel*4, basic_channel*4),
            nn.BatchNorm1d(basic_channel*4),
            nn.Sigmoid()
        )

        self.FC4 = nn.Sequential(
            nn.Linear(basic_channel*4, output_classes),
            # nn.ReLU()
            # nn.Sigmoid()
        )
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        x = self.FC4(x)

        return x
if __name__ == '__main__':
    model = MLP(input_channel=5, output_classes=1)
    
    # path, = sys.argv[1:]
    for model_file in os.listdir('./'):
        if model_file.find('.pth') == -1:
            continue
        print(model_file)
        model.load_state_dict(torch.load(os.path.join('./', model_file), map_location='cpu'))
        model.eval()
        dummy = torch.zeros([1, 5])
        traced_cell = torch.jit.trace(model, dummy)
        traced_cell.save(os.path.join('models_trace', model_file[:-4] + '.ptc'))
