from torch.utils import data
from PIL import Image
import  torch

class MyDataset(data.Dataset):
    def __init__(self, token_text ):
        self.token_text = token_text

    def __getitem__(self, index):
        tmp = torch.Tensor(self.token_text[index]).long()
        if torch.cuda.is_available():
            tmp=tmp.cuda()

        return [tmp, tmp]

    def __len__(self):
        return len(self.token_text)
