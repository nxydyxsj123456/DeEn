from torch.utils import data
from PIL import Image


class MyDataset(data.Dataset):
    def __init__(self, token_text, ):
        self.token_text = token_text
#123


    def __getitem__(self, index):


        return self.token_text[index], self.token_text[index]

    def __len__(self):
        return len(self.images)
