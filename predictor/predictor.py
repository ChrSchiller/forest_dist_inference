import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model.bert import SBERT
from model.classification_model import SBERTClassification

class SBERTPredictor:
    def __init__(self, sbert: SBERT, num_classes: int,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 seq_len: int, 
                 lr: float = 1e-5, with_cuda: bool = True,
                 cuda_devices=None, log_freq: int = 100):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.sbert = sbert
        self.model = SBERTClassification(sbert, num_classes, seq_len).to(self.device)

        self.num_classes = num_classes

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for model fine-tuning" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr)

        self.log_freq = log_freq


    def load(self, model_name):
        checkpoint = torch.load(model_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.train()
        epoch = checkpoint['epoch']

        print("EP:%d Model loaded from:" % epoch, model_name)
        return model_name

    def predict(self, data_loader):
        # Disable grad
        with torch.no_grad():
            self.model.eval()

            # pred = []
            pred_raw_output = []

            for data in data_loader:
                data = {key: value.to(self.device) for key, value in data.items()}

                ### sigmoid added because:
                ### https://discuss.pytorch.org/t/playing-with-bcewithlogitsloss/82673/2
                classification = torch.sigmoid(
                    self.model(data["bert_input"].float(),
                               data["time"].long(),
                               data["bert_mask"].long()
                               ))

                classification_result = classification.squeeze()

                pred_raw_output.append(classification_result.cpu().numpy())

        return pred_raw_output