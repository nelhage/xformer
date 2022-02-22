from torch import nn

class ARLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.xent = nn.CrossEntropyLoss(reduction='mean')

  def forward(self, logits, batch):
    targets = batch['text'][:, 1:]
    loss = self.xent(logits.permute(0, 2, 1), targets)
    return loss

class MaskedLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.xent = nn.CrossEntropyLoss(reduction='none')

  def forward(self, logits, batch):
    targets = batch['text']
    loss = self.xent(logits.permute(0, 2, 1), targets)
    return (loss * (1-batch['mask'])).mean()
