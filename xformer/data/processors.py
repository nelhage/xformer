MASK_TOKEN = 0xff

class ARProcessor:
  def __init__(self, iter, device='cpu'):
    self.device = device
    self.iterator = iter

  def __iter__(self):
    for record in iter(self.iterator):
      record['text'] = record['text'].to(self.device)
      record['input'] = record['text'][:, :-1]
      yield record

class MaskedProcessor:
  def __init__(self, iter, device='cpu'):
    self.device = device
    self.iterator = iter
    self.mask_token = torch.tensor(MASK_TOKEN, device=device, dtype=torch.long)

  def __iter__(self):
    for record in iter(self.iterator):
      record['text'] = record['text'].to(self.device)
      record['mask'] = record['mask'].to(self.device)
      record['input'] = torch.where(record['mask'], record['text'], self.mask_token)
      yield record
