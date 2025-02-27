class Validation:
    def __init__(self, model, data_loader, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device
        pass