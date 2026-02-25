import torch

class NNAlign_MA_trainer:

    """
    Trainer class for NNAlign-style multi-allele models.

    Handles the training loop over multiple epochs, using an external
    DataLoader factory to allow epoch-dependent dataset behavior (e.g.
    single-allele burn-in).
    """


    def __init__(self, model, criterion, optimizer, device, loader_ma, loader_sa, loader_val, logger, SA_burn_in):

        self.model = model.to(device)  # move model to device
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.loader_ma = loader_ma
        self.loader_sa = loader_sa
        self.loader_val = loader_val
        self.SA_burn_in = SA_burn_in
        self.MSE_train = []
        self.PCC_train = []
        self.MSE_val = []
        self.PCC_val = []
        self.logger = logger


    def _train_one_epoch(self, loader):

        self.model.train()  # set training mode

        z_max_epoch = []
        y_epoch = []

        for batch in loader:
            
            self.optimizer.zero_grad(set_to_none=True)

            # move batch tensors to device
            X, y, pep_idx = [tensor.to(self.device, non_blocking=True) for tensor in batch]

            z_max = self.model(X, pep_idx)
            batch_loss = self.criterion(z_max, y)

            z_max_epoch.append(z_max.detach())
            y_epoch.append(y.detach())

            batch_loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            z_max_epoch = torch.cat(z_max_epoch, dim=0)
            y_epoch = torch.cat(y_epoch, dim=0)

            PCC_epoch = self._pcc_torch(z_max_epoch=z_max_epoch, y_epoch=y_epoch)
            MSE_epoch = ((z_max_epoch - y_epoch) ** 2).mean().item()

        print(f"MSE train: {MSE_epoch}  --  PCC train: {PCC_epoch}")

        self.MSE_train.append(MSE_epoch)
        self.PCC_train.append(PCC_epoch)


    def _validate_one_epoch(self):

        self.model.eval()

        z_max_epoch = []
        y_epoch = []

        with torch.no_grad():

            for batch in self.loader_val:

                X, y, pep_idx = [tensor.to(self.device, non_blocking=True) for tensor in batch]
                y = y.float()

                z_max = self.model(X, pep_idx)

                z_max_epoch.append(z_max)
                y_epoch.append(y)

            z_max_epoch = torch.cat(z_max_epoch, dim=0)
            y_epoch = torch.cat(y_epoch, dim=0)

            PCC_epoch = self._pcc_torch(z_max_epoch, y_epoch)
            MSE_epoch = ((z_max_epoch - y_epoch) ** 2).mean().item()

        print(f"MSE val: {MSE_epoch}  --  PCC val: {PCC_epoch}")

        self.MSE_val.append(MSE_epoch)
        self.PCC_val.append(PCC_epoch)
        
        
    def train(self, num_epochs=300):

        for epoch in range(num_epochs):

            if epoch == 0:
                print(f"--- Phase: SA Burn-in (Epochs 1-{self.SA_burn_in}) ---")
            elif epoch == self.SA_burn_in:
                print(f"--- Phase: Full Multi-Allele (Epochs {self.SA_burn_in+1}+) ---")

            print(f"Epoch {epoch+1}")

            loader = self.loader_sa if epoch < self.SA_burn_in else self.loader_ma
            self._train_one_epoch(loader)
            self._validate_one_epoch()

            self.logger.log({
                "MSE_train": self.MSE_train[-1],
                "PCC_train": self.PCC_train[-1],
                "MSE_val": self.MSE_val[-1],
                "PCC_val": self.PCC_val[-1]
                }, step=epoch+1)


    def save(self, syn_path):

        torch.save(self.model.state_dict(), syn_path)  # save model weights


    @staticmethod
    def _pcc_torch(z_max_epoch, y_epoch, eps=1e-8):

        x = z_max_epoch - z_max_epoch.mean()
        y = y_epoch - y_epoch.mean()

        numerator = (x * y).sum()
        denominator = torch.sqrt((x**2).sum()) * torch.sqrt((y**2).sum()) + eps

        return (numerator / denominator).item()