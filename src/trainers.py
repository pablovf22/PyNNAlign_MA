import torch

class NNAlign_MA_trainer:

    """
    Trainer class for NNAlign-style multi-allele models.

    Handles the training loop over multiple epochs, using an external
    DataLoader factory to allow epoch-dependent dataset behavior (e.g.
    single-allele burn-in).
    """


    def __init__(self, model, criterion, optimizer, device, loader_ma, loader_sa, SA_burn_in):

        self.model = model.to(device)  # move model to device
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.loader_ma = loader_ma
        self.loader_sa = loader_sa
        self.SA_burn_in = SA_burn_in


    def _train_one_epoch(self, loader):

        self.model.train()  # set training mode

        for batch in loader:
            
            self.optimizer.zero_grad(set_to_none=True)

            # move batch tensors to device
            X, y, pep_idx = [tensor.to(self.device, non_blocking=True) for tensor in batch]

            z_max = self.model(X, pep_idx)
            batch_loss = self.criterion(z_max, y)

            batch_loss.backward()
            self.optimizer.step()


    def train(self, num_epochs=300):

        for epoch in range(num_epochs):

            if epoch == 0:
                print(f"--- Phase: SA Burn-in (Epochs 1-{self.SA_burn_in}) ---")
            elif epoch == self.SA_burn_in:
                print(f"--- Phase: Full Multi-Allele (Epochs {self.SA_burn_in+1}+) ---")

            loader = self.loader_sa if epoch < self.SA_burn_in else self.loader_ma
            self._train_one_epoch(loader)

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}")


    def save(self, syn_path):

        torch.save(self.model.state_dict(), syn_path)  # save model weights
