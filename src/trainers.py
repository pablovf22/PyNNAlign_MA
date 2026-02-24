import torch

class NNAlign_MA_trainer:

    """
    Trainer class for NNAlign-style multi-allele models.

    Handles the training loop over multiple epochs, using an external
    DataLoader factory to allow epoch-dependent dataset behavior (e.g.
    single-allele burn-in).
    """


    def __init__(self, model, criterion, optimizer, make_loader, device):

        self.model = model.to(device)  # move model to device
        self.criterion = criterion
        self.optimizer = optimizer
        self.make_loader = make_loader
        self.device = device


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

            loader = self.make_loader(epoch)
            self._train_one_epoch(loader)

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}")


    def save(self, syn_path):

        torch.save(self.model.state_dict(), syn_path)  # save model weights
