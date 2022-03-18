class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0 
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, x): 
        return 1.0 - max(0, x + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
