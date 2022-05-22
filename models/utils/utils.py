class Earlystopping:
    """
    early-stopping class which can be used in training stage.
    Parameters
    ----------
    patience: int, earlystopping rounds.
    delta: float, the difference with best score. (Default: 0.0)
    """
    def __init__(self, patience, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print("EarlyStopping counter:{} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0