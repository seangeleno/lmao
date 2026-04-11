from optuna.samplers import CmaEsSampler

class RealOptunaManager:
    # ... other methods

    def optimize_coin(self):
        # ... previous code
        sampler = CmaEsSampler(n_startup_trials=15, sigma0=0.12)  # Updated line
        # ... following code
