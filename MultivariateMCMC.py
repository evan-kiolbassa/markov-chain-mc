from typing import Callable, Union, List, Any, Dict, Tuple
import jax.random
import jax.numpy as jnp

class MCMCKernel:
    def init(self, rng_key, *args, **kwargs):
        """
        Initialize the sampler.
        
        Parameters:
        - rng_key: A random number generator key used for initial sampling.
        - args, kwargs: Additional arguments for initialization.
        
        Returns:
        - state: The initial state for MCMC sampling.
        
        Note: Must be implemented by the subclass.
        """
        raise NotImplementedError("The init method must be implemented by the subclass.")
    
    def step(self, state, rng_key):
        """
        Perform one MCMC step starting from the given state.
        
        Parameters:
        - state: The current state of the MCMC chain.
        - rng_key: A random number generator key used for randomness in the step.
        
        Returns:
        - new_state: The state after one MCMC step.
        - is_accepted: A boolean indicating whether the step was accepted.
        
        Note: Must be implemented by the subclass.
        """
        raise NotImplementedError("The step method must be implemented by the subclass.")

class MCMCSampler:
    def __init__(self, sampler: MCMCKernel, num_warmup: int, num_samples: int,
                 thinning: int = 1, num_chains: int = 1, 
                 postprocess_fn: Callable[[Any], Any] = None,
                 chain_method: str = 'parallel', progress_bar: bool = True,
                 jit_model_args: bool = False, target_acceptance_rate: float = 0.8,
                 gamma: float = 0.05, t0: float = 10, kappa: float = 0.75):
        """
        Initialize the MCMC Sampler.
        
        Parameters:
        - sampler: The MCMC kernel used for sampling.
        - num_warmup: Number of warm-up steps.
        - num_samples: Number of samples to collect.
        - thinning: Thinning factor for the chain.
        - num_chains: Number of MCMC chains.
        - postprocess_fn: Function to postprocess samples.
        - chain_method: Method for running chains ('parallel' or 'sequential').
        - progress_bar: Whether to show a progress bar.
        - jit_model_args: Whether to JIT compile model arguments.
        - target_acceptance_rate: Target acceptance rate for dual-averaging.
        - gamma, t0, kappa: Hyperparameters for the dual-averaging algorithm.
        """
        self.sampler = sampler
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.thinning = thinning
        self.num_chains = num_chains
        self.postprocess_fn = postprocess_fn
        self.chain_method = chain_method
        self.progress_bar = progress_bar
        self.jit_model_args = jit_model_args
        self.collected_samples = []
        self.extra_fields = {}
        self.warmup_samples = []
        self.target_acceptance_rate = target_acceptance_rate
        self.log_step_size = jnp.log(1.0)  # log(initial step size)
        self.log_step_size_avg = 0.0  # log average of step size
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.accepted_count = 0

    def warmup(self, rng_key, *args, extra_fields=(), collect_warmup=False, init_params=None, **kwargs):
        """
        Perform the warm-up phase of MCMC sampling.
        
        Parameters:
        - rng_key: A random number generator key for the warm-up phase.
        - args, kwargs: Additional arguments passed to the sampler.
        - extra_fields: Additional state fields to collect.
        - collect_warmup: Whether to collect warm-up samples.
        - init_params: Initial parameters for the MCMC chain.
        
        Side-effects:
        - Updates `self.warmup_state` to the final state after warm-up.
        - If `collect_warmup` is True, populates `self.warmup_samples`.
        - Updates `self.step_size` based on dual-averaging.
        """
        state = self.sampler.init(rng_key, *args, **kwargs)
        self.warmup_state = state
        if collect_warmup:
            self.warmup_samples = [state]
        
        for i in range(1, self.num_warmup + 1):
            rng_key, subkey = jax.random.split(rng_key)
            new_state, is_accepted = self.sampler.step(state, subkey, step_size=jnp.exp(self.log_step_size))
            
            self.accepted_count += is_accepted
            
            acceptance_rate = is_accepted - self.target_acceptance_rate  # difference from target
            self.log_step_size = self.log_step_size + (i + 1) ** -0.5 * acceptance_rate
            
            eta = (i + self.t0) ** -self.kappa
            self.log_step_size_avg = eta * self.log_step_size + (1 - eta) * self.log_step_size_avg
                
            state = new_state
            
            if collect_warmup:
                self.warmup_samples.append(new_state)
        
        self.step_size = jnp.exp(self.log_step_size_avg)
        self.warmup_state = state
