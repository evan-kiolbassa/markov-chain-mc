import numpy as np
from scipy.stats import multivariate_normal
import numpy as np
from scipy.stats import multivariate_normal, expon, geom

class MultivariateMCMC:
    def __init__(
            self, target_pdf, 
            initial_state, 
            covariance_method="empirical", 
            proposal_covariance=None,
            burn_in_steps=1000
            ):
        """
        Initialize the MCMC sampler.

        Parameters
        ----------
        target_pdf : function
            A function that takes a vector of model parameters as input
            and returns the log of the posterior probability density.
        initial_state : array-like
            A vector representing the initial state of the Markov chain.
        covariance_method : str, optional, default: "empirical"
            The method to use for choosing the proposal covariance matrix.
            Valid options are "empirical", "adaptive", or "manual".
        proposal_covariance : array-like, optional
            A positive definite covariance matrix used to generate
            proposals for the next state of the Markov chain. This
            parameter is only used if covariance_method is "manual".
        burn_in_steps : int, optional, default: 1000
            The number of burn-in steps to perform before generating samples.
        """
        self.target_pdf = target_pdf
        self.current_state = np.array(initial_state)
        self.covariance_method = covariance_method
        self.proposal_covariance = proposal_covariance
        self.burn_in_steps = burn_in_steps
        self.accepted_proposals = 0

    def step(self):
        """
        Take one step of the MCMC sampler.

        Returns
        -------
        array-like
            The next state of the Markov chain.
        """
        proposal = np.random.multivariate_normal(self.current_state, self.proposal_covariance)
        log_accept_prob = self.target_pdf(proposal) - self.target_pdf(self.current_state)
        if np.log(np.random.uniform()) < log_accept_prob:
            self.current_state = proposal
            self.accepted_proposals += 1
        return self.current_state

    def burn_in(self):
        """
        Perform the burn-in period by running the MCMC sampler for a specified number
        of steps without collecting any samples.
        """
        for _ in range(self.burn_in_steps):
            self.step()

    def sample(self, num_samples):
        """
        Generate a specified number of samples from the MCMC sampler.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.

        Returns
        -------
        numpy.ndarray
            A numpy array containing the generated samples.
        """
        self.burn_in()
        samples = np.empty((num_samples, len(self.current_state)))
        for i in range(num_samples):
            samples[i] = self.step()
        total_proposals = self.burn_in_steps + num_samples
        self.acceptance_rate = self.accepted_proposals / total_proposals
        return samples

    def choose_covariance_matrix(self):
        """
        Choose the proposal covariance matrix based on the specified covariance_method.
        """
        pass
        
    def compute_empirical_covariance(self, num_samples):
       pass
        
    def gradient_ascent(self, target_pdf_grad, learning_rate=0.01, max_iter=1000, tol=1e-6):
        """
        Perform gradient ascent to find the mode of the target distribution.

        target_pdf_grad: A function that computes the gradient of the target
                         probability density function with respect to the
                         parameters.
        learning_rate: The learning rate for the gradient ascent algorithm.
        max_iter: The maximum number of iterations for the gradient ascent algorithm.
        tol: The convergence tolerance for the gradient ascent algorithm.

        Returns:
            The mode of the target distribution.
        """
        current_state = self.current_state.copy()
        for _ in range(max_iter):
            grad = target_pdf_grad(current_state)
            next_state = current_state + learning_rate * grad

            if np.linalg.norm(next_state - current_state) < tol:
                break

            current_state = next_state

        return current_state
    
    def confidence_interval(self, samples, alpha=0.05):
        """
        Compute the credible intervals for the posterior samples.

        samples: A numpy array containing the posterior samples.
        alpha: The desired significance level for the credible intervals (default: 0.05).

        Returns:
            A numpy array of shape (n_params, 2) containing the lower and upper
            bounds of the credible intervals for each parameter.
        """
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        return np.percentile(samples, [lower_percentile, upper_percentile], axis=0).T

    def uncertainty(self, samples):
        """
        Compute the standard deviation of the posterior samples.

        samples: A numpy array containing the posterior samples.

        Returns:
            A numpy array of shape (n_params,) containing the standard deviation
            of the posterior samples for each parameter.
        """
        return np.std(samples, axis=0)
