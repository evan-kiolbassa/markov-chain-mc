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
        self.current_state = initial_state
        self.covariance_method = covariance_method
        self.proposal_covariance = proposal_covariance
        self.burn_in_steps = burn_in_steps
        self.acceptance_rate = 0.0

    def step(self):
        """
        Take one step of the MCMC sampler.

        Returns
        -------
        array-like
            The next state of the Markov chain.
        """
        # Generate a proposal for the next state of the Markov chain
        proposal = np.random.multivariate_normal(self.current_state, self.proposal_covariance)

        # Calculate the log acceptance probability for the proposal
        log_accept_prob = self.target_pdf(proposal) - self.target_pdf(self.current_state)

        # Accept or reject the proposal based on the acceptance probability
        if np.log(np.random.uniform()) < log_accept_prob:
            self.current_state = proposal
            self.acceptance_rate += 1.0

        return self.current_state

    def burn_in(self):
        """
        Perform the burn-in period by running the MCMC sampler for a specified number
        of steps without collecting any samples.
        """
        for i in range(self.burn_in_steps):
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
        self.choose_covariance_matrix()
        self.burn_in()
        samples = np.empty((num_samples, len(self.current_state)))
        for i in range(num_samples):
            samples[i] = self.step()
        self.acceptance_rate /= (self.burn_in_steps + num_samples)
        return samples

    def choose_covariance_matrix(self):
        """
        Choose the proposal covariance matrix based on the specified covariance_method.
        """
        if self.covariance_method == "empirical":
            self.proposal_covariance = self.compute_empirical_covariance(1000)
        elif self.covariance_method == "adaptive":
            if self.proposal_covariance is None:
                self.proposal_covariance = np.eye(len(self.current_state))
            for i in range(1000):
                proposal = np.random.multivariate_normal(self.current_state, self.proposal_covariance)
                log_accept_prob = self.target_pdf(proposal) - self.target_pdf(self.current_state)
                acceptance_prob = np.exp(log_accept_prob)
                if np.random.uniform() < acceptance_prob:
                    self.current_state = proposal
                    self.proposal_covariance += np.outer(
                        proposal - self.current_state, proposal - self.current_state
                        )
                else:
                    self.proposal_covariance -= np.outer(
                        proposal - self.current_state, proposal - self.current_state
                        )
        elif self.covariance_method == "manual":
            if self.proposal_covariance is None:
                raise ValueError(
                    "The proposal_covariance parameter must be provided when using the manual covariance method."
                    )
        else:
            raise ValueError("Invalid covariance_method. Valid options are 'empirical', 'adaptive', or 'manual'.")
        
    def generate_samples(self, num_samples, use_current_covariance=True):
        """
        Generate a specified number of samples without choosing the covariance matrix.
        
        Parameters
        ----------
        num_samples : int
            The number of samples to generate.
        use_current_covariance : bool, optional, default: True
            Whether to use the current proposal covariance or an identity matrix.
            
        Returns
        -------
        numpy.ndarray
            A numpy array containing the generated samples.
        """
        if not use_current_covariance:
            original_covariance = self.proposal_covariance
            self.proposal_covariance = np.eye(len(self.current_state))

        samples = np.empty((num_samples, len(self.current_state)))
        for i in range(num_samples):
            samples[i] = self.step()

        if not use_current_covariance:
            self.proposal_covariance = original_covariance

        return samples
        
    def compute_empirical_covariance(self, num_samples):
        """
        Compute the empirical covariance matrix based on the specified number of samples.

        Parameters
        ----------
        num_samples : int
            The number of samples used to compute the empirical covariance matrix.

        Returns
        -------
        numpy.ndarray
            A numpy array representing the computed empirical covariance matrix.
        """
        temp_samples = self.generate_samples(num_samples, use_current_covariance=False)
        empirical_covariance = np.cov(temp_samples.T) + np.eye(len(self.current_state)) * 1e-6
        return empirical_covariance
        
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
