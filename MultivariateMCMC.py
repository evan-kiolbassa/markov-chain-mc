import numpy as np
import multiprocessing as mp


class MultivariateMCMC:
    def __init__(
            self, target_pdf,
            initial_state,
            num_chains=1,
            covariance_method=None,
            proposal_covariance=None,
            burn_in_steps=1000,
            learning_rate=0.01,
            update_frequency=50
            ):
            """
            Initialize the MultivariateMCMC object.

            Parameters
            ----------
            target_pdf : callable
                The target probability density function to sample from. It should take an array-like object of
                the same length as initial_state and return a scalar.
            initial_state : array-like
                The initial state of the Markov chains. All chains are initialized to this state.
            num_chains : int, optional
                The number of Markov chains to run in parallel. The default is 1.
            covariance_method : list of str, optional
                The method used to determine the proposal covariance matrix for each chain.
                Possible values are "empirical", "adaptive", and "manual". If not provided, "empirical" is used for all chains.
            proposal_covariance : list of array-like, optional
                The initial proposal covariance matrices for each chain. Only used if covariance_method is "manual" or "adaptive".
                If not provided, the identity matrix is used for all chains.
            burn_in_steps : int, optional
                The number of steps to run for the burn-in period. The default is 1000.
            learning_rate : float, optional
                The learning rate for updating the proposal covariance in the adaptive method. Only used if covariance_method is "adaptive". 
                The default is 0.01.

            Raises
            ------
            ValueError
                If the covariance_method for any chain is not "empirical", "adaptive", or "manual".
                If the covariance_method for any chain is "manual" but proposal_covariance is not provided for that chain.
            """
            self.update_frequency = update_frequency
            self.batch = [[] for _ in range(num_chains)]
            self.target_pdf = target_pdf
            self.current_state = np.array([initial_state]*num_chains)
            self.initial_state = initial_state
            self.learning_rate = learning_rate
            self.num_chains = num_chains
            self.covariance_method = covariance_method or ["empirical"]*num_chains
            self.proposal_covariance = proposal_covariance or [np.eye(len(initial_state))]*num_chains
            self.burn_in_steps = burn_in_steps
            self.total_steps = np.zeros(num_chains)
            self.accepted_steps = np.zeros(num_chains)
            self.choose_covariance_matrix()
    def parallel_chain_run(self, burn_in=False, num_samples=0, thinning_factor=1):
        """
        Helper function to run burn-in or sampling on a single chain in parallel.
        """
        try:
            if burn_in:
                for _ in range(self.burn_in_steps):
                    self.step(0)
            else:
                samples = np.empty((num_samples, len(self.current_state[0])))
                for i in range(num_samples):
                    for _ in range(thinning_factor):
                        self.step(0)
                    samples[i] = self.current_state[0]
                return samples
        except Exception as e:
            print(f"Error in chain run: {e}")
            return None
        
    def is_valid_state(self, state, lower_bound=-10, upper_bound=10):
        """
        Check if a proposed state is valid.

        This method should be overridden in subclasses to implement problem-specific constraints.

        Parameters
        ----------
        state : array-like
            The proposed state.
        lower_bound : float, optional
            The lower bound of the valid interval. The default is -10.
        upper_bound : float, optional
            The upper bound of the valid interval. The default is 10.

        Returns
        -------
        bool
            True if the state is valid, False otherwise.
        """
        return np.all((state >= lower_bound) & (state <= upper_bound))

    def step(self, chain_index):
        """
        Take one step of the MCMC sampler for a specified chain.

        Parameters
        ----------
        chain_index : int
            The index of the chain for which to take a step.

        Returns
        -------
        array-like
            The next state of the Markov chain for the specified chain.
        """
        method = self.covariance_method[chain_index]

        # Generate a proposal for the next state of the Markov chain
        proposal = np.random.multivariate_normal(self.current_state[chain_index], self.proposal_covariance[chain_index])

        if not self.is_valid_state(proposal):
            # If the proposed state is not valid, return the current state without changing anything
            return self.current_state[chain_index]

        # Calculate the log acceptance probability for the proposal
        current_pdf = self.target_pdf(self.current_state[chain_index])
        proposal_pdf = self.target_pdf(proposal)

        # Ensure the returned values are finite numbers
        if not np.isfinite(current_pdf) or not np.isfinite(proposal_pdf):
            raise ValueError("The target PDF function returned a non-numeric value.")

        # Increment the total proposals
        self.total_steps[chain_index] += 1
        log_accept_prob = proposal_pdf - current_pdf

        if np.log(np.random.uniform()) < log_accept_prob:
            # If the proposal is accepted, update the current state
            self.current_state[chain_index] = proposal
            self.accepted_steps[chain_index] += 1.0
            self.batch[chain_index].append(proposal)

            if method == "adaptive" and len(self.batch[chain_index]) % self.update_frequency == 0:
                self.update_covariance(chain_index)
                self.batch[chain_index] = []  # clear the batch after updating

        return self.current_state[chain_index]
    def update_covariance(self, chain_index):
        """
        Update the proposal covariance matrix based on the batch of accepted proposals.
        """
        accepted_proposals = np.array(self.batch[chain_index])
        mean_proposal = np.mean(accepted_proposals, axis=0)
        diff = accepted_proposals - mean_proposal
        self.proposal_covariance[chain_index] += self.learning_rate * (np.outer(diff, diff) - self.proposal_covariance[chain_index])
    def burn_in(self):
        """
        Perform the burn-in period for all chains.
        """
        pool = mp.Pool(processes=self.num_chains)
        results = []
        for _ in range(self.num_chains):
            results.append(pool.apply_async(self.parallel_chain_run, args=(True, )))
        pool.close()
        pool.join()

        # Handle exceptions
        for result in results:
            try:
                result.get()
            except Exception as e:
                print(f"Error during burn-in: {e}")

    def sample(self, num_samples, thinning_factor=1, do_burn_in=True):
        """
        Generate a specified number of samples from the MCMC sampler, with thinning.
        """
        if thinning_factor <= 0 or not isinstance(thinning_factor, int):
            raise ValueError("thinning_factor must be a positive integer")

        if do_burn_in:
            self.burn_in()

        pool = mp.Pool(processes=self.num_chains)
        results = [pool.apply_async(self.parallel_chain_run, args=(False, num_samples, thinning_factor)) for _ in range(self.num_chains)]
        pool.close()
        pool.join()

        samples = np.empty((self.num_chains, num_samples, len(self.current_state[0])))
        for j in range(self.num_chains):
            try:
                result = results[j].get()
                if result is not None:
                    samples[j] = result
                else:
                    print(f"Skipping chain {j} due to error in computation.")
            except Exception as e:
                print(f"Error retrieving result for chain {j}: {e}")

        return samples
    
    def acceptance_rate(self):
        '''
        A method that tracks the acceptance rate
        '''
        return self.accepted_steps / self.total_steps
    
    def confidence_interval(self, samples, alpha=0.05):
        """
        Compute the credible intervals for the posterior samples for each chain.

        Parameters
        ----------
        samples : list of numpy.ndarray
            A list of numpy arrays, where each array contains the posterior samples for each chain.
        alpha : float, optional
            The desired significance level for the credible intervals (default: 0.05).

        Returns
        -------
        list of numpy.ndarray
            A list of numpy arrays, each of shape (n_params, 2), containing the lower and upper
            bounds of the credible intervals for each parameter in each chain. Each array in the list corresponds to
            a chain, and each row in an array corresponds to a parameter, with the first column being the lower bound
            and the second column being the upper bound of the credible interval for that parameter.
        """
        intervals = []
        for j in range(self.num_chains):
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)
            intervals.append(np.percentile(samples[j], [lower_percentile, upper_percentile], axis=0).T)
        return intervals

    def uncertainty(self, samples, per_chain=False):
        """
        Compute the standard deviation of the posterior samples.

        samples: A list of numpy arrays containing the posterior samples for each chain.
        per_chain: A boolean indicating whether to calculate uncertainty for each chain separately.

        Returns:
            numpy array of shape (n_params,) containing the standard deviation
            of the posterior samples for each parameter.
        """
        if per_chain:
            # Return a list of standard deviations, one for each chain
            return [np.std(chain_samples, axis=0) for chain_samples in samples]
        else:
            # Return a single standard deviation computed over all chains
            return np.std(np.concatenate(samples, axis=0), axis=0)
    
    def choose_covariance_matrix(self):
        """
        Choose the proposal covariance matrix based on the specified covariance_method.
        """
        if len(self.covariance_method) != self.num_chains:
            raise ValueError("Length of covariance_method does not match number of chains.")
        if len(self.proposal_covariance) != self.num_chains:
            raise ValueError("Length of proposal_covariance does not match number of chains.")
        for i in range(self.num_chains):
            if self.covariance_method[i] not in ["empirical", "adaptive", "manual"]:
                raise ValueError(f"Invalid covariance_method for chain {i}. Valid options are 'empirical', 'adaptive', or 'manual'.")
            if self.covariance_method[i] == "empirical":
                self.proposal_covariance[i] = self.compute_empirical_covariance(i, self.burn_in_steps)
            elif self.covariance_method[i] == "adaptive":
                if self.proposal_covariance[i] is None:
                    self.proposal_covariance[i] = np.eye(len(self.current_state[i]))
            elif self.covariance_method[i] == "manual":
                if self.proposal_covariance[i] is None:
                    raise ValueError(f"The proposal_covariance parameter must be provided when using the manual covariance method for chain {i}.")
        
    def compute_empirical_covariance(self, chain_index, num_samples):
        """
        Compute the empirical covariance matrix based on the past states.
        """
        return np.cov(np.array(past_states).T)
    
    def report(self):
        """
        Report the acceptance rate for each chain, and compute Gelman-Rubin and Effective Sample Size diagnostics.
        """
        for i in range(self.num_chains):
            if self.total_steps[i] != 0:
                print(f"Chain {i} acceptance rate: {self.acceptance_rate()[i]}")
            else:
                print(f"Chain {i} has not yet started sampling.")

        # Compute diagnostics
        samples = np.array(self.chain)  # shape = (num_chains, num_samples, num_params)
        R_hat = self.gelman_rubin(samples)
        ess = self.effective_sample_size(samples)

        for j in range(len(self.current_state[0])):
            print(f"Parameter {j} Gelman-Rubin diagnostic: {R_hat[j]}")
            print(f"Parameter {j} Effective Sample Size: {ess[j]}")

    def reset(self):
        """
        Reset the sampler to its initial state.
        """
        self.current_state = self.initial_state.copy()
        self.proposal_covariance = self.initial_proposal_covariance.copy()
        self.total_steps = np.zeros(self.num_chains)
        self.accepted_steps = np.zeros(self.num_chains)


    def reparameterize(self, transform, inverse_transform, jacobian):
        """
        Reparameterize the model using the specified transformation.

        Parameters
        ----------
        transform : callable
            The transformation to apply to the parameters. It should take an array-like object of
            the same length as the current state and return an array-like object of the same length.
        inverse_transform : callable
            The inverse of the transformation. It should take an array-like object of
            the same length as the current state and return an array-like object of the same length.
        jacobian: callable
            The Jacobian of the transformation. It should take an array-like object of
            the same length as the current state and return a matrix of shape (n, n), 
            where n is the length of the state.
        """

        transformed_state = transform(self.current_state)
        if transformed_state.shape != self.current_state.shape:
            raise ValueError("The transform function must return an array of the same shape as its input")
        self.current_state = transformed_state

        def new_target_pdf(x):
            y = inverse_transform(x)
            return self.target_pdf(y) * np.abs(np.linalg.det(jacobian(y)))

        self.target_pdf = new_target_pdf
       
        def gelman_rubin(self, samples):
            """
            Compute the Gelman-Rubin diagnostic (Potential Scale Reduction Factor) for each parameter.
            """
            # Calculate the mean of each chain for each parameter, shape = (num_chains, num_params)
            means_per_chain = np.mean(samples, axis=1)

            # Calculate the mean of each parameter across all chains, shape = (num_params,)
            means_overall = np.mean(means_per_chain, axis=0)

            # Calculate the variance of each chain for each parameter, shape = (num_chains, num_params)
            variances_per_chain = np.var(samples, axis=1)

            # Calculate the variance of each parameter across all chains, shape = (num_params,)
            variances_overall = np.var(means_per_chain, axis=0)

            # B/n: between-chain variance, shape = (num_params,)
            B_div_n = variances_overall

            # W: within-chain variance, shape = (num_params,)
            W = np.mean(variances_per_chain, axis=0)

            # Estimate of the variance of each parameter, shape = (num_params,)
            var_hat = (1 - 1.0 / self.num_chains) * W + B_div_n

            # Potential Scale Reduction Factor, shape = (num_params,)
            R_hat = np.sqrt(var_hat / W)

            return R_hat

        def effective_sample_size(self, samples):
            """
            Compute the effective sample size for each parameter.
            """
            # Calculate the autocorrelation of each chain for each parameter, shape = (num_chains, num_params)
            autocorrelations = np.empty((self.num_chains, len(self.current_state[0])))

            for i in range(self.num_chains):
                for j in range(len(self.current_state[0])):
                    autocorrelations[i, j] = np.correlate(samples[i, :, j] - np.mean(samples[i, :, j]),
                                                          samples[i, :, j] - np.mean(samples[i, :, j]))[0]

            # Calculate the autocorrelation time for each parameter, shape = (num_params,)
            autocorrelation_time = 1 + 2 * autocorrelations.sum(axis=0) / self.num_chains

            # Effective sample size for each parameter, shape = (num_params,)
            ess = self.num_chains * len(samples) / autocorrelation_time

            return ess
        
        def traceplot(self, samples):
            """
            Plot a trace plot for each parameter in each chain.

            Parameters
            ----------
            samples : numpy.ndarray
                An array of shape (num_chains, num_samples, num_params) containing the MCMC samples for each chain.
            """
            num_chains, num_samples, num_params = samples.shape

            for i in range(num_params):
                plt.figure(figsize=(10, 5))
                for j in range(num_chains):
                    plt.plot(samples[j, :, i], label=f'Chain {j}')
                plt.title(f'Traceplot for parameter {i}')
                plt.xlabel('Iteration')
                plt.ylabel('Parameter value')
                plt.legend()
                plt.show()
