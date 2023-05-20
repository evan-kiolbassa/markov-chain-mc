class MultivariateMCMC:
    def __init__(
            self, target_pdf,
            initial_state,
            num_chains=1,
            covariance_method=None,
            proposal_covariance=None,
            burn_in_steps=1000
            ):
        self.target_pdf = target_pdf
        self.current_state = np.array([initial_state]*num_chains)
        self.learning_rate = 0.01
        self.num_chains = num_chains
        self.covariance_method = covariance_method or ["empirical"]*num_chains
        self.proposal_covariance = proposal_covariance or [np.eye(len(initial_state))]*num_chains
        self.burn_in_steps = burn_in_steps
        self.acceptance_rate = np.zeros(num_chains)

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

        if method == "adaptive":
            # Generate a proposal for the next state of the Markov chain
            proposal = np.random.multivariate_normal(self.current_state[chain_index], self.proposal_covariance[chain_index])

            # Calculate the log acceptance probability for the proposal
            log_accept_prob = self.target_pdf(proposal) - self.target_pdf(self.current_state[chain_index])

            # Accept or reject the proposal based on the acceptance probability
            if np.log(np.random.uniform()) < log_accept_prob:
                self.current_state[chain_index] = proposal
                self.acceptance_rate[chain_index] += 1.0

                # If the proposal is accepted, update the proposal covariance matrix using the accepted proposal
                diff = proposal - self.current_state[chain_index]
                self.proposal_covariance[chain_index] += self.learning_rate * (np.outer(diff, diff) - self.proposal_covariance[chain_index])
                
        elif method == "manual":
            # This is a naive implementation. Modify this as per your requirements.
            proposal = np.random.multivariate_normal(self.current_state[chain_index], self.proposal_covariance[chain_index])

            log_accept_prob = self.target_pdf(proposal) - self.target_pdf(self.current_state[chain_index])
            if np.log(np.random.uniform()) < log_accept_prob:
                self.current_state[chain_index] = proposal
                self.acceptance_rate[chain_index] += 1.0

        elif method == "empirical":
            self.proposal_covariance[chain_index] = self.compute_empirical_covariance(chain_index, 1000)
            proposal = np.random.multivariate_normal(self.current_state[chain_index], self.proposal_covariance[chain_index])
            log_accept_prob = self.target_pdf(proposal) - self.target_pdf(self.current_state[chain_index])
            if np.log(np.random.uniform()) < log_accept_prob:
                self.current_state[chain_index] = proposal
                self.acceptance_rate[chain_index] += 1.0

        else:
            raise ValueError(f"Invalid covariance method: {method}")

        return self.current_state[chain_index]

    def burn_in(self):
        """
        Perform the burn-in period by running the MCMC sampler for a specified number
        of steps without collecting any samples. This is done in parallel for all chains.
        """
        for _ in range(self.burn_in_steps):
            for j in range(self.num_chains):
                self.step(j)
        self.acceptance_rate = np.zeros(self.num_chains)

    def sample(self, num_samples, thinning_factor=1, do_burn_in=True):
        """
        Generate a specified number of samples from the MCMC sampler, with thinning.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.
        thinning_factor : int
            The thinning factor to apply. Only every thinning_factor-th sample is kept.
        do_burn_in : bool
            Whether to perform burn-in or not.

        Returns
        -------
        numpy.ndarray
            A numpy array containing the generated samples.
        """
        if do_burn_in:
            self.burn_in()
            self.choose_covariance_matrix()

        # Ensure we generate enough samples, accounting for thinning
        total_samples = num_samples * thinning_factor

        samples = np.empty((self.num_chains, total_samples, len(self.current_state[0])))
        for j in range(self.num_chains):
            for i in range(total_samples):
                samples[j, i] = self.step(j)

        # Apply thinning by selecting only every thinning_factor-th sample
        thinned_samples = samples[:, ::thinning_factor]

        for j in range(self.num_chains):
            self.acceptance_rate[j] /= (total_samples)

        return thinned_samples
    
    def confidence_interval(self, samples, alpha=0.05):
        """
        Compute the credible intervals for the posterior samples for each chain.

        samples: A list of numpy arrays containing the posterior samples for each chain.
        alpha: The desired significance level for the credible intervals (default: 0.05).

        Returns:
            A list of numpy arrays, each of shape (n_params, 2), containing the lower and upper
            bounds of the credible intervals for each parameter in each chain.
        """
        intervals = []
        for j in range(self.num_chains):
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)
            intervals.append(np.percentile(samples[j], [lower_percentile, upper_percentile], axis=0).T)
        return intervals

    def uncertainty(self, samples):
        """
        Compute the standard deviation of the posterior samples.

        samples: A list of numpy arrays containing the posterior samples for each chain.

        Returns:
            numpy array of shape (n_params,), containing the standard deviation
            of the posterior samples for each parameter.
        """
        return np.std(np.concatenate(samples, axis=0), axis=0)
    
    def choose_covariance_matrix(self):
        """
        Choose the proposal covariance matrix based on the specified covariance_method.
        """
        for i in range(self.num_chains):
            if self.covariance_method[i] == "empirical":
                self.proposal_covariance[i] = self.compute_empirical_covariance(i, 1000)
            elif self.covariance_method[i] == "adaptive":
                if self.proposal_covariance[i] is None:
                    self.proposal_covariance[i] = np.eye(len(self.current_state[i]))
            elif self.covariance_method[i] == "manual":
                if self.proposal_covariance[i] is None:
                    raise ValueError(f"The proposal_covariance parameter must be provided when using the manual covariance method for chain {i}.")
            else:
                raise ValueError(f"Invalid covariance_method for chain {i}. Valid options are 'empirical', 'adaptive', or 'manual'.")
        
    def compute_empirical_covariance(self, chain_index, num_samples):
        """
        Compute the empirical covariance matrix based on the specified number of samples.
        """
        samples = [self.step(chain_index) for _ in range(num_samples)]
        return np.cov(np.array(samples).T)
    
    def report(self):
        """
        Report the acceptance rate for each chain.
        """
        for i in range(self.num_chains):
            print(f"Chain {i+1} acceptance rate: {self.acceptance_rate[i]}")

    def reset(self):
        """
        Reset the sampler's state and acceptance rate.
        """
        self.current_state = np.array([self.initial_state]*self.num_chains)
        self.acceptance_rate = np.zeros(self.num_chains)
