class MultivariateMCMC:
    def __init__(
            self, target_pdf,
            initial_state,
            num_chains=1,
            covariance_method=None,
            proposal_covariance=None,
            burn_in_steps=1000,
            learning_rate=0.01
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

        Raises
        ------
        ValueError
            If the covariance_method for any chain is not "empirical", "adaptive", or "manual".
            If the covariance_method for any chain is "manual" but proposal_covariance is not provided for that chain.
        """
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
            self.current_state[chain_index] = proposal
            self.accepted_steps[chain_index] += 1.0

            if method == "adaptive":
                # If the proposal is accepted, update the proposal covariance matrix using the accepted proposal
                diff = proposal - self.current_state[chain_index]
                self.proposal_covariance[chain_index] += self.learning_rate * (np.outer(diff, diff) - self.proposal_covariance[chain_index])

        return self.current_state[chain_index]



    def burn_in(self):
        """
        Perform the burn-in period for all chains.
        """
        for _ in range(self.burn_in_steps):
            for j in range(self.num_chains):
                self.step(j)

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
        if thinning_factor <= 0 or not isinstance(thinning_factor, int):
            raise ValueError("thinning_factor must be a positive integer")

        if do_burn_in:
            self.burn_in()

        total_samples = num_samples * thinning_factor

        samples = np.empty((self.num_chains, total_samples, len(self.current_state[0])))
        for j in range(self.num_chains):
            for i in range(total_samples):
                samples[j, i] = self.step(j)

        thinned_samples = samples[:, ::thinning_factor]

        return thinned_samples
    
    def acceptance_rate(self):
        '''
        A method that tracks the acceptance rate
        '''
        return self.accepted_steps / self.total_steps
    
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
        Compute the empirical covariance matrix based on the specified number of samples.
        """
        samples = [self.step(chain_index) for _ in range(num_samples)]
        return np.cov(np.array(samples).T)
    
    ef report(self):
        """
        Report the acceptance rate for each chain.
        """
        for i in range(self.num_chains):
            if self.total_steps[i] != 0:
                print(f"Chain {i} acceptance rate: {self.acceptance_rate()[i]}")
            else:
                print(f"Chain {i} has not yet started sampling.")

    def reset(self):
        """
        Reset the sampler to its initial state.
        """
        self.current_state = self.initial_state.copy()
        self.proposal_covariance = self.initial_proposal_covariance.copy()
        self.total_steps = np.zeros(self.num_chains)
        self.accepted_steps = np.zeros(self.num_chains)


    def reparameterize(self, transform, inverse_transform):
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
        """

        transformed_state = transform(self.current_state)
        if transformed_state.shape != self.current_state.shape:
            raise ValueError("The transform function must return an array of the same shape as its input")
        self.current_state = transformed_state
        old_target_pdf = self.target_pdf
        self.target_pdf = lambda x: old_target_pdf(inverse_transform(x))
