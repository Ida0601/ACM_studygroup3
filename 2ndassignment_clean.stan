
// data-block: define the data types we will be giving the model
data {
  int<lower=1> n; // trial n is type integer, with a lower boundary of 1
  
  array[n] int<lower=0, upper=1> choice; //array "choice" of length n taking the value of either 0=left or 1=right 
  
  array[n] int<lower=0, upper=1> other; //array "other" of length n taking the value of either 0=opponent played left or 1= opponent played right
  
  // define alpha priors (mean and sd) that we can specify when calling the stan model
  real prior_alpha_m; // mean prior_alpha is a continuous number (real)

  real<lower = 0> prior_alpha_sd; // sd prior_alpha is a continuous number that cannot be lower than 0
}

transformed data {
  array[n] int reward; //array "reward" of length n 

  // for-loop calculating reward for each trial
  for (trial in 1:n) { // starting 
    
  //if the RL agent (the matcher) played the same as the opponent, then reward is 1.
    if (choice[trial] == other[trial])
      reward[trial] = 1;
      
  // if the RL agent (the matcher) didn't play the same as the opponent, then reward is 0.
    else
      reward[trial]= 0;   
  }
}

// Define the parameters of the model
parameters {
  real alpha_logit; // Learning rate alpha_logit is a continuous number
}

// Define the transformed parameters and model formular
transformed parameters {
  real alpha = inv_logit(alpha_logit); //Learning rate alpha is the inverse log-odds of our "free" alpha
  
  array[n] real expected_value; // array "expected_value" of length n
  
  expected_value[1] = 0.5; // define expected_value on trial 1
  
  real feedback; // feedback is continous number
  
  // for-loop calculating the expected value (EV) for each trial from trial 2 and onwards.
  for (trial in 2:n) { // starting 
    
  //if the RL agent (the matcher) played left on the previous trial, then 
  // feedback is 0 if the reward was 1 (reinforcing left choice)
    if (choice[trial-1] == 0)
      feedback = 1 - reward[trial-1];
      
  // if the RL agent (the matcher) played right on the previous trial, then 
  // feedback is 1 if the reward was 1 (reinforcing right)
    else
      feedback = reward[trial-1];
  
  // Formula, where EV for each trial is calculated
    expected_value[trial] = expected_value[trial-1] + alpha * (feedback - expected_value[trial-1]);
  }
}

// Define the distributions from which our model should assume the data comes from 
model {
  //Prior: Our belief about alpha before seeing the data
  target += normal_lpdf(alpha_logit|prior_alpha_m, prior_alpha_sd); //The log of the normal density of alpha given prior alpha mu and sd

  // Likelihood: How the data 'choice' depend on the parameter 'expected value'
  target += bernoulli_lpmf(choice|expected_value); //The log Bernoulli probability mass of choice given probability of EV

}

// Calculate values for estimating model quality and parameter recovery
generated quantities {
  
  // sample prior vals for alpha for prior posterior update checks
  real alpha_logit_prior = normal_rng(prior_alpha_m, prior_alpha_sd); //Generate a normal variate with location prior_alpha_m and prior_alpha_sd 
  real<lower=0, upper=1> alpha_prior = inv_logit(alpha_logit_prior); //transform alpha_logit_prior to be between 0 and 1 using the inverse logit function
  
  //  ------- Prior Predictive Check Simulation ------- 
  // Simulate data based on the prior distribution
  // Similar to the transformed parameters block, but using alpha_prior instead
  
  array[n] int reward_prior; //Defining reward_prior as this should be updated accodording to the simulated choices
  
  array[n] real expected_value_prior;
  
  array[n] int choice_prior; //Array of lenght n, to store all simulated choices
  
  //Defining the first trial values
  expected_value_prior[1] = 0.5; 
  choice_prior[1] = 1; 
  reward_prior[1] = 1;
  
  real feedback_prior; 
  
  // for-loop
  for (trial in 2:n) {
    //if the agent plays left:
    if (choice_prior[trial-1] == 0)
      feedback_prior = 1 - reward_prior[trial-1];
      
    //if the agent plays right:
    else
      feedback_prior = reward_prior[trial-1];

    expected_value_prior[trial] = expected_value_prior[trial-1] + alpha_prior * (feedback_prior - expected_value_prior[trial-1]);
    
    //Each simualted choice is a Bernoulli draw with probability of EV prior
    choice_prior[trial] = bernoulli_rng(expected_value_prior[trial]);
    
    //Calculate reward for the trial 
    if (choice_prior[trial] == other[trial])
      reward_prior[trial] = 1;
    else 
      reward_prior[trial]= 0;
  }
  
  // Summary: Cumulative Choice Rate
  int prior_sum = sum(choice_prior); 
  
  // ------- Posterior Predictive Check -------
  array[n] int posterior_choices; //define array posterior_choices of length n
  
  //draw choices from bernoulli given the expected value
  for (trial in 1:n) {
    posterior_choices[trial] = bernoulli_rng(expected_value[trial]);
  }
  
  //Summary: Cumulative Choice Rate
  int post_sum = sum(posterior_choices); 
  
  // Log-likelihoods
  vector[n] log_lik; // define vector log_lik of length n
// Compute the log-probability of the observed choice for each trial under a Bernoulli distribution with probability EV
  for (trial in 1:n) {
    log_lik[trial] = bernoulli_lpmf(choice[trial] | expected_value[trial]);
  }

  // Total Log-Prior (for Sensitivity/Update analysis)
  real lprior = normal_lpdf(alpha_logit | prior_alpha_m, prior_alpha_sd); //Compute the log of the normal density of alpha_logit given alpha mu and sd

}
