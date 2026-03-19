data {
  int<lower=1> n; // trial n is type integer, with a lower boundary of 1
  
  array[n] int<lower=0, upper=1> choice; //array "choice" of length n taking the value of either 0=left or 1=right 
  
  array[n] int<lower=0, upper=1> reward; //array "reward" of length n taking the value of either 0=RL agent lost or 1=RL agent won 
}

parameters {
  real alpha_logit; // Learning rate aLpha_logit is a continous number (real)
}

transformed parameters {
  real alpha = inv_logit(alpha_logit); //Learning rate alpha is the inverse log-odds of our "free" alpha
  
  array[n] real expected_value; // array "expected_value" of length n
  
  expected_value[1] = 0.5; // define expected_value on trial 1
  
  real feedback;
  
  // for-loop calculating the expected value for each trial
  for (trial in 2:n) {
    //if the agent plays left
    if (choice[trial-1] == 0)
      feedback = 1 - reward[trial-1];
    else
      feedback = reward[trial-1];

    expected_value[trial] = expected_value[trial-1] + alpha * (feedback - expected_value[trial-1]);
  }
}

model {
  //Prior: Our belief about alpha before seeing the data
  target += normal_lpdf(alpha_logit|0, 1.5);
  
  // Likelihood: How the data 'choice' depend on the parameter 'expected value'
  target += bernoulli_lpmf(choice|expected_value); //Indexing??
}

generated quantities {
  // sample prior vals for alpha
  real alpha_logit_prior = normal_rng(0, 1.5);
  real<lower=0, upper=1> alpha_prior = inv_logit(alpha_logit_prior);
  
  //posterior predictive check
  array[n] int posterior_choices;
  
  //draw choices from bernoulli
  for (i in 1:n) {
    posterior_choices[i] = bernoulli_rng(expected_value[i]);
  }
}
