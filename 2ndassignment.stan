data {
  //int <lower=1> players; // no of players
  
  // trial n is type integer, with a lower boundary of 1
  int<lower=1> n; 
  
  
  array[n] int<lower=0, upper=1> choice; //array "choice" of length n taking the value of either 0 or 1 
  
  array[n] int<lower=0, upper =1> feedback; //array "feedback" of length n taking the value of either 0 or 1
  
  //then we create an array "right" of dimension n x players, that takes the value of either 0 or 1 
  // array[n, players] int<lower=0, upper=1> choice; 
}

parameters {
  // Learning rate aLpha is a continous number (real) between 0 and 1
  real alpha_logit; 
  
  // The exploratory-ness of the model, inv_temp, is also a real number always 
  // greater than 0 and (probably) lower than 20
  // real <lower=0, upper=20> inv_temp;
  
  // tau_logit;
}

transformed parameters {
  
  real <lower=0,upper=1> alpha;
  alpha = inv_logit(alpha_logit);
  
  OR
  
  real alpha = inv_logit(alpha_logit);
  
  
  array[n] expected_value;
  
  expected_value[1] = 0.5;
  
  for (trial in 1:n {
     expected_value[trial] = expected_value[trial-1] + alpha * (feedback[trial-1] - expected_value[trial-1])
  }
  // real tau20 = inv-logit(tau_logit)*20;
  //p = softmax(expected_value, tau)
  
}

model {
  //Prior: Our belief about alpha and inverse temperature before seeing the data
  //If we use a beta(1,1) prior, its equevalent to a UnIform(1,1) function
  // lpdf: log-probability density function
  //target += beta_lpdf(paramter_here *linje her *1,1); 
  // If instead we assume a gaussian distribution it would look like
  target += normal_lpdf(alpha_logit|0, 1.5)
  
  // Likelihood: How the data 'h' depend on the parameter 'theta'
  // The model assesses how likely the observed sequence 'h' is given a value of 'theta' in a Bernoulli distribution
  target += bernoulli_logit_lpdf(choice|expected_value) //Indexing??
  //target += bernoulli_logit_lpdf(choice|p)
}

generated quantities {
  // Space for playing around with what different priors do to our outcome
  // Here we can also make and rund predictions
  
}