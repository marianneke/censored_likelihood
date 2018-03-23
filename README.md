# Maximum Likelihood Estimation with censored data

Right-censoring often occurs in survival (or: time-to-event) data, where there
are subjects whose time of death is observed, and subjects who are still alive
at the time of measurement. For the latter group, we have partial estimation on
their total lifetime, since we know how long they have been alive, but we do
not know for certain how long they will end up living - just that it is at
least the number of years they have been alive. We call the value we have
recorded for their lifetime "right-censored", since the true value lies
somewhere to the right of the recorded value.

When performing a naive OLS regression on a target variable that has been
right-censored, we end up with estimates that are always too low. This can be
corrected by performing a maximum likelihood estimate of the parameters where
we assume that all data is normally distributed with finite mean and variance,
just as in OLS regression, but treat observed and censored data points
differently:
 - if a data point has been observed, we compute the log-likelihood as the log
   of the pdf
 - if a data point is censored, we compute the log-likelihood as the log of the
   survival function, with the survival function `S(t)` being defined as the
   probability of the time-of-death being to the _right_ of time `t`:

   ```S(t) = 1 - CDF(t)```
