# Artificial Neural Networks with stochastic gradient descent optimization

Single layer neural network to predict the gender of crabs. Functions can be scaled to n-layers.

## Installation

Download from https://github.com/tim-hoff/ann

## Usage

You can get output for feeding the crab data (http://vincentarelbundock.github.io/Rdatasets/csv/MASS/crabs.csv) (http://vincentarelbundock.github.io/Rdatasets/doc/MASS/crabs.html) through the trained neural network by running:

    $ java -jar target/uberjar/ann-0.1.0-SNAPSHOT-standalone.jar

## Options


## Examples
If you're feeling brave enough to use the repl, you can try: 

    (error-check crabv (refeed crabv2 (first (weight-gen `(7 1))) [0.2 0.1 0.05 0.025 0.01]) 0.5)

### Bugs
None. :)

## License

Copyright © 2015 Timothy Hoff

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
