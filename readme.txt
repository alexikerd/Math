I watched a youtube video by numberphile that interested me.

Take a number, and multiply every digit together to get another number, and repeat the process until you get a single digit number

There must be a number that has the most amount of steps required, and the video said that the most steps we have is 11 with the number 2377777788888899

I decided to try to train an AI to be able to explore the hilbert space of numbers (each dimension is a digit) and figure out the most amount of steps possible

I did this by training an AI to adjust a number to a maximum length of 20 digits and evaluate the value step by step.

In theory, if I chained these steps together in training and then plugged in a bunch of random numbers I'd end up with some local minima but also 2377777788888899.

Unfortunately, the feature space is so barren of distinctions in rewards that it keeps finding the best strategy is to add either 8's or 9's to the end of the number for the most part