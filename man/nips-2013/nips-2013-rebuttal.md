# NIPS 2013 Rebuttal

**Paper ID**: 1352  
**Title**: Mental Rotation as Bayesian Quadrature

## Reviewer 3

Reviewer 3 was primarily concerned with the comparison between models
and human data. While we did not collect human data ourselves, we
compared our model to the human response times reported by Shepard and
Metzler [6] (the key comparison is between Figure 1, depicting the
classic results, and Figure 5, illustrating our modeling results).
The mental rotation result is one of the most famous findings in
cognitive science ([6] alone has 2,061 citations according to Web of
Knowledge; 3,640 according to Google Scholar) and has been replicated
many times with many types of stimuli (Thomas, 2013). As a
proof-of-concept, we showed a high correlation between our model and a
linear function (which is the same form as the classic result).  No
other computational models have been proposed which can explain how
people choose the correct direction and duration of rotation.

The reviewer was also concerned about how the prior over stimuli
related to the process by which the shapes were created. To clarify,
the shapes were generated in a manner consistent with the prior:
vertex angles were chosen uniformly at random between 0 and 2*pi, and
vertex radii were chosen uniformly at random between 0 and 1.  Edges
were drawn between vertices which were adjacent when sorted by angle.

## Reviewer 4

Reviewer 4 was concerned with the explanatory power of our model,
suggesting that people could "just rotate until they find a match that
is easy to recognize or exhaust the set of potential
rotations". However, it is not obvious how this strategy should be
implemented. What does it mean for a match to be easy to recognize?
Similarly, stopping when a local maximum is found is not equivalent to
quitting "even if it's obviously not a match at that rotation".  A
perfect match cannot truly be identified due to perceptual uncertainty
in people's visual system, and one cannot just rotate until a
threshold of similarity has been reached, because the similarity
function is not normalized and therefore the value of its global
maximum is not known *a priori*.

If the value of the global maximum of the similarity function is not
known, then it is impossible to know if any local maximum is the
global maximum or not. To reliably find the global maximum, then,
would require an exhaustive search (a full rotation of 360 degrees),
which is inconsistent with the linear nature of human response times
from [6]. Thus, it is necessary to either be satisfied with a local
maximum (as found by hill-climbing), or to use a more sophisticated
method.

Even if it is possible to perfectly recognize a match after rotating
the appropriate amount, there is a separate issue: how do people know
which direction to rotate? Choosing one direction at random and
rotating to the true angle produces a pattern of data which is
inconsistent with the behavioral data from [6]. For any true minimum
angle r, we would have expected response times proportional to E[r] =
0.5r + 0.5(2*pi - r) = pi, which is constant rather than monotonic.

The "model rotations" described around line 372 and shown in Figure 5
do refer to the number of steps the models takes. A more precise label
would be "total distance rotated, in degrees" which is directly
proportional to the number of steps the model takes.

To address the reviewer's question about whether the behavioral
results hold for our stimuli, we emphasize (as above in response to
Reviewer 3) that the Shepard and Metzler results [6] are very robust
and have been replicated with many other types of stimuli. In
particular, our stimuli are similar to those used by [23].

The reviewer correctly notes that the 95% confidence threshold used by
the BQ model's hypothesis test is likely the source of the 4% error
rate. The reviewer also points out that the MSE on line 382 is
inconsistent with our definition of MSE. This is indeed a mistake: the
MSE cannot be normalized between 0 and 1 because it has no upper
bound. The correct, non-normalized MSEs are 0.03 for the naive model,
0.002 for the parametric model, and 0.001 for the nonparametric model.

## Reviewer 5

We thank Reviewer 5 for their encouraging comments. We do intend to
collect human data in the future and to perform a more detailed,
item-specific comparison between the model and participants. We
further hope that the model will be useful in uncovering previously
unexamined nuances in people's behavior.

Regarding people's mental representations, we do not think that people
are necessarily performing full Bayesian quadrature in their
heads. Ours is a computational-level model [3], which means that it is
a formal way of describing how to solve the problem of determining if
two objects are the same. Although we incorporate the constraint of
sequential rotations, this is not an algorithmic-level constraint
because we have not given specific limits on working memory,
processing power, knowledge representation, etc. The algorithmic-level
form of this solution would be an approximation to our
computational-level model and would make the resource limitations
explicit. Currently, however, we have not explored possible
algorithmic approximations. Doing so is a direction for future work.

# References

* Thomas, Nigel J.T., "Supplement: Mental Rotation", *The Stanford
  Encyclopedia of Philosophy*, Edward N. Zalta (ed.). Retrieved on
  08/05/2013 from
  http://plato.stanford.edu/archives/spr2013/entries/mental-imagery/mental-rotation.html

* [3] D. Marr, *Vision: A Computational Investigation into the Human
  Representation and Processing of Visual Information*. Henry Holt and
  Company, 1983.

* [6] R. N. Shepard and J. Metzler, "Mental Rotation of
  Three-Dimensional Objects," *Science*, vol. 171, no. 3972,
  pp. 701–703, 1971.

* [23] L. A. Cooper, "Mental Rotation of Random Two-Dimensional
  Shapes," *Cognitive Psychology*, vol. 7, pp. 20-43, 1975.
