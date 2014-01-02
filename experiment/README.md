# Mental Rotation Experiment

## Version A

### Design

8 conditions with 90 trials each (plus 10 training trials). Each of
the 20 stimuli is repeated 9 times and each rotation (0-340) is
repeated 10 times (5 same, 5 flipped), giving a block of 180 trials,
which is then split in half.

### Data

Collected data for 16 participants.

### Notes/Issues

* High error rate (9/16 subjects had at least 75% accuracy, 7/16 have
  at least 85% accuracy, and 4/16 have at least 90% accuracy).
* Fewer data points for 0-degree and 180-degree rotations, because all
  other rotations have a corresponding angle (e.g. 20 and 340).


## Version B

### Design

8 conditions with 100 trials each (plus 10 training trials). Each of
the 20 stimuli is repeated 10 times and each rotation (0-340, plus
another repetition of 0 and 180) is repeated 10 times (5 same, 5
flipped), giving a block of 200 trials, which is then split in half.

### Data

Collected data for 20 participants.

### Notes/Issues

* Somewhat better error rates (only 5 had less than 75% accuracy)
* But, there was an issue where people could not submit the HIT. Still
  investigating why this was the case. Update: I *think* this is
  because I had the "use_sandbox" option set in PsiTurk, whch means
  that it tried to submit to the worker sandbox rather than Turk
  itself.
* Also, it appears that you only ever get one flipped/same version of
  each stimulus -- i.e., there are different rotations, but each
  participant only sees each stimulus as flipped or same.


## Version C

### Design

8 conditions with 100 trials each (plus 10 training trials). Each of
the 20 stimuli is repeated 10 times and each rotation (0-340, plus
another repetition of 0 and 180) is repeated 10 times (5 same, 5
flipped), giving a block of 200 trials, which is then split in half.

Fixed the bug with trial balancing that was in Version B.

### Data



### Notes/Issues

