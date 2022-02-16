This code repo differs from the original [hanabi_SAD](https://github.com/facebookresearch/hanabi_SAD) project.
- We modify the hanabi environment to provide player ID's as observations to be able to assign "accommodator" and "specializer" roles.
- We modify the hanabi environment to provide a randomly drawn laten tvariable as observations as intent for the "specializer" role.
- We expand the cross-play scripts to be able to play multiple different algorithms against each other
- We create a selfplay_reloader.py script and adjust the instrinsic reward weight and reload selfplay if selfplay is not converging.
- We augment agents with Any-Play.

For a comprehensive list of changes made since forking from hanabi_SAD, run command:

```bash
git diff 4892df62633c11b99c09b6871015a9ebaa338618..HEAD
```

Or if you have `difftool` configured

```bash
git difftool --dir-diff 4892df62633c11b99c09b6871015a9ebaa338618..HEAD
```
