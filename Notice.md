This code repo differs from the original hanabi_SAD project.
- We modify the hanabi environment to provide player ID's as observations to be able to assign "accommodator" and "specializer" roles.
- We modify the hanabi environment to provide a randomly drawn laten tvariable as observations as intent for the "specializer" role.
- We expand the cross-play scripts to be able to play multiple different algorithms against each other
- We create a selfplay_reloader.py script and adjust the instrinsic reward weight and reload selfplay if selfplay is not converging.
- We augment agents with Any-Play.