# Legged Control Suite

A collection of legged robot environments built upon the DeepMind control
suite and MuJoCo physics engine.

## Environments

- Hopper (3D)
- Bipedal (3D)
- Agile Bipedal Robot (ours)

## To Do

As excersize, fork this repo and hack around the walker environment and 
  make a simple bipedal walker. The `lcs` module contains two walker 
  scripts copied from DeepMind Control Suite. You need to:

- [ ] First be able to load `lcs:BipedalWalker-v0`. This is currently not 
      runnable.
- [ ] Change the name to `BipedalWalker` and add the following tasks: `["walk",
      "run", "hop"]`