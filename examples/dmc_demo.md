```python
env = gym.make('dmc:CartPole-v1')
env.reset()
env.seed(42)
img = env.render('rgb_array')
doc.savefig(f"{Path(__file__).stem}/cartpole.png")
```

<img style="align-self:center;" src="dmc_demo/cartpole.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>
