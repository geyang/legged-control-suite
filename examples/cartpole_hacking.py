from cmx import doc

doc @ "# DMC environment hacking"

with doc:
    import os
    from pathlib import Path
    import tempfile

    import numpy as np
    import dm_control
    from dm_control.mujoco.wrapper import core
    from dm_control.mujoco.wrapper.mjbindings import mjlib
    import gym

    import matplotlib.pyplot as plt


with doc:
    env = gym.make('lcs:Paramcartpole-swingup-v1')
    env.reset()
    env.seed(42)
    img = env.render('rgb_array')

doc @ "## Model XML"
with doc:
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, 'model.xml')
        core.save_last_parsed_model_to_xml(fname, check_model=env.unwrapped.env.physics.model)

        with open(fname, 'r') as f:
            xml = f.read()

doc.print(xml)

doc @ "## Parameter hacking"

with doc:
    doc.print(env.unwrapped.env.physics.named.model.body_mass)

with doc:
    env.reset(cart_mass=2.0, pole_mass=0.5)
    doc.print(env.unwrapped.env.physics.named.model.body_mass)

with doc:
    env.reset()
    doc.print(env.unwrapped.env.physics.named.model.body_mass)

doc @ "## Run simulation"
with doc:
    position_history = []
    for _ in range(100):
        action = np.full(env.action_space.shape, 0.1)
        obs, reward, done, info = env.step(action)
        position_history.append(obs)

    img = env.render('rgb_array', camera_id=1)
    doc.figure(img, f"{Path(__file__).stem}/cartpole_0.png", zoom="400%", title="Cartpole", caption="")

    env.reset(pole_length=0.5, cart_mass=2.0)

    img = env.render('rgb_array', camera_id=1)
    doc.figure(img, f"{Path(__file__).stem}/cartpole_1.png", zoom="400%", title="Cartpole", caption="")

    for _ in range(100):
        action = np.full(env.action_space.shape, 0.1)
        obs, reward, done, info = env.step(action)
        position_history.append(obs)

    position_history = np.stack(position_history)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 9))

    pos_names = ['cart_position', 'pole_cos', 'pole_sin', 'cart_velocity', 'pole_velocity']
    for i, pos_name in enumerate(pos_names):
        plt.sca(axes[i // 3][i % 3])
        plt.plot(np.arange(position_history.shape[0]), position_history[:, i])
        plt.xlabel('step')
        plt.ylabel(pos_name)
    plt.subplots_adjust(wspace=0.5)

    doc.savefig(f"{Path(__file__).stem}/position_history.png")


doc @ "## Resetting parameters while preserving state"
with doc:
    env.reset()


    position_history = []
    for _ in range(100):
        action = np.full(env.action_space.shape, 0.1)
        obs, reward, done, info = env.step(action)
        position_history.append(obs)

    img = env.render('rgb_array', camera_id=1)
    doc.figure(img, f"{Path(__file__).stem}/cartpole_state_0.png", zoom="400%", title="Cartpole", caption="")

    # env.reset(pole_length=0.5, cart_mass=2.0)
    env.unwrapped.env.change_model(pole_length=0.5, cart_mass=2.0)


    img = env.render('rgb_array', camera_id=1)
    doc.figure(img, f"{Path(__file__).stem}/cartpole_state_1.png", zoom="400%", title="Cartpole", caption="")

    for _ in range(100):
        action = np.full(env.action_space.shape, 0.1)
        obs, reward, done, info = env.step(action)
        position_history.append(obs)

    position_history = np.stack(position_history)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 9))

    pos_names = ['cart_position', 'pole_cos', 'pole_sin', 'cart_velocity', 'pole_velocity']
    for i, pos_name in enumerate(pos_names):
        plt.sca(axes[i // 3][i % 3])
        plt.plot(np.arange(position_history.shape[0]), position_history[:, i])
        plt.xlabel('step')
        plt.ylabel(pos_name)
    plt.subplots_adjust(wspace=0.5)

    doc.savefig(f"{Path(__file__).stem}/position_history_state.png")


with doc:
    env.close()
