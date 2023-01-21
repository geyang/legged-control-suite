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
    env = gym.make('lcs:bipedal_v1')
    env.reset()
    env.seed(42)
    img = env.render('rgb_array')

doc @ "## Model XML"
with doc:
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, 'bipedal_v1.xml')
        core.save_last_parsed_model_to_xml(fname, check_model=env.unwrapped.env.physics.model)

        with open(fname, 'r') as f:
            xml = f.read()

doc.print(xml)

doc @ "## Parameter hacking"

with doc:
    doc.print(env.unwrapped.env.physics.named.model.body_mass)

with doc:
    env.reset(cart_mass=2.0, pole_mass=0.5) env.reset()
    doc.print(env.unwrapped.env.physics.named.model.body_mass)

with doc:
    env.reset()
    doc.print(env.unwrapped.env.physics.named.model.body_mass)

doc @ "## Run simulation"
with doc, doc.table().figure_row() as r:
    position_history = []
    for _ in range(100):
        action = np.full(env.action_space.shape, 0.1)
        obs, reward, done, info = env.step(action)
        position_history.append(obs)

    img = env.render('rgb_array', camera_id=1)
    r.figure(img, f"{Path(__file__).stem}/bipedal.png", zoom="400%", title="bipedal", caption="")

    #env.reset(pole_length=0.5, cart_mass=2.0)

    img = env.render('rgb_array', camera_id=1)
    r.figure(img, f"{Path(__file__).stem}/bipdeal.png", zoom="400%", title="bipdeal", caption="")

    for _ in range(100):
        action = np.full(env.action_space.shape, 0.1)
        obs, reward, done, info = env.step(action)
        position_history.append(obs)

    position_history = np.stack(position_history)

    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 9))

    plt.figure(figsize=[13, 9])

    pos_names = ['cart_position', 'pole_cos', 'pole_sin', 'cart_velocity', 'pole_velocity']
    for i, pos_name in enumerate(pos_names):
        # plt.sca(axes[i // 3][i % 3])
        plt.subplot(3, 3, i + 1)
        plt.plot(np.arange(position_history.shape[0]), position_history[:, i])
        plt.xlabel('step')
        plt.ylabel(pos_name)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.5)

    r.savefig(f"{Path(__file__).stem}/position_history.png", title="position_history")

env.close()
