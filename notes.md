# spec = env.action_spec()
# force is relate to energy expenditure
# which could be used for the reward function (minimal energy expenditure)


$r_{energy} = \sum_{i=1}^{n_joints} (1-|q_{ddot}[i]|)^2$


## mias experiment viscosity density velocity relationships

## maybe rewrite trainer.run to return more state space info

## what happens when we impose bounds on the upper velocity

## does using a reward function based on energy expenditure gets rid of the reward hacking problem?


## bounded actuators

from dm_control.mujoco.engine

```python
def action_spec(physics):
  """Returns a `BoundedArraySpec` matching the `physics` actuators."""
  num_actions = physics.model.nu
  is_limited = physics.model.actuator_ctrllimited.ravel().astype(bool)
  control_range = physics.model.actuator_ctrlrange
  minima = np.full(num_actions, fill_value=-mujoco.mjMAXVAL, dtype=float)
  maxima = np.full(num_actions, fill_value=mujoco.mjMAXVAL, dtype=float)
  minima[is_limited], maxima[is_limited] = control_range[is_limited].T

  return specs.BoundedArray(
      shape=(num_actions,), dtype=float, minimum=minima, maximum=maxima)
```

from dm_control.suite.base

```python
  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    return mujoco.action_spec(physics)
```

from dm_control.rl.control

```python
  @abc.abstractmethod
  def action_spec(self, physics):
    """Returns a specification describing the valid actions for this task.

    Args:
      physics: Instance of `Physics`.

    Returns:
      A `BoundedArraySpec`, or a nested structure containing `BoundedArraySpec`s
      that describe the shapes, dtypes and elementwise lower and upper bounds
      for the action array(s) passed to `self.step`.
    """
```

The task of maintaining a particular velocity box may be unnatural, maybe thats why ncap is not that much better there.


> physics.named.data.qfrc_actuator is the total force exerted on the joint. actuator_force is the motor input; Another pod found out (as an alternative to get the last layer outputs for calculating the reward function we discussed.

how do you know which strategy the worm is using?
you can see the policty in different states, softmax in the last layer, it may be that some joints have converged (exploitation ) or not (exploration)

for the target task, the reward function should be sparse, so that the agent can learn to explore the state space. (thats why distance based rewards are not good).

a sparse reward function could be being at a certain distance from the target, or being in a certain region of the state space.

Expplore the energy expenditure thing.