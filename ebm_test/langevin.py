import tensorflow as tf
import pdb

def sample_langevin(x, model, stepsize, n_steps, noise_scale=None, intermediate_samples=False):
    """Draw samples using Langevin dynamics
    x: tf.Tensor, initial points
    model: An energy-based model
    stepsize: float
    n_steps: integer
    noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
    """
    if noise_scale is None:
        noise_scale = tf.sqrt(stepsize * 2.0)

    l_samples = []
    l_dynamics = []
    x = tf.Variable(x, trainable=True, dtype=tf.float32)
    
    pdb.set_trace()

    for _ in range(n_steps):
        l_samples.append(tf.identity(x).numpy())
        noise = tf.random.normal(shape=x.shape) * noise_scale
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = model(x)
        grad = tape.gradient(tf.reduce_sum(out), x)
        dynamics = stepsize * grad + noise
        x.assign_add(dynamics)
        l_samples.append(tf.identity(x).numpy())
        l_dynamics.append(tf.identity(dynamics).numpy())

    if intermediate_samples:
        return l_samples, l_dynamics
    else:
        return l_samples[-1]
