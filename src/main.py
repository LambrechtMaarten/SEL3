import jax

from src.cpg import create_cpg, map_cpg_outputs_to_actions, modulate_cpg
from src.env import create_environment, environment_configuration
from src.render import save_video, post_render

env = create_environment()
jit_step = jax.jit(env.step)
jit_reset = jax.jit(env.reset)

cpg = create_cpg()
cpg_state = cpg.reset(rng=jax.random.PRNGKey(0))
# We set the max_joint_limit to only 25% of the true joint range of motion (you can test yourself what happens if we don't by changing this value).
cpg_state = modulate_cpg(cpg_state=cpg_state, leading_arm_index=0, max_joint_limit=env.action_space.high[0] * 1)

done = False
frames = []
env_state = jit_reset(rng=jax.random.PRNGKey(seed=0))
while not (env_state.terminated | env_state.truncated):
    cpg_state = cpg.step(state=cpg_state)
    actions = map_cpg_outputs_to_actions(cpg_state=cpg_state)
    env_state = jit_step(state=env_state, action=actions)
    frame = post_render(env.render(state=env_state), environment_configuration=environment_configuration)
    frames.append(frame)
save_video(frames, "../output/video.mp4")
