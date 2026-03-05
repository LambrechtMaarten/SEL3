import jax

from src.cpg import create_cpg, map_cpg_outputs_to_actions, modulate_cpg
from src.env import Environment
from src.render import save_video, post_render

env = Environment()

cpg = create_cpg()
cpg_state = cpg.reset(rng=jax.random.PRNGKey(0))
cpg_state = modulate_cpg(cpg_state=cpg_state, leading_arm_index=0, max_joint_limit=env.action_space.high[0] * 1)

frames = []
env_state = env.reset()
while not (env_state.terminated | env_state.truncated):
    cpg_state = cpg.step(state=cpg_state)
    actions = map_cpg_outputs_to_actions(cpg_state=cpg_state)
    env_state = env.step(actions)
    frame = env.render()
    frames.append(frame)

save_video(frames, "../output/video.mp4")
