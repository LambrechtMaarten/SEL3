import jax
from jax import numpy as jnp

from configs.config import Configuration
from src.cpg.cpg import CPG
from src.cpg.cpg_generators.cpg_generators import CPGGenerator
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


class FullyConnectedSymmetricCPGGenerator(CPGGenerator):
    """
    This class has 2 generators per motor
    """

    def generate(self, configuration: Configuration) -> CPG:
        morphology = configuration.simulation.morphology_configuration
        adjacency_matrix = jnp.ones((
            2 * morphology.number_of_arms * morphology.number_of_segments_per_arm[0],
            2 * morphology.number_of_arms * morphology.number_of_segments_per_arm[0]))

        return CPG(1 * adjacency_matrix, configuration)

    def outputs_to_actions(self, _outputs: jarr, configuration: Configuration) -> jarr:

        def negate_even_indices(a):
            indices = jnp.arange(len(a))
            mask = indices % 2 == 0
            return jnp.where(mask, -a, a)

        outputs = jnp.zeros_like(_outputs).astype(_outputs.dtype)

        # leading arm
        outputs = outputs.at[jnp.arange(len(_outputs)//5*0,len(_outputs)//5*1)].set(_outputs[jnp.arange(len(_outputs)//5*0,len(_outputs)//5*1)])

        # front arms
        outputs = outputs.at[jnp.arange(len(_outputs)//5*1,len(_outputs)//5*2)].set(_outputs[jnp.arange(len(_outputs)//5*1,len(_outputs)//5*2)])
        outputs = outputs.at[jnp.arange(len(_outputs)//5*4,len(_outputs)//5*5)].set(negate_even_indices(_outputs[jnp.arange(len(_outputs)//5*1,len(_outputs)//5*2)]))

        # back arms
        outputs = outputs.at[jnp.arange(len(_outputs)//5*2,len(_outputs)//5*3)].set(_outputs[jnp.arange(len(_outputs)//5*2,len(_outputs)//5*3)])
        outputs = outputs.at[jnp.arange(len(_outputs)//5*3,len(_outputs)//5*4)].set(negate_even_indices(_outputs[jnp.arange(len(_outputs)//5*2,len(_outputs)//5*3)]))

        return outputs

    def modulate_symmetric_rotation(self, cpg_state: CPGState, clockwise_rotations: int) -> CPGState:
        raise Exception("Not implemented")
