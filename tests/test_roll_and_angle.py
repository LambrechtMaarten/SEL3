import jax.numpy as jnp

def test_roll_symmetry():
    """
    Simuleer: robot moet naar arm k bewegen.
    - Canoniseer obs (roll joints naar sector 0)
    - Netwerk geeft actie in sector 0 frame
    - Decanoniseer actie (roll terug naar wereldframe)
    - Verwacht: actie == originele expert actie voor arm k
    """

    n_joints_per_arm = 6
    n_arms = 5

    # Stel: expert actie voor arm 0 (wereldframe, sector 0)
    expert_action_arm0 = jnp.array([float(i) for i in range(30)])

    # Expert actie voor arm k = roll naar rechts met k*6
    # (arm k heeft zijn joints verschoven tov arm 0)
    for k in range(n_arms):
        print(f"\n--- Test k={k} (arm {k}, {k*72}°) ---")

        # Wereldframe expert actie voor arm k
        expert_action_armk = jnp.roll(expert_action_arm0, k * n_joints_per_arm)

        # Stap 1: canoniseer joints in obs (roll naar links met k)
        joints_canonical = jnp.roll(expert_action_armk, -k * n_joints_per_arm)

        # Stap 2: netwerk output in sector 0 frame
        # (we simuleren: netwerk leert perfect -> output == arm0 expert)
        network_output = joints_canonical  # perfecte kloon van arm 0

        # Stap 3: decanoniseer actie naar wereldframe
        action_world = jnp.roll(network_output, k * n_joints_per_arm)

        # Check: moet gelijk zijn aan de expert actie voor arm k
        ok = jnp.allclose(action_world, expert_action_armk)
        print(f"  expert_armk:    {expert_action_armk[:6]}")
        print(f"  canonical:      {joints_canonical[:6]}")
        print(f"  network_output: {network_output[:6]}")
        print(f"  action_world:   {action_world[:6]}")
        print(f"  ✅ OK" if ok else f"  ❌ FAIL")
        assert ok, f"Roll symmetry broken for k={k}"

    print("\n✅ Alle tests geslaagd")


def test_local_angle_range():
    """
    Check dat local_angle altijd in [-36°, 36°] valt
    voor willekeurige relative_angles.
    """
    sector_size = 2 * jnp.pi / 5

    test_angles = jnp.linspace(-2*jnp.pi, 2*jnp.pi, 1000)

    for angle in test_angles:
        angle_norm = jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi

        k = int(jnp.round(angle_norm / sector_size))  # kan -2, -1, 0, 1, 2 zijn
        local_angle = angle_norm - k * sector_size
        print(jnp.degrees(angle), jnp.degrees(local_angle), k)

        in_range = jnp.abs(local_angle) <= (sector_size / 2 + 1e-6)
        assert in_range, (
            f"local_angle {jnp.degrees(local_angle):.2f}° buiten [-36°, 36°] "
            f"voor angle={jnp.degrees(angle):.2f}°, angle_norm={jnp.degrees(angle_norm):.2f}°, k={k}"
        )
    print("✅ local_angle altijd binnen [-36°, 36°]")


if __name__ == "__main__":
    test_roll_symmetry()
    test_local_angle_range()