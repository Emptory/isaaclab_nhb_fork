"""S2 action terms that attach physics-rate virtual-environment updates."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class VirtualSpringJointPositionAction(JointPositionAction):
    """Apply joint targets, then refresh S2's virtual spring before simulation.

    Isaac Lab calls :meth:`apply_actions` once per physics substep and writes
    asset buffers to PhysX immediately afterwards. This gives the virtual
    spring the same 200 Hz placement as the reference implementation without
    duplicating ``ManagerBasedRLEnv.step`` and its recorder/reset lifecycle.
    """

    def apply_actions(self) -> None:
        super().apply_actions()
        spring = getattr(self._env, "_virtual_spring", None)
        if spring is None:
            raise RuntimeError(
                "VirtualSpringJointPositionAction requires env._virtual_spring. "
                "The S2 environment did not initialize its virtual force manager."
            )
        spring.apply_physics_substep()


@configclass
class VirtualSpringJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the S2 joint action with a physics-rate spring hook."""

    class_type: type = VirtualSpringJointPositionAction
