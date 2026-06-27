from pxr import Gf, Sdf, UsdGeom, UsdPhysics
import torch

from isaaclab.envs import ManagerBasedRLEnv


D6_ENABLE_JOINT = False
D6_DEBUG_ANCHORS = False
LEFT_HAND_ANCHOR_LOCAL_POS = (0.05361310808, -0.00295905240, 0.00215413091)
RIGHT_HAND_ANCHOR_LOCAL_POS = (0.05361310808, 0.00295905240, 0.00215413091)
BOX_LEFT_ANCHOR_LOCAL_POS = (0.09, 0.22, -0.08)
BOX_RIGHT_ANCHOR_LOCAL_POS = (0.09, -0.22, -0.08)


class CoopG1S2Env(ManagerBasedRLEnv):
    """S2 env with a non-interactive payload reference for residual training/playback."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._d6_anchor_refs = [[] for _ in range(self.num_envs)]
        self._bind_hands_to_box()
        self._hand_reference_term = self.command_manager.get_term("hand_reference")
        self._hand_reference_term.align_to_current_state()

    def step(self, action):
        # Rewards are computed after physics and before CommandManager.compute().
        # Advance the reference once here so post-physics tracking uses t + step_dt.
        self._hand_reference_term.update_from_episode_step(preview_steps=1)
        return super().step(action)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_d6_anchor_refs"):
            self._sync_d6_anchors(env_ids)
        if hasattr(self, "_hand_reference_term"):
            self._hand_reference_term.align_to_current_state(env_ids)

    def refresh_hand_reference_alignment(
        self, env_ids: torch.Tensor | list[int] | None = None
    ) -> None:
        """Realign CSV time zero to the current simulated grasp."""
        self._hand_reference_term.align_to_current_state(env_ids)

    def _bind_hands_to_box(self):
        if not D6_ENABLE_JOINT or not hasattr(self.cfg.scene, "hold_box"):
            return

        stage = self.scene.stage

        for env_id in range(self.num_envs):
            env_path = self._env_path(env_id)
            box_path = f"{env_path}/HoldBox"
            self._place_box_relative_to_hand_anchors(stage, env_id)
            self._create_linear_d6_joint(
                stage=stage,
                env_id=env_id,
                hand_path=f"{env_path}/Robot/left_rubber_hand",
                box_path=box_path,
                joint_path=f"{env_path}/LeftHandHoldBoxLinearD6",
                hand_local_pos=LEFT_HAND_ANCHOR_LOCAL_POS,
                box_local_pos=BOX_LEFT_ANCHOR_LOCAL_POS,
            )
            self._create_linear_d6_joint(
                stage=stage,
                env_id=env_id,
                hand_path=f"{env_path}/Robot/right_rubber_hand",
                box_path=box_path,
                joint_path=f"{env_path}/RightHandHoldBoxLinearD6",
                hand_local_pos=RIGHT_HAND_ANCHOR_LOCAL_POS,
                box_local_pos=BOX_RIGHT_ANCHOR_LOCAL_POS,
            )

    def _create_linear_d6_joint(
        self,
        stage,
        env_id: int,
        hand_path: str,
        box_path: str,
        joint_path: str,
        hand_local_pos: tuple[float, float, float],
        box_local_pos: tuple[float, float, float],
    ):
        if stage.GetPrimAtPath(joint_path).IsValid():
            return

        hand_prim = stage.GetPrimAtPath(hand_path)
        box_prim = stage.GetPrimAtPath(box_path)
        if not hand_prim.IsValid():
            raise ValueError(f"D6 parent prim does not exist: {hand_path}")
        if not box_prim.IsValid():
            raise ValueError(f"D6 child prim does not exist: {box_path}")

        joint = UsdPhysics.Joint.Define(stage, joint_path)
        joint_prim = joint.GetPrim()
        joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(box_path)])
        joint.CreateBreakForceAttr().Set(3.4028234663852886e38)
        joint.CreateBreakTorqueAttr().Set(3.4028234663852886e38)
        joint_prim.CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(True)

        hand_local_rot = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        box_local_rot = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*hand_local_pos))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*box_local_pos))
        joint.CreateLocalRot0Attr().Set(hand_local_rot)
        joint.CreateLocalRot1Attr().Set(box_local_rot)

        for axis in ("transX", "transY", "transZ"):
            limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
            limit_api.CreateLowAttr().Set(1.0)
            limit_api.CreateHighAttr().Set(-1.0)

        self._d6_anchor_refs[env_id].append(
            (
                hand_prim,
                box_prim,
                hand_local_pos,
                box_local_pos,
                joint.GetLocalPos0Attr(),
                joint.GetLocalPos1Attr(),
                joint.GetLocalRot0Attr(),
                joint.GetLocalRot1Attr(),
            )
        )

        if D6_DEBUG_ANCHORS and env_id == 0:
            joint_name = joint_path.rsplit("/", maxsplit=1)[-1]
            self._create_anchor_marker(stage, box_path, f"debug_box_anchor_{joint_name}", box_local_pos, joint_name)
            self._create_anchor_marker(stage, hand_path, f"debug_hand_anchor_{joint_name}", hand_local_pos, joint_name)

    def _place_box_relative_to_hand_anchors(self, stage, env_id: int):
        env_path = self._env_path(env_id)
        env_prim = stage.GetPrimAtPath(env_path)
        torso_prim = stage.GetPrimAtPath(f"{env_path}/Robot/torso_link")
        left_hand_prim = stage.GetPrimAtPath(f"{env_path}/Robot/left_rubber_hand")
        right_hand_prim = stage.GetPrimAtPath(f"{env_path}/Robot/right_rubber_hand")
        box_prim = stage.GetPrimAtPath(f"{env_path}/HoldBox")
        if not (
            env_prim.IsValid()
            and torso_prim.IsValid()
            and left_hand_prim.IsValid()
            and right_hand_prim.IsValid()
            and box_prim.IsValid()
        ):
            return

        left_hand_world = UsdGeom.Xformable(left_hand_prim).ComputeLocalToWorldTransform(0.0)
        right_hand_world = UsdGeom.Xformable(right_hand_prim).ComputeLocalToWorldTransform(0.0)
        torso_world = UsdGeom.Xformable(torso_prim).ComputeLocalToWorldTransform(0.0)
        env_world = UsdGeom.Xformable(env_prim).ComputeLocalToWorldTransform(0.0)

        left_anchor_world = left_hand_world.Transform(Gf.Vec3d(*LEFT_HAND_ANCHOR_LOCAL_POS))
        right_anchor_world = right_hand_world.Transform(Gf.Vec3d(*RIGHT_HAND_ANCHOR_LOCAL_POS))
        hand_anchor_mid_world = (left_anchor_world + right_anchor_world) * 0.5

        box_anchor_mid_local = (
            (BOX_LEFT_ANCHOR_LOCAL_POS[0] + BOX_RIGHT_ANCHOR_LOCAL_POS[0]) * 0.5,
            (BOX_LEFT_ANCHOR_LOCAL_POS[1] + BOX_RIGHT_ANCHOR_LOCAL_POS[1]) * 0.5,
            (BOX_LEFT_ANCHOR_LOCAL_POS[2] + BOX_RIGHT_ANCHOR_LOCAL_POS[2]) * 0.5,
        )

        box_world = Gf.Matrix4d().SetRotate(torso_world.ExtractRotation())
        rotated_box_anchor_mid = box_world.Transform(Gf.Vec3d(*box_anchor_mid_local))
        box_world.SetTranslateOnly(hand_anchor_mid_world - rotated_box_anchor_mid)
        box_in_env = box_world * env_world.GetInverse()

        box_xform = UsdGeom.Xformable(box_prim)
        box_xform.ClearXformOpOrder()
        box_xform.AddTransformOp().Set(box_in_env)

    def _sync_d6_anchors(self, env_ids):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = range(self.num_envs)
        if hasattr(env_ids, "detach"):
            env_ids = env_ids.detach().cpu().tolist()

        for env_id in env_ids:
            self._place_box_relative_to_hand_anchors(self.scene.stage, int(env_id))
            for (
                hand_prim,
                box_prim,
                hand_local_pos,
                box_local_pos,
                local_pos0_attr,
                local_pos1_attr,
                local_rot0_attr,
                local_rot1_attr,
            ) in self._d6_anchor_refs[int(env_id)]:
                local_pos0_attr.Set(Gf.Vec3f(*hand_local_pos))
                local_pos1_attr.Set(Gf.Vec3f(*box_local_pos))
                local_rot0_attr.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                local_rot1_attr.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def _env_path(self, env_id: int) -> str:
        env_prim_paths = getattr(self.scene, "env_prim_paths", None)
        if env_prim_paths is not None:
            return env_prim_paths[env_id]
        return f"/World/envs/env_{env_id}"

    @staticmethod
    def _create_anchor_marker(
        stage,
        parent_path: str,
        marker_name: str,
        local_pos: tuple[float, float, float],
        joint_name: str,
    ):
        marker_path = f"{parent_path}/{marker_name}"
        marker = UsdGeom.Sphere.Define(stage, marker_path)
        marker.CreateRadiusAttr(0.06)
        marker.AddTranslateOp().Set(Gf.Vec3f(*local_pos))
        color = Gf.Vec3f(1.0, 0.0, 0.0) if "Left" in joint_name else Gf.Vec3f(0.0, 0.2, 1.0)
        marker.GetDisplayColorAttr().Set([color])
