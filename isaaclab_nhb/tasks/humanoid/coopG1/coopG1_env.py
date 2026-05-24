import math
import omni.physx.scripts.utils as physx_utils
from pxr import UsdPhysics, Gf, UsdGeom
from isaaclab.envs import ManagerBasedRLEnv

D6_TARGET_STIFFNESS = 50.0
D6_TARGET_DAMPING = 10.0
D6_RAMP_DURATION_S = 2.0
D6_ENABLE_JOINT = True
D6_USE_DRIVE = True
D6_DEBUG_ANCHORS = False

class CoopG1Env(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._d6_drive_attrs = [[] for _ in range(self.num_envs)]
        self._d6_anchor_refs = [[] for _ in range(self.num_envs)]
        self._bind_robots_to_box()

    def step(self, action):
        if D6_USE_DRIVE:
            self._update_d6_drive_ramp()
        return super().step(action)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._sync_d6_anchors(env_ids)

    def _bind_robots_to_box(self):
        if not D6_ENABLE_JOINT or not hasattr(self.cfg.scene, "box"):
            return
            
        stage = self.scene.stage
        num_robots = getattr(self.cfg, "num_robots", 1)

        for env_id in range(self.num_envs):
            env_path = f"/World/envs/env_{env_id}"
            box_path = f"{env_path}/Box" 
            
            for i in range(num_robots):
                left_hand_path = f"{env_path}/Robot_{i}/left_rubber_hand"
                right_hand_path = f"{env_path}/Robot_{i}/right_rubber_hand"
                
                self._create_high_stiffness_joint(stage, env_id, left_hand_path, box_path, f"joint_L_{i}", i, num_robots, "left")
                self._create_high_stiffness_joint(stage, env_id, right_hand_path, box_path, f"joint_R_{i}", i, num_robots, "right")

    def _create_high_stiffness_joint(self, stage, env_id, hand_path, box_path, joint_name, robot_index, num_robots, hand_side):
        from pxr import UsdPhysics, Gf, UsdGeom
        hand_prim = stage.GetPrimAtPath(hand_path)
        box_prim = stage.GetPrimAtPath(box_path)
        
        if not hand_prim or not box_prim:
            return

        joint = physx_utils.createJoint(stage, "D6", from_prim=hand_prim, to_prim=box_prim)
        d6_prim = joint.GetPrim()

        box_local_pos = self._box_side_anchor_local_pos(robot_index, num_robots, hand_side)
        hand_local_pos, hand_local_rot, box_local_rot = self._matching_hand_local_pos(hand_prim, box_prim, box_local_pos)
        physics_joint = UsdPhysics.Joint(d6_prim)
        
        physics_joint.GetLocalPos0Attr().Set(hand_local_pos)
        physics_joint.GetLocalPos1Attr().Set(Gf.Vec3f(*box_local_pos))
        physics_joint.GetLocalRot0Attr().Set(hand_local_rot)
        physics_joint.GetLocalRot1Attr().Set(box_local_rot)
        
        self._d6_anchor_refs[env_id].append((hand_prim, box_prim, box_local_pos, physics_joint.GetLocalPos0Attr(), physics_joint.GetLocalPos1Attr(), physics_joint.GetLocalRot0Attr(), physics_joint.GetLocalRot1Attr()))
        if D6_DEBUG_ANCHORS and env_id == 0:
            self._create_box_anchor_marker(stage, box_path, joint_name, box_local_pos, hand_side)
        
        if D6_USE_DRIVE:
            for axis in ["transX", "transY", "transZ"]:
                drive = UsdPhysics.DriveAPI.Apply(d6_prim, axis)
                stiffness_attr = drive.CreateStiffnessAttr(0.0)
                damping_attr = drive.CreateDampingAttr(0.0)
                self._d6_drive_attrs[env_id].append((stiffness_attr, damping_attr))

    def _update_d6_drive_ramp(self):
        ramp_steps = max(int(D6_RAMP_DURATION_S / self.step_dt), 1)
        episode_steps = self.episode_length_buf.detach().cpu().tolist()

        for env_id, drive_attrs in enumerate(self._d6_drive_attrs):
            ratio = min(float(episode_steps[env_id]) / ramp_steps, 1.0)
            stiffness = D6_TARGET_STIFFNESS * ratio
            damping = D6_TARGET_DAMPING * ratio
            for stiffness_attr, damping_attr in drive_attrs:
                stiffness_attr.Set(stiffness)
                damping_attr.Set(damping)

    def _sync_d6_anchors(self, env_ids):
        from pxr import Gf
        if hasattr(env_ids, "detach"):
            env_ids = env_ids.detach().cpu().tolist()
        for env_id in env_ids:
            for hand_prim, box_prim, box_local_pos, local_pos0_attr, local_pos1_attr, local_rot0_attr, local_rot1_attr in self._d6_anchor_refs[env_id]:
                hand_local_pos, hand_local_rot, box_local_rot = self._matching_hand_local_pos(hand_prim, box_prim, box_local_pos)
                local_pos0_attr.Set(hand_local_pos)
                local_pos1_attr.Set(Gf.Vec3f(*box_local_pos))
                local_rot0_attr.Set(hand_local_rot)
                local_rot1_attr.Set(box_local_rot)

    @staticmethod
    def _box_side_anchor_local_pos(robot_index: int, num_robots: int, hand_side: str) -> tuple[float, float, float]:
        lateral_offset = 0.20
        height_offset = 0.0
        lateral_sign = 1.0 if hand_side == "left" else -1.0
        return (lateral_sign * lateral_offset, 0.0, height_offset)

    @staticmethod
    def _matching_hand_local_pos(hand_prim, box_prim, box_local_pos: tuple[float, float, float]):
        from pxr import Gf, UsdGeom
        box_xform = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(0.0)
        hand_xform = UsdGeom.Xformable(hand_prim).ComputeLocalToWorldTransform(0.0)
        
        box_anchor_w = box_xform.Transform(Gf.Vec3d(*box_local_pos))
        J_w = Gf.Matrix4d().SetRotate(box_xform.ExtractRotation())
        J_w.SetTranslateOnly(box_anchor_w)
        
        box_quat_f = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        hand_inv = hand_xform.GetInverse()
        joint_in_hand = J_w * hand_inv
        
        hand_pos_d = joint_in_hand.ExtractTranslation()
        hand_rot_d = joint_in_hand.ExtractRotation().GetQuat()
        
        hand_pos_f = Gf.Vec3f(float(hand_pos_d[0]), float(hand_pos_d[1]), float(hand_pos_d[2]))
        hand_quat_f = Gf.Quatf(float(hand_rot_d.GetReal()), float(hand_rot_d.GetImaginary()[0]), float(hand_rot_d.GetImaginary()[1]), float(hand_rot_d.GetImaginary()[2]))
        
        return hand_pos_f, hand_quat_f, box_quat_f

    @staticmethod
    def _create_box_anchor_marker(stage, box_path, joint_name, box_local_pos, hand_side):
        from pxr import UsdGeom, Gf
        marker_path = f"{box_path}/debug_anchor_{joint_name}"
        marker = UsdGeom.Sphere.Define(stage, marker_path)
        marker.CreateRadiusAttr(0.025)
        marker.AddTranslateOp().Set(Gf.Vec3f(*box_local_pos))
        color = Gf.Vec3f(1.0, 0.1, 0.1) if hand_side == "left" else Gf.Vec3f(0.1, 0.3, 1.0)
        marker.GetDisplayColorAttr().Set([color])
