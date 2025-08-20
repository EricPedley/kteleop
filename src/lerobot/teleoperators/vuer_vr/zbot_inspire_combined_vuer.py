#!/usr/bin/env python

import json
import logging
import math
import socket
import time
from typing import Any
import numpy as np
import cv2
import yaml
from pathlib import Path


from vuer import Vuer, VuerSession
import asyncio
from vuer.schemas import ImageBackground
from ..teleoperator import Teleoperator
from .config_zbot_inspire_combined_vuer import VuerVRConfig
from dex_retargeting.retargeting_config import RetargetingConfig
from .ik import KBot_ArmIK

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret

hand2inspire = np.array([[0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 1]])


grd_yup2grd_zup = np.array([[0, 0, -1, 0],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])


logger = logging.getLogger(__name__)

cam_mat = np.array([[266.61728276,0.,643.83126137],[0.,266.94450686,494.81811813],[0.,0.,1.,]])
dist_coeffs = np.array([[-6.07417419e-02,9.95447444e-02,-2.26448001e-04,1.22881804e-03,3.42134205e-03,1.45361886e-01,8.03248099e-02,2.11170107e-02,-3.80620047e-03,2.48350591e-05,-8.33565666e-04,2.97806723e-05]])

async def stream_cameras(session: VuerSession, left_src=0, right_src=1):
    left_pipeline = "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36 exposure-time-mode=0 analogue-gain-mode=0 ae-enable=true awb-enable=true af-mode=manual ! video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 ! videoconvert ! appsink drop=1 max-buffers=1"
    right_pipeline = "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@88000/ov5647@36 exposure-time-mode=0 analogue-gain-mode=0 ae-enable=true awb-enable=true af-mode=manual ! video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 ! videoconvert ! appsink drop=1 max-buffers=1"
    cam_left = cv2.VideoCapture(left_pipeline, cv2.CAP_GSTREAMER)
    cam_right = cv2.VideoCapture(right_pipeline, cv2.CAP_GSTREAMER)
    
    while True:
        ret_left, frame_left = cam_left.read()
        ret_right, frame_right = cam_right.read()
        if not ret_left or not ret_right:
            continue
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left_rgb = cv2.undistort(frame_left_rgb, cam_mat, dist_coeffs)
        frame_right_rgb = cv2.undistort(frame_right_rgb, cam_mat, dist_coeffs)
        # Add text labels for left/right cameras
        cv2.putText(frame_left_rgb, "Left Camera", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.putText(frame_right_rgb, "Right Camera", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        # Send both images as ImageBackground objects for left/right eye
        session.upsert([
            ImageBackground(
                frame_left_rgb,
                aspect=1.778,
                height=1,
                distanceToCamera=1,
                layers=1,
                format="jpeg",
                quality=50,
                key="background-left",
                interpolate=True,
            ),
            ImageBackground(
                frame_right_rgb,
                aspect=1.778,
                height=1,
                distanceToCamera=1,
                layers=2,
                format="jpeg",
                quality=50,
                key="background-right",
                interpolate=True,
            ),
        ], to="bgChildren")
        await asyncio.sleep(1/30)  # ~30 FPS for smoother streaming


class VuerVR(Teleoperator):
    """
    Combined ZBot + Inspire hand teleoperator that receives both joint and finger data
    over a single UDP port in one packet.
    """

    config_class = VuerVRConfig
    name = "vuer_vr"

    def __init__(self, config: VuerVRConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize UDP socket
        self.cam_mat = np.array([[266.61728276,0.,643.83126137],[0.,266.94450686,494.81811813],[0.,0.,1.,]])
        self.dist_coeffs = np.array([[-6.07417419e-02,9.95447444e-02,-2.26448001e-04,1.22881804e-03,3.42134205e-03,1.45361886e-01,8.03248099e-02,2.11170107e-02,-3.80620047e-03,2.48350591e-05,-8.33565666e-04,2.97806723e-05]])

        left_pipeline = "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36 exposure-time-mode=0 analogue-gain-mode=0 ae-enable=true awb-enable=true af-mode=manual ! video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 ! videoconvert ! appsink drop=1 max-buffers=1"
        self.cam_left = cv2.VideoCapture(left_pipeline, cv2.CAP_GSTREAMER)

        
        # Initialize data storage
        self.joint_positions = {}
        self.finger_positions = {}
        self._raw_finger_values = [0] * 6
        self.last_update_time = 0
        
        # Build joint ID mappings
        self.joint_id_to_name = {}
        for i, joint_id in enumerate(config.left_arm_ids):
            if i < len(config.left_arm_names):
                self.joint_id_to_name[joint_id] = config.left_arm_names[i]
        for i, joint_id in enumerate(config.right_arm_ids):
            if i < len(config.right_arm_names):
                self.joint_id_to_name[joint_id] = config.right_arm_names[i]
        
        # Initialize joint positions to zero
        for joint_name in config.left_arm_names + config.right_arm_names:
            self.joint_positions[f"{joint_name}.pos"] = 0.0
            
        # Initialize finger positions to zero
        for finger_name in config.finger_names:
            self.finger_positions[f"{finger_name}.pos"] = 0.0
            
        assets_path = Path('/home/dpsh/kteleop/src/lerobot/teleoperators/vuer_vr/assets')
        RetargetingConfig.set_default_urdf_dir(str(assets_path))
        with (assets_path / 'inspire_hand/inspire_hand.yml').open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()
        self.arm_ik = KBot_ArmIK()
        
        # Initialize VR data storage
        self.connected = False
        self.left_hand_shared = np.eye(4, dtype=np.float32)
        self.right_hand_shared = np.eye(4, dtype=np.float32)
        self.head_matrix_shared = np.eye(4, dtype=np.float32)
        self.left_landmarks_shared = np.zeros(75, dtype=np.float32)  # 25 landmarks * 3 coordinates
        self.right_landmarks_shared = np.zeros(75, dtype=np.float32)
        self.aspect_shared = type('obj', (object,), {'value': 1.0})()
        self.app = None
        self.vuer_session = None


    def _convert_udp_to_hand_value(self, raw_value: float) -> float:
        """Convert raw finger value to hand value."""
        # For now, just return the raw value as it's already processed by retargeting
        # You can add your own conversion logic here if needed
        return float(raw_value)

    @property
    def action_features(self) -> dict[str, type]:
        features = {}
        
        # Joint features
        for joint_name in self.config.left_arm_names + self.config.right_arm_names:
            features[f"zbot_{joint_name}.pos"] = float
            
        # Finger features  
        for finger_name in self.config.finger_names:
            features[f"hand_{finger_name}.pos"] = float
            
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect to VR system."""
        self.app = Vuer()
        
        @self.app.spawn(start=True)
        async def main(session: VuerSession):
            self.vuer_session = session
            self.connected = True
            await stream_cameras(session)
            
        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        # Note: app.run() should be called from the main event loop, not here
    
    def on_hand_move(self, event, session, fps=60):
        self.left_hand_shared[:] = event.value["leftHand"]
        self.right_hand_shared[:] = event.value["rightHand"]
        self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
        self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()

    async def on_cam_move(self, event, session, fps=60):
        self.head_matrix_shared[:] = event.value["camera"]["matrix"]
        self.aspect_shared.value = event.value['camera']['aspect']

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """UDP-based teleoperator doesn't need calibration."""
        logger.info("Combined UDP teleoperator doesn't require calibration")
        pass

    def configure(self) -> None:
        """UDP-based teleoperator doesn't need configuration."""
        logger.info("Combined UDP teleoperator doesn't require configuration")
        pass

    def setup_motors(self) -> None:
        """UDP-based teleoperator doesn't have motors."""
        logger.info("Combined UDP teleoperator doesn't have motors to setup")
        pass

    def get_action(self) -> dict[str, float]:
        tip_indices = [4, 9, 14, 19, 24]

        head_mat = grd_yup2grd_zup @ self.head_matrix_shared @ fast_mat_inv(grd_yup2grd_zup)
        right_wrist_mat = grd_yup2grd_zup @ self.right_hand_shared @ fast_mat_inv(grd_yup2grd_zup)
        left_wrist_mat = grd_yup2grd_zup @ self.left_hand_shared @ fast_mat_inv(grd_yup2grd_zup)

        left_fingers = np.concatenate([self.left_landmarks_shared.reshape(-1, 3).T, np.ones((1, 25))])
        right_fingers = np.concatenate([self.right_landmarks_shared.reshape(-1, 3).T, np.ones((1, 25))])

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        left_qpos = self.left_retargeting.retarget(rel_left_fingers[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(rel_right_fingers[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

        # if latest_data is None:
        #     # No new data, return last known positions
        #     action = {}
        #     action.update({f"zbot_{k}": v for k, v in self.joint_positions.items()})
        #     action.update({f"hand_{k}": v for k, v in self.finger_positions.items()})
        #     return action

        rel_left_wrist_mat = left_wrist_mat @ hand2inspire
        rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]

        rel_right_wrist_mat = right_wrist_mat @ hand2inspire  # wTr = wTh @ hTr
        rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]
                
            
        joints = self.arm_ik.solve_ik(rel_left_wrist_mat, rel_right_wrist_mat)
        # Process joint data
        for joint_id_str, position in joints.items():
            joint_id = int(joint_id_str)
            if joint_id in self.joint_id_to_name:
                joint_name = self.joint_id_to_name[joint_id]
                joint_key = f"{joint_name}.pos"
                self.joint_positions[joint_key] = float(position)
        
        finger_values = right_qpos
        # Process finger data
        if len(finger_values) >= 6:
            self._raw_finger_values = finger_values[:6]
            
            for i, raw_value in enumerate(finger_values[:6]):
                if i < len(self.config.finger_names):
                    finger_name = self.config.finger_names[i]
                    hand_value = self._convert_udp_to_hand_value(raw_value)
                    self.finger_positions[f"{finger_name}.pos"] = hand_value
        
        self.last_update_time = time.time()
            
        # Combine all actions with prefixes
        action = {}
        action.update({f"zbot_{k}": v for k, v in self.joint_positions.items()})
        action.update({f"hand_{k}": v for k, v in self.finger_positions.items()})
        
        logger.debug(f"Received combined data: {len(joints)} joints, {len(finger_values)} fingers")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """UDP teleoperator doesn't send feedback."""

        ret_left, frame_left = self.cam_left.read()
        if not ret_left:
            return
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
        frame_left_rgb = cv2.undistort(frame_left_rgb, self.cam_mat, self.dist_coeffs)
        frame_right_rgb = frame_left_rgb

        self.vuer_session.upsert([
            ImageBackground(
                frame_left_rgb,
                aspect=1.778,
                height=1,
                distanceToCamera=1,
                layers=1,
                format="jpeg",
                quality=50,
                key="background-left",
                interpolate=True,
            ),
            ImageBackground(
                frame_right_rgb,
                aspect=1.778,
                height=1,
                distanceToCamera=1,
                layers=2,
                format="jpeg",
                quality=50,
                key="background-right",
                interpolate=True,
            ),
        ], to="bgChildren")

    def disconnect(self) -> None:
        """Disconnect from VR system."""
        if self.is_connected:
            self.connected = False
            if self.cam_left is not None:
                self.cam_left.release()
            logger.info("Disconnected from VR teleoperator") 