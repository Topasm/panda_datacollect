import numpy as np
import config
import threading
import time
from typing import Dict, Any
from multiprocessing.managers import SharedMemoryManager
from franka.utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from scipy.spatial.transform import Rotation as R
from _pylibfranka import (
    CartesianVelocities,
    ControllerMode,
    Gripper,
    JointPositions,
    RealtimeConfig,
    Robot,
)


class RealTimeRobotInterface:
    """Real-time robot interface using _libfranka with 1kHz Cartesian velocity control."""

    def __init__(self, robot_ip="172.16.0.2"):
        self.robot_ip = robot_ip

        # Shared memory for real-time communication
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # Robot state template
        robot_state = {
            "q": np.zeros(7, dtype=np.float32),
            "dq": np.zeros(7, dtype=np.float32),
            "tau_J": np.zeros(7, dtype=np.float32),
            "EE_position": np.zeros(3, dtype=np.float32),
            "EE_orientation": np.zeros(4, dtype=np.float32),
            "gripper_state": 0.0
        }

        self.robot_state_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=self.shm_manager,
            examples=robot_state,
            get_max_k=32,  # Keep large buffer for data collection
            get_time_budget=0.1,
            put_desired_frequency=1000  # Robot state updates at 1000Hz (every iteration)
        )

        # Control threading
        self._control_thread = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        # Movement deltas for real-time control (like reference code)
        self._delta_translation = np.zeros(3, dtype=np.float32)
        self._delta_rotation = np.zeros(3, dtype=np.float32)
        self._command_timestamp = 0.0  # Track when command was last updated
        self._command_timeout = 0.2    # Commands expire after 200ms (increased for slower data collection)
        self._command_lock = threading.Lock()

        # Gripper control
        self._gripper_thread = None
        self._gripper_stop_event = threading.Event()
        self._gripper_command = None
        self._gripper_lock = threading.Lock()
        # Track previous button states for edge detection
        self._last_button_state = [False, False]
        self._last_gripper_command_time = 0.0  # For debouncing gripper commands
        # Temporarily disable gripper commands to avoid collisions with arm control
        self._disable_gripper = True

        # Real-time control variables
        self.current_translation = None
        self.current_rotation = None
        self._active_control = None

        # Gripper state cache
        self._gripper = None
        self._last_gripper_width = config.GRIPPER_OPEN_WIDTH
        self._use_pose_fallback = False  # Set after init when velocity control is unavailable

        # Initialize robot connection
        self._initialize_robot()

    def _initialize_robot(self):
        """Initialize the robot connection using _libfranka."""
        try:
            self.robot = Robot(self.robot_ip, RealtimeConfig.kIgnore)
            print(f"✅ Connected to robot at {self.robot_ip} with kIgnore realtime config")

            # Carry over collision thresholds from franky setup
            lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
            upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
            lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
            upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

            try:
                self.robot.set_collision_behavior(
                    lower_torque_thresholds,
                    upper_torque_thresholds,
                    lower_force_thresholds,
                    upper_force_thresholds,
                )
                print("✅ Collision behavior set (carried from franky settings)")
            except Exception as e:
                print(f"⚠️ Could not set collision behavior: {e}")

            # Attempt recovery in case of residual errors
            try:
                self.robot.stop()
                try:
                    self.robot.automatic_error_recovery()
                except Exception:
                    pass
                self.robot.recover_from_errors()
            except Exception:
                pass

            # Clear reflex state before any motion
            try:
                self.robot.recover_from_errors()
            except Exception:
                pass

            # Detect availability of direct cartesian velocity control
            if not hasattr(self.robot, "start_cartesian_velocity_control"):
                self._use_pose_fallback = True
                print("⚠️ start_cartesian_velocity_control not available, using pose-control fallback")

            # Initialize gripper connection
            try:
                self._gripper = Gripper(self.robot_ip)
                # Homing ensures width is calibrated
                try:
                    self._gripper.homing()
                except Exception as e:
                    print(f"⚠️ Gripper homing issue: {e}")
                try:
                    self._gripper.move(config.GRIPPER_OPEN_WIDTH, 0.1)
                except Exception as e:
                    print(f"⚠️ Gripper open on init issue: {e}")
                print("✅ Gripper initialized")
            except Exception as e:
                print(f"⚠️ Gripper connection issue: {e}")

            # Move to home joint configuration if possible
            try:
                self._move_to_home_joint_position()
            except Exception as e:
                print(f"⚠️ Home move skipped: {e}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to robot at {self.robot_ip}. Error: {e}")

    def start_realtime_control(self):
        """Start the real-time control thread."""
        if self._control_thread is None or not self._control_thread.is_alive():
            self._stop_event.clear()
            self._ready_event.clear()
            self._control_thread = threading.Thread(
                target=self._realtime_control_loop, daemon=True)
            self._control_thread.start()

            # Wait for control loop to be ready
            if not self._ready_event.wait(timeout=10.0):
                raise RuntimeError(
                    "Real-time control loop failed to start within 10 seconds")
            print("✅ Real-time control started")

    def _realtime_control_loop(self):
        """Real-time control loop running in separate thread using _libfranka at ~1kHz."""
        try:
            # Try to clear reflex state before starting control
            try:
                self.robot.stop()
                try:
                    self.robot.automatic_error_recovery()
                except Exception:
                    pass
                self.robot.recover_from_errors()
            except Exception:
                pass

            # Start control session (velocity control preferred)
            try:
                if not self._use_pose_fallback:
                    self._active_control = self.robot.start_cartesian_velocity_control(
                        ControllerMode.JointImpedance
                    )
                else:
                    self._active_control = self.robot.start_cartesian_pose_control(
                        ControllerMode.JointImpedance
                    )
            except Exception as e:
                # Attempt one recovery and retry once
                try:
                    self.robot.stop()
                    try:
                        self.robot.automatic_error_recovery()
                    except Exception:
                        pass
                    self.robot.recover_from_errors()
                    if not self._use_pose_fallback:
                        self._active_control = self.robot.start_cartesian_velocity_control(
                            ControllerMode.JointImpedance
                        )
                    else:
                        self._active_control = self.robot.start_cartesian_pose_control(
                            ControllerMode.JointImpedance
                        )
                except Exception:
                    raise e

            # Signal that we're ready
            self._ready_event.set()
            mode_label = "pose-fallback" if self._use_pose_fallback else "velocity"
            print(f"🔄 Real-time control loop started at 1000Hz using _libfranka ({mode_label})")

            # Start gripper control thread
            self._gripper_stop_event.clear()
            self._gripper_thread = threading.Thread(
                target=self._gripper_control_loop, daemon=True)
            self._gripper_thread.start()

            control_period = 0.001  # 1kHz
            next_iteration_time = time.time()
            current_pose = None

            while not self._stop_event.is_set():
                try:
                    # Read latest robot state and duration from control loop
                    robot_state, duration = self._active_control.readOnce()
                    dt = getattr(duration, "to_sec", lambda: control_period)()

                    if current_pose is None:
                        current_pose = np.array(robot_state.O_T_EE).reshape(4, 4, order="F")

                    # Get movement deltas from main thread with command aging
                    with self._command_lock:
                        current_time = time.time()
                        if current_time - self._command_timestamp > self._command_timeout:
                            dpos = np.zeros(3, dtype=np.float32)
                            drot = np.zeros(3, dtype=np.float32)
                        else:
                            dpos = self._delta_translation * 10.0   # scale to m/s
                            drot = self._delta_rotation * 5.0       # scale to rad/s

                    if not self._use_pose_fallback:
                        # Send cartesian velocity command
                        vel_cmd = CartesianVelocities([
                            float(dpos[0]),
                            float(dpos[1]),
                            float(dpos[2]),
                            float(drot[0]),
                            float(drot[1]),
                            float(drot[2]),
                        ])
                        self._active_control.writeOnce(vel_cmd)
                    else:
                        # Integrate twist to pose and send as CartesianPose
                        lin_delta = np.array(dpos) * dt
                        rotvec = np.array(drot) * dt
                        rot_delta = R.from_rotvec(rotvec)
                        rot_mat = rot_delta.as_matrix()
                        new_pose = current_pose.copy()
                        new_pose[:3, 3] += lin_delta
                        new_pose[:3, :3] = rot_mat @ new_pose[:3, :3]

                        from _pylibfranka import CartesianPose  # local import to avoid top-level clutter
                        pose_cmd = CartesianPose(new_pose.flatten(order="F"))
                        self._active_control.writeOnce(pose_cmd)
                        current_pose = new_pose

                    # Update state buffer using the latest robot state
                    self._update_robot_state(robot_state)

                    # Precise timing control for 1000Hz using absolute timing
                    next_iteration_time += control_period
                    current_time = time.time()
                    sleep_time = next_iteration_time - current_time

                    if sleep_time > 0:
                        if sleep_time < 0.0005:
                            while time.time() < next_iteration_time:
                                pass
                        else:
                            time.sleep(sleep_time)
                    else:
                        next_iteration_time = time.time()

                except Exception as e:
                    print(f"❌ Error in control loop iteration: {e}")
                    time.sleep(0.01)
                    if "Reflex" in str(e):
                        try:
                            self.robot.stop()
                            try:
                                self.robot.automatic_error_recovery()
                            except Exception:
                                pass
                            self.robot.recover_from_errors()
                            # Clear pending motion to avoid immediate re-trigger
                            with self._command_lock:
                                self._delta_translation = np.zeros(3, dtype=np.float32)
                                self._delta_rotation = np.zeros(3, dtype=np.float32)
                                self._command_timestamp = time.time()
                            continue
                        except Exception:
                            pass

            # Stop gripper thread
            self._gripper_stop_event.set()
            if self._gripper_thread:
                self._gripper_thread.join(timeout=1.0)

        except Exception as e:
            print(f"❌ Error in real-time control loop: {e}")
        finally:
            try:
                if self._active_control:
                    if not self._use_pose_fallback:
                        stop_cmd = CartesianVelocities([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    else:
                        from _pylibfranka import CartesianPose
                        stop_cmd = CartesianPose(np.eye(4).flatten(order="F"))
                    stop_cmd.motion_finished = True
                    self._active_control.writeOnce(stop_cmd)
            except Exception:
                pass
            self._active_control = None
            print("🔄 Real-time control loop stopped")

    def _gripper_control_loop(self):
        """Non-blocking gripper control loop running in its own thread."""
        while not self._gripper_stop_event.is_set():
            try:
                if self._disable_gripper:
                    time.sleep(0.01)
                    continue

                command = None
                with self._gripper_lock:
                    command = self._gripper_command
                    if command:
                        self._gripper_command = None

                if command == "open" and self._gripper:
                    try:
                        self._gripper.move(config.GRIPPER_OPEN_WIDTH, 0.05)
                        self._last_gripper_width = config.GRIPPER_OPEN_WIDTH
                    except Exception:
                        pass
                elif command == "close" and self._gripper:
                    try:
                        self._gripper.grasp(
                            width=config.GRIPPER_CLOSED_WIDTH,
                            speed=0.05,
                            force=20,
                        )
                        self._last_gripper_width = config.GRIPPER_CLOSED_WIDTH
                    except Exception:
                        pass

            except Exception:
                pass

            time.sleep(0.01)  # 100Hz check rate

    def _update_robot_state(self, robot_state):
        """Update the shared robot state buffer using _libfranka robot state."""
        try:
            # Convert column-major flattened pose to 4x4 matrix
            ee_pose = np.array(robot_state.O_T_EE).reshape(4, 4, order="F")
            ee_pos = ee_pose[:3, 3]
            ee_ori = R.from_matrix(ee_pose[:3, :3]).as_quat()

            gripper_width = self._last_gripper_width
            try:
                if self._gripper:
                    gripper_state = self._gripper.read_once()
                    gripper_width = float(gripper_state.width)
                    self._last_gripper_width = gripper_width
            except Exception:
                pass

            state_data = {
                "q": robot_state.q,
                "dq": robot_state.dq,
                "tau_J": robot_state.tau_J,
                "EE_position": ee_pos,
                "EE_orientation": ee_ori,
                "gripper_state": gripper_width,
            }

            self.robot_state_buffer.put(state_data)
        except Exception:
            pass

    def set_movement_delta(self, translation_delta, rotation_delta):
        """Set movement deltas for real-time control with timestamp for command aging."""
        with self._command_lock:
            self._delta_translation = np.array(
                translation_delta, dtype=np.float32)
            self._delta_rotation = np.array(rotation_delta, dtype=np.float32)
            self._command_timestamp = time.time()  # Mark when command was set

    def set_gripper_button_state(self, button_0_pressed, button_1_pressed):
        """Handle gripper control via button PRESS events with debouncing."""
        if self._disable_gripper:
            return
        current_time = time.time()
        current_button_state = [button_0_pressed, button_1_pressed]

        # Check if enough time has passed since last gripper command (debouncing)
        if current_time - self._last_gripper_command_time < config.GRIPPER_TOGGLE_DEBOUNCE:
            # Update button state but don't send command
            self._last_button_state = current_button_state
            return

        # Only trigger on button press (rising edge), not continuous press
        # Left button just pressed
        if button_0_pressed and not self._last_button_state[0]:
            with self._gripper_lock:
                self._gripper_command = "close"
            self._last_gripper_command_time = current_time
        # Right button just pressed
        elif button_1_pressed and not self._last_button_state[1]:
            with self._gripper_lock:
                self._gripper_command = "open"
            self._last_gripper_command_time = current_time

        # Update last button state for next comparison
        self._last_button_state = current_button_state

    def get_obs(self) -> Dict[str, Any]:
        """Get current robot observation/state from shared memory buffer."""
        try:
            state_data = self.robot_state_buffer.get()
            if state_data is not None:
                return {
                    'panda_joint_positions': state_data['q'],
                    'panda_hand_pose': self._pose_from_position_orientation(
                        state_data['EE_position'],
                        state_data['EE_orientation']
                    ),
                    'panda_gripper_width': state_data['gripper_state']
                }
        except Exception:
            pass

        # Fallback if no data available
        return {
            'panda_joint_positions': np.zeros(7),
            'panda_hand_pose': np.eye(4),
            'panda_gripper_width': 0.0
        }

    def get_ee_pose(self):
        """Get current end-effector pose as 4x4 matrix."""
        obs = self.get_obs()
        return obs['panda_hand_pose']

    def step(self, action: Dict[str, Any]):
        """Execute a single action step by setting movement deltas."""
        # Use direct deltas if provided (preferred for real-time control)
        if 'delta_translation' in action and 'delta_rotation' in action:
            delta_translation = action['delta_translation']
            delta_rotation = action['delta_rotation']
            self.set_movement_delta(delta_translation, delta_rotation)
        else:
            # Fallback: calculate deltas from pose difference
            target_pose = action.get('ee_pose')
            if target_pose is not None:
                current_pose = self.get_ee_pose()
                delta_pos = target_pose[:3, 3] - current_pose[:3, 3]

                # Calculate rotation delta
                current_rot = R.from_matrix(current_pose[:3, :3])
                target_rot = R.from_matrix(target_pose[:3, :3])
                delta_rot = (target_rot * current_rot.inv()).as_euler("xyz")

                # Set movement deltas for real-time control
                self.set_movement_delta(delta_pos, delta_rot)

        # Gripper control is now handled by button states only, not policy actions
        # This prevents continuous gripper commands from the policy
        # The gripper will only respond to button press events via set_gripper_button_state()

    def open_gripper(self):
        """Open the robot gripper using config value."""
        if self._disable_gripper:
            return
        with self._gripper_lock:
            self._gripper_command = "open"

    def close_gripper(self):
        """Close the robot gripper using config value."""
        if self._disable_gripper:
            return
        with self._gripper_lock:
            self._gripper_command = "close"

    def reset_joints(self):
        """Reset robot to home position."""
        try:
            print("🏠 Moving to home position...")
            self._move_to_home_joint_position()
            print("✅ Robot moved to home position")
        except Exception as e:
            print(f"⚠️ Could not reset joints: {e}")

    def reset_to_home_pose_realtime(self):
        """Reset robot to home position within real-time control context."""
        try:
            # Clear any movement deltas first
            with self._command_lock:
                self._delta_translation = np.zeros(3, dtype=np.float32)
                self._delta_rotation = np.zeros(3, dtype=np.float32)

            print("🏠 Reset to home pose (clearing movement commands)")
        except Exception as e:
            print(f"⚠️ Could not reset to home pose: {e}")

    def safe_episode_reset(self):
        """Safely reset robot for new episode by temporarily stopping real-time control."""
        try:
            print("🔄 Stopping real-time control for safe reset...")

            # Stop the real-time control temporarily
            self._stop_event.set()
            if self._control_thread and self._control_thread.is_alive():
                self._control_thread.join(timeout=3.0)

            # Now safely reset to home position using _libfranka joint position control
            self._move_to_home_joint_position()

            # Clear movement deltas
            with self._command_lock:
                self._delta_translation = np.zeros(3, dtype=np.float32)
                self._delta_rotation = np.zeros(3, dtype=np.float32)

            # Restart real-time control
            print("🔄 Restarting real-time control...")
            self._stop_event.clear()
            self._ready_event.clear()
            self._control_thread = threading.Thread(
                target=self._realtime_control_loop, daemon=True)
            self._control_thread.start()

            # Wait for control loop to be ready
            if not self._ready_event.wait(timeout=10.0):
                raise RuntimeError("Real-time control loop failed to restart")

            print("✅ Episode reset complete with real-time control restarted")

        except Exception as e:
            print(f"⚠️ Could not perform safe episode reset: {e}")
            raise

    def stop(self):
        """Stop the real-time control loop."""
        self._stop_event.set()
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)
        print("✅ Real-time control stopped")

    def end(self):
        """Cleanup and end robot connection."""
        self.stop()
        if self.shm_manager:
            self.shm_manager.shutdown()
        print("✅ Robot interface cleaned up")

    def _pose_from_position_orientation(self, position, orientation):
        """Convert position and orientation to 4x4 pose matrix."""
        pose = np.eye(4)
        pose[:3, 3] = position

        # Convert quaternion to rotation matrix
        if len(orientation) == 4:  # quaternion
            rotation = R.from_quat(orientation)
            pose[:3, :3] = rotation.as_matrix()

        return pose


    def _move_to_home_joint_position(self):
        """Move robot to predefined home joint configuration using joint position control."""
        home_pose = [
            -0.01588696,
            -0.25534376,
            0.18628714,
            -2.28398158,
            0.0769999,
            2.02505396,
            0.07858208,
        ]

        try:
            self.robot.stop()
            try:
                self.robot.automatic_error_recovery()
            except Exception:
                pass
            self.robot.recover_from_errors()
        except Exception:
            pass

        control = self.robot.start_joint_position_control(ControllerMode.JointImpedance)
        try:
            joint_cmd = JointPositions(home_pose)
            control.writeOnce(joint_cmd)
            # Mark finish on a follow-up send to exit control
            finish_cmd = JointPositions(home_pose)
            finish_cmd.motion_finished = True
            control.writeOnce(finish_cmd)
            time.sleep(0.2)
        finally:
            try:
                finish_cmd = JointPositions(home_pose)
                finish_cmd.motion_finished = True
                control.writeOnce(finish_cmd)
            except Exception:
                pass


# Backward compatibility alias
RobotInterface = RealTimeRobotInterface
