import numpy as np
import config
import threading
import time
from typing import Dict, Any
from multiprocessing.managers import SharedMemoryManager
from franka.utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from scipy.spatial.transform import Rotation as R
import franky
from franky import Robot, JointMotion, ReferenceType


class FrankaRobotWrapper:
    """Wrapper for franky Robot to provide cartesian velocity control."""
    
    def __init__(self, robot_ip):
        self.robot = Robot(robot_ip)
        self.robot.recover_from_errors()
        self.gripper = franky.Gripper(robot_ip)
        
        # Set lower dynamics to avoid velocity discontinuities (match working example)
        try:
            self.robot.relative_dynamics_factor = 0.1  # Match the working example
            print("[FrankaRobotWrapper] Set relative dynamics factor to 0.1")
        except Exception as e:
            print(f"[FrankaRobotWrapper] Could not set dynamics: {e}")
        
    def cartesian_velocity_control(self, velocity_cmd):
        """Cartesian velocity control compatible with reference code."""
        try:
            from franky import CartesianVelocityMotion, Twist
            
            # Extract velocity components
            x = velocity_cmd.get("x", 0.0)
            y = velocity_cmd.get("y", 0.0) 
            z = velocity_cmd.get("z", 0.0)
            R = velocity_cmd.get("R", 0.0)
            P = velocity_cmd.get("P", 0.0)
            Y = velocity_cmd.get("Y", 0.0)
            # No duration needed - franky handles this automatically
            
            # Clamp velocities to robot limits (conservative for franky)
            # Higher velocity limits for more responsive movement
            max_linear_vel = 0.15   # 15cm/s - faster but safe speed
            max_angular_vel = 0.3   # 0.3 rad/s - faster angular speed
            
            x = max(-max_linear_vel, min(max_linear_vel, x))
            y = max(-max_linear_vel, min(max_linear_vel, y))
            z = max(-max_linear_vel, min(max_linear_vel, z))
            R = max(-max_angular_vel, min(max_angular_vel, R))
            P = max(-max_angular_vel, min(max_angular_vel, P))
            Y = max(-max_angular_vel, min(max_angular_vel, Y))
            

            # Use franky API for continuous velocity control
            # Create motion with both linear and angular velocity
            linear_velocity = np.array([x, y, z])
            angular_velocity = np.array([R, P, Y]) 
            motion = CartesianVelocityMotion(Twist(linear_velocity, angular_velocity))
            
            # Execute motion (always async for velocity control)
            self.robot.move(motion, asynchronous=True)
            
        except Exception as e:
            # Handle reflex mode automatically
            if "Reflex" in str(e):
                try:
                    print("[FrankaRobotWrapper] Reflex mode detected, recovering...")
                    self.robot.recover_from_errors()
                    time.sleep(0.05)  # Brief pause after recovery (reduced from 0.1)
                except Exception:
                    pass  # Continue even if recovery fails
            # Don't print every error to avoid spamming the console during real-time control
            elif "motion_generator_velocity_discontinuity" not in str(e) and "exceeds maximum velocity" not in str(e):
                if hasattr(self, '_last_error_time'):
                    if time.time() - self._last_error_time > 1.0:  # Only print once per second
                        print(f"[FrankaRobotWrapper] Velocity control error: {e}")
                        self._last_error_time = time.time()
                else:
                    print(f"[FrankaRobotWrapper] Velocity control error: {e}")
                    self._last_error_time = time.time()
    
    @property
    def state(self):
        """Get robot state."""
        return self.robot.state
    
    def move(self, motion, asynchronous=False):
        """Move robot with given motion."""
        return self.robot.move(motion, asynchronous=asynchronous)
    
    def stop(self):
        """Stop any current robot motion."""
        return self.robot.stop()
    
    def join_motion(self, timeout=None):
        """Wait for current motion to finish."""
        return self.robot.join_motion(timeout or 10.0)


class RealTimeRobotInterface:
    """
    Real-time robot interface using franky with proper control loop and shared memory.
    Based on the reference implementation in robot.py but adapted for the modular system.
    """

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
        # Store gripper future objects to prevent them from being destroyed (fixes blocking issue)
        self._gripper_futures = []

        # Real-time control variables
        self.current_translation = None
        self.current_rotation = None
        self.controller = None
        self.ctx = None

        # Initialize robot connection
        self._initialize_robot()

    def _initialize_robot(self):
        """Initialize the robot connection using franky."""
        try:
            self.robot = FrankaRobotWrapper(self.robot_ip)
            print(f"✅ Connected to robot at {self.robot_ip}")

            # Move to home position (initial setup) - matches reference robot.py
            home_pose = [
                -0.01588696,
                -0.25534376,
                0.18628714,
                -2.28398158,
                0.0769999,
                2.02505396,
                0.07858208,
            ]

            print("✅ Moving to home position...")
            motion = JointMotion(home_pose, reference_type=ReferenceType.Absolute)
            
            # Stop any existing motion first
            try:
                self.robot.stop()
                time.sleep(0.5)  # Give time for robot to stop
            except Exception:
                pass  # Robot might not be moving
            
            # Use asynchronous motion and wait for completion
            self.robot.move(motion, asynchronous=True)
            
            # Wait for motion to complete with timeout
            try:
                self.robot.join_motion(timeout=10.0)
                print("✅ Robot moved to home position")
            except Exception as e:
                print(f"Home position motion may not have completed: {e}")

            # Initialize gripper
            try:
                # Use async operation during initialization but wait for completion
                gripper_future = self.robot.gripper.move_async(width=config.GRIPPER_OPEN_WIDTH, speed=0.1)
                # Store future to prevent destruction
                self._gripper_futures.append(gripper_future)
                print("✅ Gripper initialized")
            except Exception as e:
                print(f"Gripper initialization issue: {e}")

        except ImportError as e:
            raise ImportError(
                f"franky library not found. Please install it with: pip install franky\nError: {e}"
            )
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
        """Real-time control loop running in separate thread using franky at 1kHz frequency."""
        try:
            # Signal that we're ready
            self._ready_event.set()
            print("🔄 Real-time control loop started at 1000Hz using franky")

            # Start gripper control thread
            self._gripper_stop_event.clear()
            self._gripper_thread = threading.Thread(
                target=self._gripper_control_loop, daemon=True)
            self._gripper_thread.start()

            # Ultra-high frequency control loop for maximum responsiveness
            control_period = 0.001  # 1ms = 1000Hz for maximum responsiveness
            iteration_count = 0
            next_iteration_time = time.time()  # Track absolute timing for precise frequency

            while not self._stop_event.is_set():
                try:                    
                    # Get movement deltas from main thread with command aging
                    with self._command_lock:
                        current_time = time.time()
                        # Check if command is too old (stale)
                        if current_time - self._command_timestamp > self._command_timeout:
                            # Command is stale, use zero movement
                            dpos = np.zeros(3, dtype=np.float32)
                            drot = np.zeros(3, dtype=np.float32)
                        else:
                            # Scale deltas to proper velocities for responsive movement
                            # Since franky expects velocities (m/s), we need to scale the small deltas up
                            dpos = self._delta_translation * 10.0   # Scale up for velocity control (10x faster)
                            drot = self._delta_rotation * 5.0       # Scale up for angular velocity control

                    # Always send velocity command - either movement or explicit stop
                    velocity_cmd = {
                        "x": float(dpos[0]),
                        "y": float(dpos[1]), 
                        "z": float(dpos[2]),
                        "R": float(drot[0]),
                        "P": float(drot[1]),
                        "Y": float(drot[2]),
                        "is_async": True,
                    }
                    self.robot.cartesian_velocity_control(velocity_cmd)

                    # Update state buffer every iteration for 1000Hz state updates
                    # This gives us maximum responsiveness for data collection
                    self._update_robot_state()

                    iteration_count += 1
                    
                    # Precise timing control for 1000Hz using absolute timing
                    next_iteration_time += control_period
                    current_time = time.time()
                    sleep_time = next_iteration_time - current_time
                    
                    if sleep_time > 0:
                        # For very short sleep times, use busy waiting for better precision
                        if sleep_time < 0.0005:  # Less than 0.5ms, use busy wait
                            while time.time() < next_iteration_time:
                                pass  # Busy wait for precise timing
                        else:
                            time.sleep(sleep_time)
                    else:
                        # If we're running behind, reset the timing to prevent drift
                        next_iteration_time = time.time()

                except Exception as e:
                    print(f"❌ Error in control loop iteration: {e}")
                    time.sleep(0.01)

            # Stop gripper thread
            self._gripper_stop_event.set()
            if self._gripper_thread:
                self._gripper_thread.join(timeout=1.0)

        except Exception as e:
            print(f"❌ Error in real-time control loop: {e}")
        finally:
            print("🔄 Real-time control loop stopped")

    def _gripper_control_loop(self):
        """
        Gripper control loop - execute commands only once per button press with async operations.
        
        Uses async gripper operations and retains future objects to prevent blocking of other threads.
        This fixes the issue where synchronous gripper calls would block the robot control thread.
        Based on solution from franky GitHub issue: TimSchneider42/franky#158
        """
        while not self._gripper_stop_event.is_set():
            try:
                command = None
                with self._gripper_lock:
                    command = self._gripper_command
                    # Clear command immediately to prevent re-execution
                    if command:
                        self._gripper_command = None

                if command == "open":
                    # Use async operation and retain future to prevent blocking
                    future = self.robot.gripper.move_async(width=config.GRIPPER_OPEN_WIDTH, speed=0.05)
                    self._gripper_futures.append(future)
                    # Keep only recent futures to prevent memory buildup
                    if len(self._gripper_futures) > 10:
                        self._gripper_futures.pop(0)
                elif command == "close":
                    # Use async operation and retain future to prevent blocking
                    future = self.robot.gripper.grasp_async(
                        width=config.GRIPPER_CLOSED_WIDTH, speed=0.05, force=20
                    )
                    self._gripper_futures.append(future)
                    # Keep only recent futures to prevent memory buildup
                    if len(self._gripper_futures) > 10:
                        self._gripper_futures.pop(0)

            except Exception:
                pass  # Silently continue on gripper errors to maintain control loop speed

            time.sleep(0.01)  # 100Hz check rate

    def _update_robot_state(self):
        """Update the shared robot state buffer using franky robot state."""
        try:
            gripper_width = self.robot.gripper.width
            robot_state = self.robot.state
            ee_pose = robot_state.O_T_EE
            ee_pos = ee_pose.translation
            ee_ori = ee_pose.quaternion
            
            state_data = {
                "q": robot_state.q,
                "dq": robot_state.dq,
                "tau_J": robot_state.tau_J,
                "EE_position": ee_pos,
                "EE_orientation": ee_ori,
                "gripper_state": gripper_width
            }

            self.robot_state_buffer.put(state_data)
        except Exception:
            pass  # Don't spam errors in real-time loop

    def set_movement_delta(self, translation_delta, rotation_delta):
        """Set movement deltas for real-time control with timestamp for command aging."""
        with self._command_lock:
            self._delta_translation = np.array(
                translation_delta, dtype=np.float32)
            self._delta_rotation = np.array(rotation_delta, dtype=np.float32)
            self._command_timestamp = time.time()  # Mark when command was set

    def set_gripper_button_state(self, button_0_pressed, button_1_pressed):
        """Handle gripper control via button PRESS events with debouncing."""
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
        with self._gripper_lock:
            self._gripper_command = "open"

    def close_gripper(self):
        """Close the robot gripper using config value."""
        with self._gripper_lock:
            self._gripper_command = "close"

    def reset_joints(self):
        """Reset robot to home position."""
        try:
            home_pose = [
                -0.01588696,
                -0.25534376,
                0.18628714,
                -2.28398158,
                0.0769999,
                2.02505396,
                0.07858208,
            ]

            print("🏠 Moving to home position...")
            motion = JointMotion(home_pose, reference_type=ReferenceType.Absolute)
            
            # Stop any existing motion first
            try:
                self.robot.stop()
                time.sleep(0.5)  # Give time for robot to stop
            except Exception:
                pass  # Robot might not be moving
            
            # Use synchronous motion for reset
            self.robot.move(motion, asynchronous=False)
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

            # Now safely reset to home position using franky
            home_pose = [
                -0.01588696,
                -0.25534376,
                0.18628714,
                -2.28398158,
                0.0769999,
                2.02505396,
                0.07858208,
            ]
            
            motion = JointMotion(home_pose, reference_type=ReferenceType.Absolute)
            self.robot.move(motion, asynchronous=False)

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


# Backward compatibility alias
RobotInterface = RealTimeRobotInterface
