import time
import config
import numpy as np
from multiprocessing import Process, shared_memory, Event

# Import our modular classes
from src.input_manager import SpaceMouseManager
from src.policy import TeleopPolicy
from src.incremental_recorder import IncrementalDataRecorder
from src.robot_interface import RobotInterface
from src.utils import Rate
from src.camera_reader import camera_process_realsense, dummy_camera_process


class DataCollectionController:
    """Orchestrates the entire data collection process with incremental saving."""

    def __init__(self,
                 data_log_root=None,
                 robot_ip=None,
                 translation_scale=None,
                 rotation_scale=None,
                 loop_rate_hz=None,
                 primary_camera_serial=None,
                 wrist_camera_serial=None):

        # Use config file values as defaults
        self.data_log_root = data_log_root or config.DATA_ROOT_DIR
        self.robot_ip = robot_ip or config.ROBOT_IP
        self.loop_rate_hz = loop_rate_hz or config.LOOP_RATE_HZ

        print("Initializing controller...")
        print(f"Robot IP: {self.robot_ip}")
        print(f"Data directory: {self.data_log_root}")
        print(f"Loop rate: {self.loop_rate_hz} Hz")
        print(f"Translation scale: {translation_scale or config.TRANSLATION_SCALE}")
        print(f"Rotation scale: {rotation_scale or config.ROTATION_SCALE}")
        print(f"SpaceMouse deadzone: {config.SPACEMOUSE_DEADZONE}")
        print(f"Gripper open width: {config.GRIPPER_OPEN_WIDTH}m")
        print(f"Gripper closed width: {config.GRIPPER_CLOSED_WIDTH}m")
        print(f"Gripper debounce: {config.GRIPPER_TOGGLE_DEBOUNCE}s")
        print(f"Max translation delta: {config.MAX_TRANSLATION_DELTA}m")
        print(f"Max rotation delta: {config.MAX_ROTATION_DELTA} rad")

        # Initialize components with config parameters
        self.robot = RobotInterface(robot_ip=self.robot_ip)
        self.mouse = SpaceMouseManager()
        self.policy = TeleopPolicy(
            translation_scale=translation_scale or config.TRANSLATION_SCALE,
            rotation_scale=rotation_scale or config.ROTATION_SCALE,
            spacemouse_deadzone=config.SPACEMOUSE_DEADZONE,
            max_translation_delta=config.MAX_TRANSLATION_DELTA,
            max_rotation_delta=config.MAX_ROTATION_DELTA
        )
        self.recorder = IncrementalDataRecorder(self.data_log_root)

        # Initialize precise rate limiter for stable control loop
        self.rate_limiter = Rate(self.loop_rate_hz)

        # Initialize cameras with shared memory for fast access
        print("Initializing shared memory cameras...")
        camera_success = self._initialize_shared_memory_cameras(
            primary_camera_serial, wrist_camera_serial)
        if not camera_success:
            print("⚠️ Cameras failed to initialize, continuing without cameras")

    def _initialize_shared_memory_cameras(self, primary_serial=None, wrist_serial=None):
        """Initialize shared memory cameras for fast, non-blocking access."""
        try:
            # Define camera properties
            self.cam_shape = (480, 640, 3)  # Height, Width, Channels
            self.cam_dtype = np.uint8
            frame_size = int(np.prod(self.cam_shape) *
                             np.dtype(self.cam_dtype).itemsize)            # Create shared memory blocks for both cameras
            self.shm_primary = shared_memory.SharedMemory(
                create=True, size=frame_size)
            self.shm_wrist = shared_memory.SharedMemory(
                create=True, size=frame_size)

            # Create NumPy array views for easy access
            self.latest_frame_primary = np.ndarray(
                self.cam_shape, dtype=self.cam_dtype, buffer=self.shm_primary.buf)
            self.latest_frame_wrist = np.ndarray(
                self.cam_shape, dtype=self.cam_dtype, buffer=self.shm_wrist.buf)

            # Initialize with black frames
            self.latest_frame_primary.fill(0)
            self.latest_frame_wrist.fill(0)

            # Start camera process
            self.camera_stop_event = Event()

            # Try RealSense cameras first, fall back to dummy if not available
            try:
                self.camera_process = Process(
                    target=camera_process_realsense,
                    args=(self.shm_primary.name, self.shm_wrist.name,
                          self.cam_shape, self.cam_dtype, self.camera_stop_event,
                          primary_serial, wrist_serial)
                )
                print("📸 Starting RealSense camera processes...")
            except Exception:
                self.camera_process = Process(
                    target=dummy_camera_process,
                    args=(self.shm_primary.name, self.shm_wrist.name,
                          self.cam_shape, self.cam_dtype, self.camera_stop_event)
                )
                print("📸 Starting dummy camera processes...")

            self.camera_process.start()

            # Give cameras time to initialize
            time.sleep(2.0)

            return True

        except Exception as e:
            print(f"❌ Shared memory camera initialization failed: {e}")
            return False

    def get_camera_images(self):
        """Get the latest camera images from shared memory (non-blocking)."""
        try:
            # Copy to avoid race conditions (this is very fast)
            img_primary = self.latest_frame_primary.copy()
            img_wrist = self.latest_frame_wrist.copy()
            return img_primary, img_wrist
        except Exception as e:
            print(f"❌ Error getting camera images: {e}")
            # Return dummy images if something goes wrong
            dummy_img = np.zeros(self.cam_shape, dtype=self.cam_dtype)
            return dummy_img, dummy_img

    def collect_episode(self, language_instruction: str, episode_num: int, total_episodes: int):
        """Collect a single episode of demonstration data."""
        print(f"\n{'='*60}")
        print(f"🎬 EPISODE {episode_num}/{total_episodes}")
        print(f"📝 Instruction: '{language_instruction}'")
        print(f"{'='*60}")

        # Reset robot and clear control signals for new episode
        print("🔄 Preparing for new episode...")
        self._reset_for_new_episode()

        # Start new episode
        episode_dir = self.recorder.start_new_episode(language_instruction)

        print("Controls:")
        print("  • Move SpaceMouse: Control robot position")
        print("  • Left button (first): Close gripper")
        print("  • Right button (last): Open gripper")
        print("  • Hold BOTH buttons for 3 seconds: End episode and continue to next")
        print("="*60 + "\n")

        step_count = 0
        both_buttons_start_time = None  # Track when both buttons were pressed
        episode_ending = False  # Track if episode is ending to skip recording

        # Reset the rate limiter for this episode
        self.rate_limiter.reset()

        try:
            while True:

                # 1. Get current robot state (Observation) from shared memory at precise 10Hz
                # The robot updates state at 1000Hz, we sample it at 10Hz for data collection
                current_obs = self.robot.get_obs()
                current_ee_pose = current_obs['panda_hand_pose']

                # 2. Get user input
                mouse_state = self.mouse.get_state()
                if mouse_state is None:
                    time.sleep(0.1)  # Small sleep if no mouse data
                    continue

                # Parse button states for different SpaceMouse models
                button_left = False
                button_right = False

                if hasattr(mouse_state, 'buttons') and mouse_state.buttons:
                    if len(mouse_state.buttons) >= 15:
                        # SpaceMouse Pro/Enterprise with 15 buttons
                        # First button (left)
                        button_left = mouse_state.buttons[0]
                        # Last button (right)
                        button_right = mouse_state.buttons[14]
                    elif len(mouse_state.buttons) >= 2:
                        # Standard SpaceMouse with 2 buttons
                        button_left = mouse_state.buttons[0]
                        button_right = mouse_state.buttons[1]
                    elif len(mouse_state.buttons) == 1:
                        # Some SpaceMouse models only report one button
                        button_left = mouse_state.buttons[0]
                        # Remove print to avoid slowing down control loop

                # Check for both buttons held for episode end
                if button_left and button_right:  # Both buttons pressed
                    if both_buttons_start_time is None:
                        both_buttons_start_time = time.time()
                        episode_ending = True  # Start skipping data recording
                        print("⏳ Hold both buttons for 3 seconds to end episode...")
                        print(
                            "🗑️ Skipping data recording during episode transition...")
                    elif time.time() - both_buttons_start_time >= 3.0:
                        recorded_timesteps = step_count  # All steps are recorded at 10Hz
                        print(
                            f"\n🛑 Episode {episode_num} ended by user. Recorded {recorded_timesteps} timesteps at 10Hz.")
                        break
                else:
                    both_buttons_start_time = None  # Reset if buttons released
                    episode_ending = False  # Resume recording if buttons released

                # 3. Get the desired action from the policy
                action = self.policy.get_action(mouse_state, current_ee_pose)

                # 4. Execute the action on the robot (real-time)
                self.robot.step(action)

                # 5. Send direct button states for gripper control (only if not ending episode)
                if not (button_left and button_right):
                    self.robot.set_gripper_button_state(
                        button_left, button_right)

                # 6. Record the timestep at precise 10Hz data collection rate
                # Record every iteration since we're now running at 10Hz data collection frequency
                # Robot control runs independently at 1000Hz while data is collected at 10Hz
                if not episode_ending:
                    try:
                        # Get camera images from shared memory (fast, non-blocking)
                        img_primary, img_wrist = self.get_camera_images()

                        # Use threading to make recording non-blocking
                        import threading
                        recording_thread = threading.Thread(
                            target=self.recorder.record_timestep_with_images,
                            args=(current_obs, action,
                                  language_instruction, img_primary, img_wrist),
                            daemon=True
                        )
                        recording_thread.start()
                    except Exception as e:
                        print(f"⚠️ Recording error (continuing): {e}")

                step_count += 1  # Always increment step count

                # Monitor control loop performance less frequently to reduce overhead
                # Check every 20 steps (2 seconds at 10Hz) to reduce console spam
                if step_count % 20 == 0 and step_count > 0:
                    actual_hz = self.rate_limiter.get_actual_rate()
                    recorded_timesteps = step_count  # All steps are now recorded at 10Hz

                    # Check if rate is within bounds (9.5-10.5Hz for data collection)
                    if not self.rate_limiter.is_rate_stable():
                        if actual_hz > 10.5:
                            print(
                                f"⚠️ Data collection rate too high: {actual_hz:.2f}Hz > 10.5Hz")
                        elif actual_hz < 9.5:
                            print(
                                f"⚠️ Data collection rate too low: {actual_hz:.2f}Hz < 9.5Hz")

                    # Show control command info for debugging
                    cmd_age = time.time() - getattr(self.robot, '_command_timestamp', 0)
                    print(
                        f"📊 Episode {episode_num}: {recorded_timesteps} recorded timesteps | Data collection: {actual_hz:.2f}Hz | Robot control: 1000Hz | Command age: {cmd_age*1000:.1f}ms")

                # Maintain precise 10Hz data collection loop using Rate class
                # Robot control runs independently at 1000Hz
                self.rate_limiter.sleep()

        except KeyboardInterrupt:
            recorded_timesteps = step_count  # All steps are recorded at 10Hz
            print(
                f"\n🛑 Episode {episode_num} interrupted by Ctrl+C. Recorded {recorded_timesteps} timesteps at 10Hz.")

        print(f"📁 Saved to: {episode_dir}")
        # Clear any residual control signals after episode
        self._clear_control_signals()
        recorded_timesteps = step_count  # Return actual recorded timesteps at 10Hz
        return recorded_timesteps

    def _reset_for_new_episode(self):
        """Reset robot state and clear control signals for new episode."""
        # Clear any existing movement commands
        self._clear_control_signals()

        # Reset recorder state (clears action history)
        self.recorder.reset_episode_state()

        # For episode 1, no need to reset (already at home)
        # For subsequent episodes, robot position reset is handled in the inter-episode wait
        print("🏠 Preparing for new episode...")

        # Restart SpaceMouse process if it died during Ctrl+C
        if not self.mouse.is_alive():
            print("🔄 Restarting SpaceMouse process...")
            self.mouse.stop()
            time.sleep(0.5)
            self.mouse.start()

        # Clear all movement deltas (real-time safe)
        self.robot.set_movement_delta([0, 0, 0], [0, 0, 0])

        # Open gripper
        self.robot.open_gripper()

        # Brief pause to let robot settle
        time.sleep(0.5)  # Much shorter wait

        print("✅ Episode reset complete")

    def _clear_control_signals(self):
        """Clear all control signals and movement commands."""
        # Clear spacemouse input buffer
        if self.mouse:
            # Read and discard any pending spacemouse data
            for _ in range(10):
                self.mouse.get_state()

        # Clear robot movement deltas
        if self.robot:
            self.robot.set_movement_delta([0, 0, 0], [0, 0, 0])

        print("🧹 Control signals cleared")

    def run(self):
        """Main data collection loop with multiple episodes."""
        print("Starting SpaceMouse input process...")
        self.mouse.start()

        # Reset robot to a neutral position and open gripper
        print("Resetting robot to initial pose...")
        # This is OK during initial setup before real-time control starts
        self.robot.reset_joints()
        self.robot.open_gripper()

        # Start real-time control loop
        print("Starting real-time control loop...")
        self.robot.start_realtime_control()

        try:
            # Get language instruction and number of episodes
            print("\n" + "="*60)
            print("🤖 MULTI-EPISODE DATA COLLECTION SETUP")
            print("="*60)

            language_instruction = input(
                "📝 Enter language instruction for demonstrations: ")
            while not language_instruction.strip():
                language_instruction = input(
                    "📝 Please enter a valid instruction: ")

            num_episodes = input("🔢 Enter number of episodes to collect: ")
            while True:
                try:
                    num_episodes = int(num_episodes)
                    if num_episodes > 0:
                        break
                    else:
                        num_episodes = input(
                            "📝 Please enter a positive number: ")
                except ValueError:
                    num_episodes = input("📝 Please enter a valid number: ")

            print(
                f"\n🎯 Will collect {num_episodes} episodes with instruction: '{language_instruction}'")

            total_timesteps = 0

            for episode_num in range(1, num_episodes + 1):
                # Wait between episodes (except for the first one)
                if episode_num > 1:
                    print(
                        f"\n⏳ Waiting 10 seconds before episode {episode_num}...")
                    print("🏠 Moving robot to initial position...")

                    # Actually move robot to initial position using safe reset
                    try:
                        self.robot.safe_episode_reset()
                        print("✅ Robot moved to initial position")
                    except Exception as e:
                        print(f"⚠️ Could not reset robot position: {e}")
                        # Fallback to simple movement clearing
                        self.robot.set_movement_delta([0, 0, 0], [0, 0, 0])
                        self.robot.open_gripper()

                    for i in range(10, 0, -1):
                        print(f"   Starting in {i} seconds...", end="\r")
                        time.sleep(1)
                    print("   Starting now!                ")

                # Collect the episode
                timesteps = self.collect_episode(
                    language_instruction, episode_num, num_episodes)
                total_timesteps += timesteps

                # Short pause before next episode
                if episode_num < num_episodes:
                    time.sleep(2)

            print("\n🎉 ALL EPISODES COMPLETED!")
            print(f"📊 Total timesteps collected: {total_timesteps}")
            print(
                f"📁 Data saved to experiment directory: {self.recorder.experiment_dir}")

        except KeyboardInterrupt:
            print("\n🛑 Data collection interrupted.")
        except Exception as e:
            print(f"\n❌ Error during data collection: {e}")
        finally:
            print("🔄 Shutting down...")
            self.mouse.stop()
            self.recorder.cleanup()
            self._cleanup_cameras()
            self.robot.end()
            print("✅ Shutdown complete.")

    def _cleanup_cameras(self):
        """Clean up shared memory cameras."""
        try:
            if hasattr(self, 'camera_stop_event'):
                self.camera_stop_event.set()
            if hasattr(self, 'camera_process'):
                self.camera_process.join(timeout=5.0)
            if hasattr(self, 'shm_primary'):
                self.shm_primary.close()
                self.shm_primary.unlink()
            if hasattr(self, 'shm_wrist'):
                self.shm_wrist.close()
                self.shm_wrist.unlink()
            print("📸 Camera resources cleaned up")
        except Exception as e:
            print(f"⚠️ Camera cleanup error: {e}")


def main():
    """Main entry point with command line argument support."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Collect robot demonstration data with incremental saving')
    parser.add_argument('--robot-ip', type=str, help='Robot IP address')
    parser.add_argument('--data-dir', type=str, help='Data storage directory')
    parser.add_argument('--rate', type=int, help='Control loop rate (Hz)')
    parser.add_argument('--translation-scale', type=float,
                        help='Translation sensitivity')
    parser.add_argument('--rotation-scale', type=float,
                        help='Rotation sensitivity')
    parser.add_argument('--primary-camera', type=str,
                        help='Primary camera serial number')
    parser.add_argument('--wrist-camera', type=str,
                        help='Wrist camera serial number')

    args = parser.parse_args()

    controller = DataCollectionController(
        robot_ip=args.robot_ip,
        data_log_root=args.data_dir,
        loop_rate_hz=args.rate,
        translation_scale=args.translation_scale,
        rotation_scale=args.rotation_scale,
        primary_camera_serial=args.primary_camera,
        wrist_camera_serial=args.wrist_camera
    )
    controller.run()


if __name__ == "__main__":
    main()
