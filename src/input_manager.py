import time
import numpy as np
from multiprocessing import shared_memory, Event, Process
import pyspacemouse

# Define a structure for your command state
# [dx, dy, dz, rx, ry, rz, button1, button2, ..., button15]
# Extended to support 15 buttons for SpaceMouse Pro/Enterprise
COMMAND_SHAPE = (21,)  # 6 axes + 15 buttons
COMMAND_DTYPE = np.float32


def spacemouse_process(shm_name: str, stop_event):
    """
    Background process to read SpaceMouse input.
    Continuously reads the SpaceMouse state and writes it to shared memory.
    This is completely non-blocking for the main process.
    """
    existing_shm = None
    try:
        # Attach to the existing shared memory
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shared_command = np.ndarray(
            COMMAND_SHAPE, dtype=COMMAND_DTYPE, buffer=existing_shm.buf)

        print("SpaceMouse process started.")

        # Initialize SpaceMouse
        success = pyspacemouse.open()
        if not success:
            print("❌ Failed to open SpaceMouse")
            return

        while not stop_event.is_set():
            try:
                state = pyspacemouse.read()
                if state:
                    # Check if state is effectively zero (within dead zone)
                    is_zero_state = (
                        abs(state.x or 0) < 0.01 and abs(state.y or 0) < 0.01 and abs(state.z or 0) < 0.01 and
                        abs(state.roll or 0) < 0.01 and abs(state.pitch or 0) < 0.01 and abs(state.yaw or 0) < 0.01
                    )
                    
                    # If at zero, try to clear any additional buffered readings to prevent lag
                    if is_zero_state:
                        for _ in range(3):  # Clear up to 3 additional buffered readings
                            try:
                                extra_state = pyspacemouse.read()
                                if extra_state:
                                    # If we find a non-zero reading in the buffer, use it instead
                                    if not (abs(extra_state.x or 0) < 0.01 and abs(extra_state.y or 0) < 0.01 and 
                                           abs(extra_state.z or 0) < 0.01 and abs(extra_state.roll or 0) < 0.01 and 
                                           abs(extra_state.pitch or 0) < 0.01 and abs(extra_state.yaw or 0) < 0.01):
                                        state = extra_state  # Use the non-zero state
                                        break
                            except Exception:
                                break  # No more buffered readings
                    
                    # Write the latest state to shared memory (non-blocking)
                    shared_command[0] = state.x or 0.0
                    shared_command[1] = state.y or 0.0
                    shared_command[2] = state.z or 0.0
                    shared_command[3] = state.roll or 0.0
                    shared_command[4] = state.pitch or 0.0
                    shared_command[5] = state.yaw or 0.0

                    # Handle buttons (up to 15 buttons)
                    if hasattr(state, 'buttons') and state.buttons:
                        for i in range(min(15, len(state.buttons))):
                            shared_command[6 + i] = float(state.buttons[i])
                        # Zero out unused button slots
                        for i in range(len(state.buttons), 15):
                            shared_command[6 + i] = 0.0
                    else:
                        # No buttons, zero them all
                        for i in range(15):
                            shared_command[6 + i] = 0.0

                time.sleep(0.001)  # Small sleep to be CPU efficient
            except Exception as e:
                print(f"SpaceMouse read error: {e}")
                time.sleep(0.01)

    except Exception as e:
        print(f"SpaceMouse Error: {e}")
    finally:
        try:
            pyspacemouse.close()
        except Exception:
            pass
        try:
            if existing_shm is not None:
                existing_shm.close()
        except Exception:
            pass


class SpaceMouseManager:
    """
    Manages reading from a SpaceMouse using shared memory for non-blocking access.
    This completely eliminates queue-based blocking in the main control loop.
    """

    def __init__(self):
        # Create shared memory for command state
        command_size = int(np.prod(COMMAND_SHAPE) *
                           np.dtype(COMMAND_DTYPE).itemsize)
        self.input_shm = shared_memory.SharedMemory(
            create=True, size=command_size)
        self.latest_command_view = np.ndarray(
            COMMAND_SHAPE, dtype=COMMAND_DTYPE, buffer=self.input_shm.buf
        )

        # Initialize with zeros
        self.latest_command_view.fill(0.0)

        self.stop_event = Event()
        self.process = None

    def start(self):
        """Starts the background process for reading mouse input."""
        if self.process is None:
            self.process = Process(
                target=spacemouse_process,
                args=(self.input_shm.name, self.stop_event),
                daemon=True
            )
            self.process.start()

    def get_state(self):
        """
        Gets the latest command state from shared memory. COMPLETELY NON-BLOCKING.
        Returns a simple object with the parsed state.
        """
        try:
            # Copy the current state (very fast operation)
            command = self.latest_command_view.copy()

            # Create a simple state object similar to the original
            class SpaceMouseState:
                def __init__(self, cmd):
                    self.x = cmd[0]
                    self.y = cmd[1]
                    self.z = cmd[2]
                    self.roll = cmd[3]
                    self.pitch = cmd[4]
                    self.yaw = cmd[5]
                    # Extract buttons
                    self.buttons = [bool(cmd[6 + i]) for i in range(15)]

            return SpaceMouseState(command)
        except (IndexError, ValueError, TypeError):
            return None

    def is_alive(self):
        """Check if the SpaceMouse process is still running."""
        return self.process is not None and self.process.is_alive()

    def stop(self):
        """Terminates the background process and cleans up shared memory."""
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()

        try:
            self.input_shm.close()
            self.input_shm.unlink()
        except Exception:
            pass
