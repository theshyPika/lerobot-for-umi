import ctypes
import os

DHPGC_OK = 0
DHPGC_ERROR_CONNECT = -1

class DHGripper:
    def __init__(self, lib_path=None):
        if lib_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, "libdhpgc.so")
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"找不到库文件: {lib_path}")
        self.lib = ctypes.CDLL(lib_path)

        self.handle = ctypes.c_void_p(None)
        
        # int dhpgc_gripper_create(const char* device, int baud, int slave_id, dhpgc_gripper_t** out_handle)
        self.lib.dhpgc_gripper_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.dhpgc_gripper_create.restype = ctypes.c_int

        # void dhpgc_gripper_destroy(dhpgc_gripper_t** gripper_handle)
        self.lib.dhpgc_gripper_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.dhpgc_gripper_destroy.restype = None

        # int dhpgc_gripper_init_calibrate(dhpgc_gripper_t* gripper_handle)
        self.lib.dhpgc_gripper_init_calibrate.argtypes = [ctypes.c_void_p]
        self.lib.dhpgc_gripper_init_calibrate.restype = ctypes.c_int

        # int dhpgc_gripper_set_position(dhpgc_gripper_t* gripper_handle, int position)
        self.lib.dhpgc_gripper_set_position.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.dhpgc_gripper_set_position.restype = ctypes.c_int

        # int dhpgc_gripper_get_gripping_status(dhpgc_gripper_t* gripper_handle, int* out_status)
        self.lib.dhpgc_gripper_get_gripping_status.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.dhpgc_gripper_get_gripping_status.restype = ctypes.c_int

        # int dhpgc_gripper_wait_for_movement_complete(dhpgc_gripper_t* gripper_handle, int timeout_ms)
        self.lib.dhpgc_gripper_wait_for_movement_complete.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.dhpgc_gripper_wait_for_movement_complete.restype = ctypes.c_int

    def connect(self, device="/dev/ttyUSB0", baud=115200, slave_id=1):
        res = self.lib.dhpgc_gripper_create(device.encode('utf-8'), baud, slave_id, ctypes.byref(self.handle))
        return res == DHPGC_OK

    def calibrate(self):
        return self.lib.dhpgc_gripper_init_calibrate(self.handle)

    def set_position(self, pos):
        """pos in [0-1000]"""
        return self.lib.dhpgc_gripper_set_position(self.handle, pos)

    def get_status(self):
        status = ctypes.c_int()
        res = self.lib.dhpgc_gripper_get_gripping_status(self.handle, ctypes.byref(status))
        if res == DHPGC_OK:
            return status.value
        return res

    def wait_move(self, timeout=5000):
        return self.lib.dhpgc_gripper_wait_for_movement_complete(self.handle, timeout)

    def disconnect(self):

        if self.handle and self.handle.value is not None:
            self.lib.dhpgc_gripper_destroy(ctypes.byref(self.handle))
            self.handle.value = None
            return True
        return False
    
    def __del__(self):
        try:
            if hasattr(self, 'handle') and self.handle and self.handle.value is not None:
                if hasattr(self, 'lib'):
                    try:
                        self.lib.dhpgc_gripper_destroy(ctypes.byref(self.handle))
                    except Exception:
                        pass  
                self.handle.value = None
        except Exception:
            pass
