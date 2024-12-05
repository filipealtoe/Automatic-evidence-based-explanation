import ctypes as ct
from ctypes.wintypes import UINT, DWORD
from typing import Callable, Optional

class mmPeriodicTimer:
    """Periodic timer based upon the Windows Multimedia library
    This produces a timed callback with high precision ( ~ < 1ms difference)
    executed outside of a threaded structure (based on Windows runtime ).
    The advantage of this is that the Python GIL limitation is avoided,
    and an aligned "lag" time between all Python processes is not generated;

    The disadvantage is that since this is thread independent, the callback
    procedure must complete its tasks before the next callback is executed 
    otherwise collisions may occur

    This is based on the example:
    https://stackoverflow.com/questions/10717589/how-to-implement-high-speed-consistent-sampling
    """

    timeproc = ct.WINFUNCTYPE(None, ct.c_uint, ct.c_uint, DWORD, DWORD, DWORD)
    timeSetEvent = ct.windll.winmm.timeSetEvent
    timeKillEvent = ct.windll.winmm.timeKillEvent

    def Tick(self):
        self.i_reps += 1
        self.tickFunc()

        if not self.periodic:
            self.stop()

        if self.n_reps is not None and self.i_reps >= self.n_reps:
            self.stop()

    def CallBack(self, uID, uMsg, dwUser, dw1, dw2):
        if self.running:
            self.Tick()

    def __init__(
        self,
        interval: int,
        tickfunc: Callable,
        resolution: Optional[int] = 0,
        stopFunc: Optional[Callable] = None,
        periodic: Optional[bool] = True,
        n_reps: Optional[int] = None,
    ):
        self.interval = UINT(int(interval * 1000))
        self.resolution = UINT(int(resolution * 1000))
        self.tickFunc = tickfunc
        self.stopFunc = stopFunc
        self.periodic = periodic
        self.n_reps = n_reps
        if not self.periodic and self.n_reps is not None:
            raise ValueError("n_reps must be None if periodic is False")

        self.i_reps = 0
        self.id = None
        self.running = False
        self.calbckfn = self.timeproc(self.CallBack)

    def start(self, instant: bool = False):
        if not self.running:
            self.running = True
            if instant:
                self.Tick()

            self.id = self.timeSetEvent(
                self.interval,
                self.resolution,
                self.calbckfn,
                ct.c_ulong(0),
                ct.c_uint(self.periodic),
            )

    def stop(self):
        if self.running:
            self.timeKillEvent(self.id)
            self.running = False

            if self.stopFunc:
                self.stopFunc()