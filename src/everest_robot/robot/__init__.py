"""The Everest robot SDK layer.

Everything under this package sits between the durable workflow in
:mod:`everest_robot.workflow` and the hardware/policy SDKs. Third-party robotics
dependencies (``maker_arm``, ``lerobot``) are imported lazily inside the modules that
need them, so the contracts, configuration, motion planning and fakes remain importable
and testable without the ``hardware`` extra installed.
"""
