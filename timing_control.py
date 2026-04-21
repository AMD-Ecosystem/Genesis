#!/usr/bin/env python3
"""
Centralized timing control for Genesis detailed analysis
"""

import os
from enum import Enum

class TimingMode(Enum):
    DISABLED = "disabled"
    BASIC = "basic"
    DETAILED = "detailed"

class TimingController:
    """Centralized control for Genesis timing features"""

    def __init__(self):
        self._mode = TimingMode.DISABLED
        self._debug_prints = False
        self._sync_enabled = False

    def configure_from_env(self):
        """Configure timing based on environment variables"""
        timing_env = os.getenv('GENESIS_TIMING_MODE', 'disabled').lower()

        if timing_env == 'detailed':
            self.enable_detailed_timing()
        elif timing_env == 'basic':
            self.enable_basic_timing()
        else:
            self.disable_timing()

        # Override debug prints separately if needed
        if os.getenv('GENESIS_DEBUG_PRINTS', 'false').lower() == 'true':
            self._debug_prints = True

    def enable_detailed_timing(self):
        """Enable full detailed timing with individual kernel breakdown"""
        self._mode = TimingMode.DETAILED
        self._debug_prints = True
        self._sync_enabled = True

    def enable_basic_timing(self):
        """Enable basic timing (overall kernel_step_2 timing only)"""
        self._mode = TimingMode.BASIC
        self._debug_prints = False
        self._sync_enabled = True

    def disable_timing(self):
        """Disable all timing (fastest execution)"""
        self._mode = TimingMode.DISABLED
        self._debug_prints = False
        self._sync_enabled = False

    @property
    def is_detailed_timing_enabled(self):
        return self._mode == TimingMode.DETAILED

    @property
    def is_basic_timing_enabled(self):
        return self._mode in [TimingMode.BASIC, TimingMode.DETAILED]

    @property
    def debug_prints_enabled(self):
        return self._debug_prints

    @property
    def sync_enabled(self):
        return self._sync_enabled

    @property
    def timing_disabled(self):
        return self._mode == TimingMode.DISABLED

    def print_debug(self, message):
        """Print debug message only if debug prints are enabled"""
        if self._debug_prints:
            print(message)

# Global timing controller instance
timing_controller = TimingController()

def configure_timing_from_args(args):
    """Configure timing based on command line arguments"""
    if hasattr(args, 'enable_detailed_timing') and args.enable_detailed_timing:
        timing_controller.enable_detailed_timing()
    elif hasattr(args, 'enable_debug_timing') and args.enable_debug_timing:
        timing_controller.enable_basic_timing()
    else:
        timing_controller.disable_timing()

    # Also check environment variables
    timing_controller.configure_from_env()

def should_use_detailed_timing():
    """Quick check for detailed timing"""
    return timing_controller.is_detailed_timing_enabled

def should_use_basic_timing():
    """Quick check for basic timing"""
    return timing_controller.is_basic_timing_enabled