"""
MECH/ROBT 301 - Applied Industrial Automation
Command-Line Industrial Sorting Cell Digital Twin

Usage Example:
    python cli_sorting_sim.py --n_lead 6 --duration 30 --verbose
    python cli_sorting_sim.py --n_lead 0 --belt_speed 0.20
"""

import argparse
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

# =====================================================================
# DATA STRUCTURES
# =====================================================================
class PartQuality(Enum):
    NONE = 0
    GOOD = 1
    DEFECTIVE = 2

class PusherState(Enum):
    IDLE = 0
    EXTENDING = 1
    DWELLING = 2
    RETRACTING = 3
    FAULT_JAMMED = 4

@dataclass
class Workpiece:
    part_id: int
    quality: PartQuality
    timestamp_infeed: float

# =====================================================================
# INDUSTRIAL CONVEYOR SORTING CELL CLASS
# =====================================================================
class IndustrialSortingCell:
    def __init__(self, n_lead: int, belt_speed: float, scan_time: float, stroke_ext: float, verbose: bool):
        self.verbose = verbose
        
        # System Parameters
        self.n_lead = n_lead
        self.belt_speed = belt_speed
        self.scan_time = scan_time
        self.stroke_ext = stroke_ext
        self.stroke_ret = 0.150
        self.dwell_time = 0.050
        
        # Spatial Quantization
        self.tracking_length = 0.60
        self.cell_width = self.belt_speed * self.scan_time
        self.num_cells = int(self.tracking_length / self.cell_width)
        
        self.idx_inspect = int(0.10 / self.cell_width)
        self.idx_pusher = self.num_cells - 1
        self.idx_trigger = max(0, self.idx_pusher - self.n_lead)

        # Process Image & Internal State
        self.DI_PE1_Trigger = False
        self.DO_Conveyor_Run = True
        self.DO_A1_Extend_Sol = False
        
        self.shift_register: List[Optional[Workpiece]] = [None] * self.num_cells
        self.pusher_fsm_state = PusherState.IDLE
        self.pusher_timer = 0.0
        self.sim_pusher_pos = 0.0  # 0.0 = Retracted, 1.0 = Extended
        
        # Performance KPIs
        self.kpi_total_infeed = 0
        self.kpi_defects_generated = 0
        self.kpi_defects_diverted = 0
        self.kpi_good_passed = 0
        self.kpi_false_rejects = 0
        self.kpi_escaped_defects = 0

        self._sim_part_id_seq = 100
        self._sim_last_infeed_time = -2.0

    def log(self, current_time: float, message: str):
        if self.verbose:
            print(f"[{current_time:06.3f}s] {message}")

    def run_scan_cycle(self, current_time: float):
        """Simulates one deterministic PLC execution cycle."""
        
        # -------------------------------------------------------------
        # 1. READ INPUTS (Update Physical Plant)
        # -------------------------------------------------------------
        # Spawn parts randomly (~1.6s gaps, 30% defect probability)
        if (current_time - self._sim_last_infeed_time > 1.6) and (random.random() < 0.35):
            self.DI_PE1_Trigger = True
            self._sim_last_infeed_time = current_time
        else:
            self.DI_PE1_Trigger = False

        # Cylinder Physical Dynamics
        if self.DO_A1_Extend_Sol:
            self.sim_pusher_pos = min(1.0, self.sim_pusher_pos + (self.scan_time / self.stroke_ext))
        else:
            self.sim_pusher_pos = max(0.0, self.sim_pusher_pos - (self.scan_time / self.stroke_ret))

        # -------------------------------------------------------------
        # 2. EXECUTE LOGIC (Shift Register & FSM)
        # -------------------------------------------------------------
        if self.DO_Conveyor_Run:
            # A. Process part leaving the conveyor
            exit_part = self.shift_register[-1]
            if exit_part is not None:
                if exit_part.quality == PartQuality.DEFECTIVE:
                    self.kpi_escaped_defects += 1
                    self.log(current_time, f"[ALARM] Escape! Defect #{exit_part.part_id} reached Good Bin.")
                else:
                    self.kpi_good_passed += 1

            # B. Shift array downstream
            for i in range(self.num_cells - 1, 0, -1):
                self.shift_register[i] = self.shift_register[i - 1]
            self.shift_register[0] = None

            # C. Ingest new part
            if self.DI_PE1_Trigger:
                self._sim_part_id_seq += 1
                self.kpi_total_infeed += 1
                is_defect = (random.random() < 0.30)
                if is_defect:
                    self.kpi_defects_generated += 1
                
                new_part = Workpiece(self._sim_part_id_seq, PartQuality.DEFECTIVE if is_defect else PartQuality.GOOD, current_time)
                self.shift_register[0] = new_part
                self.log(current_time, f"[INFEED] Part #{new_part.part_id} ingested (Defect: {is_defect})")

        # D. Physical Ejection Physics (Pusher interaction at index 149)
        part_at_pusher = self.shift_register[self.idx_pusher]
        if part_at_pusher is not None and self.sim_pusher_pos >= 0.70:
            # Cylinder is extended enough to strike the part
            if part_at_pusher.quality == PartQuality.DEFECTIVE:
                self.kpi_defects_diverted += 1
                self.log(current_time, f"[REJECT] Defect #{part_at_pusher.part_id} successfully diverted!")
            else:
                self.kpi_false_rejects += 1
                self.log(current_time, f"[WARN] Good Part #{part_at_pusher.part_id} falsely diverted!")
            self.shift_register[self.idx_pusher] = None # Knocked off belt

        # E. Pusher Lead Actuation FSM
        part_at_trigger = self.shift_register[self.idx_trigger]

        if self.pusher_fsm_state == PusherState.IDLE:
            self.DO_A1_Extend_Sol = False
            if part_at_trigger is not None and part_at_trigger.quality == PartQuality.DEFECTIVE:
                self.pusher_fsm_state = PusherState.EXTENDING
                self.pusher_timer = 0.0
                self.log(current_time, f"[FSM] Pre-Triggering Pusher (Lead={self.n_lead} cells) for Part #{part_at_trigger.part_id}")

        elif self.pusher_fsm_state == PusherState.EXTENDING:
            self.DO_A1_Extend_Sol = True
            if self.sim_pusher_pos >= 0.98: # Reached full extension
                self.pusher_fsm_state = PusherState.DWELLING
                self.pusher_timer = 0.0

        elif self.pusher_fsm_state == PusherState.DWELLING:
            self.DO_A1_Extend_Sol = True
            self.pusher_timer += self.scan_time
            if self.pusher_timer >= self.dwell_time:
                self.pusher_fsm_state = PusherState.RETRACTING

        elif self.pusher_fsm_state == PusherState.RETRACTING:
            self.DO_A1_Extend_Sol = False
            if self.sim_pusher_pos <= 0.02: # Reached home
                self.pusher_fsm_state = PusherState.IDLE

# =====================================================================
# COMMAND LINE INTERFACE & MAIN EXECUTION
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Industrial Conveyor Sorting Cell Simulation (Digital Twin)")
    parser.add_argument("--n_lead", type=int, default=6, help="Actuation lead compensation in spatial cells (Default: 6)")
    parser.add_argument("--belt_speed", type=float, default=0.20, help="Conveyor belt speed in m/s (Default: 0.20)")
    parser.add_argument("--scan_time", type=float, default=0.020, help="PLC scan cycle period in seconds (Default: 0.020)")
    parser.add_argument("--stroke_ext", type=float, default=0.120, help="Actuator extension stroke time in seconds (Default: 0.120)")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration in seconds (Default: 30.0)")
    parser.add_argument("--verbose", action="store_true", help="Print real-time event logs to the console")
    
    args = parser.parse_args()

    print("=" * 60)
    print(" DIGITAL TWIN: DISCRETE STATE CONTROL SIMULATION")
    print("=" * 60)
    print(f" Parameters Configured:")
    print(f"  - Duration:    {args.duration} s")
    print(f"  - Belt Speed:  {args.belt_speed} m/s")
    print(f"  - Scan Time:   {args.scan_time * 1000} ms")
    print(f"  - N_Lead:      {args.n_lead} cells")
    print("=" * 60)
    print(" Running simulation (Fast-Forward)...")

    # Initialize and run the simulation headlessly
    cell = IndustrialSortingCell(
        n_lead=args.n_lead,
        belt_speed=args.belt_speed,
        scan_time=args.scan_time,
        stroke_ext=args.stroke_ext,
        verbose=args.verbose
    )

    sim_time = 0.0
    while sim_time <= args.duration:
        cell.run_scan_cycle(sim_time)
        sim_time += args.scan_time

    # =================================================================
    # CALCULATE METRICS (Including Corrected FPY)
    # =================================================================
    total = cell.kpi_total_infeed
    diverted = cell.kpi_defects_diverted
    escaped = cell.kpi_escaped_defects
    good_passed = cell.kpi_good_passed

    ppm = (total / args.duration) * 60.0
    
    # Capture Rate: Diverted / Total Defects Encountered
    total_defects_encountered = diverted + escaped
    capture_rate = (diverted / total_defects_encountered * 100.0) if total_defects_encountered > 0 else 100.0
    
    # Corrected FPY: Truly Good Parts / Total Parts in the Good Bin
    total_in_good_bin = good_passed + escaped
    fpy = (good_passed / total_in_good_bin * 100.0) if total_in_good_bin > 0 else 100.0

    print("\n" + "=" * 60)
    print(" FINAL TELEMETRY & KPIs")
    print("=" * 60)
    print(f" Total Ingested:          {total} parts")
    print(f" Throughput Rate:         {ppm:.1f} PPM")
    print(f" Defects Diverted:        {diverted}")
    print(f" Quality Escapes:         {escaped}")
    print("-" * 60)
    print(f" Defect Capture Rate (η): {capture_rate:.1f}%")
    print(f" First-Pass Yield (FPY):  {fpy:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()