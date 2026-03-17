"""Safety Logger v3 — tracks distinct violation events and phase breakdown."""

import torch


class SafetyLogger:

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        self.total_steps = 0
        self.torque_violation_count = 0
        self.contact_force_violation_count = 0
        self.orientation_violation_count = 0

        # Distinct violation events (transitions from no-viol to viol)
        self.torque_event_count = 0
        self.contact_event_count = 0
        self.orient_event_count = 0
        self._prev_torque_viol = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_contact_viol = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_orient_viol = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Violation occupancy (fraction of time in violation)
        self.torque_occupancy_steps = 0
        self.contact_occupancy_steps = 0
        self.orient_occupancy_steps = 0

        self.fall_count = 0
        self.recovery_success_count = 0
        self.recovery_failure_count = 0
        self.recovery_times = []
        self.violations_during_fall = 0
        self.total_fallen_steps = 0
        self._recovery_had_violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recoveries_with_violation = 0

        # Violation breakdown by type during recovery
        self.torque_viol_during_recovery = 0
        self.contact_viol_during_recovery = 0
        self.orient_viol_during_recovery = 0

        self._prev_is_fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def log_step(self, torque_violation, contact_force_violation, orientation_violation, is_fallen):
        self.total_steps += 1

        self.torque_violation_count += torque_violation.sum().item()
        self.contact_force_violation_count += contact_force_violation.sum().item()
        self.orientation_violation_count += orientation_violation.sum().item()

        # Distinct events
        new_torque = (~self._prev_torque_viol) & torque_violation
        new_contact = (~self._prev_contact_viol) & contact_force_violation
        new_orient = (~self._prev_orient_viol) & orientation_violation
        self.torque_event_count += new_torque.sum().item()
        self.contact_event_count += new_contact.sum().item()
        self.orient_event_count += new_orient.sum().item()
        self._prev_torque_viol = torque_violation.clone()
        self._prev_contact_viol = contact_force_violation.clone()
        self._prev_orient_viol = orientation_violation.clone()

        # Occupancy
        self.torque_occupancy_steps += torque_violation.sum().item()
        self.contact_occupancy_steps += contact_force_violation.sum().item()
        self.orient_occupancy_steps += orientation_violation.sum().item()

        # Falls
        new_falls = (~self._prev_is_fallen) & is_fallen
        self.fall_count += new_falls.sum().item()

        # Coupling
        any_violation = torque_violation | contact_force_violation | orientation_violation
        fallen_and_violating = any_violation & is_fallen
        self.violations_during_fall += fallen_and_violating.sum().item()
        self.total_fallen_steps += is_fallen.sum().item()

        # Coupling by type
        self.torque_viol_during_recovery += (torque_violation & is_fallen).sum().item()
        self.contact_viol_during_recovery += (contact_force_violation & is_fallen).sum().item()
        self.orient_viol_during_recovery += (orientation_violation & is_fallen).sum().item()

        self._recovery_had_violation |= (any_violation & is_fallen)
        self._recovery_had_violation[new_falls] = False

        self._prev_is_fallen = is_fallen.clone()

    def log_recovery(self, recovered_mask, recovery_times):
        n = recovered_mask.sum().item()
        self.recovery_success_count += n
        self.recovery_times.extend(recovery_times.cpu().tolist())
        self.recoveries_with_violation += self._recovery_had_violation[recovered_mask].sum().item()
        self._recovery_had_violation[recovered_mask] = False

    def log_recovery_failure(self, timed_out_mask):
        self.recovery_failure_count += timed_out_mask.sum().item()
        self.recoveries_with_violation += self._recovery_had_violation[timed_out_mask].sum().item()
        self._recovery_had_violation[timed_out_mask] = False

    def summarize(self):
        total_env_steps = max(self.total_steps * self.num_envs, 1)
        total_falls = max(self.fall_count, 1)
        total_recovery_attempts = max(self.recovery_success_count + self.recovery_failure_count, 1)
        mean_recovery_time = (
            sum(self.recovery_times) / len(self.recovery_times)
            if self.recovery_times else float("nan")
        )
        total_fallen_seconds = max(self.total_fallen_steps * 0.02, 0.001)
        viol_per_recovery_sec = self.violations_during_fall / total_fallen_seconds
        pct_recovery_with_viol = self.recoveries_with_violation / total_recovery_attempts

        return {
            # Axis 2: Safety — per-step rates
            "safety/torque_violation_rate": self.torque_violation_count / total_env_steps,
            "safety/contact_force_violation_rate": self.contact_force_violation_count / total_env_steps,
            "safety/orientation_violation_rate": self.orientation_violation_count / total_env_steps,
            "safety/total_violations": (
                self.torque_violation_count + self.contact_force_violation_count + self.orientation_violation_count
            ),
            # Axis 2: Safety — distinct events
            "safety/torque_events": self.torque_event_count,
            "safety/contact_events": self.contact_event_count,
            "safety/orient_events": self.orient_event_count,
            "safety/total_events": self.torque_event_count + self.contact_event_count + self.orient_event_count,
            # Axis 2: Safety — occupancy (fraction of env-steps in violation)
            "safety/torque_occupancy": self.torque_occupancy_steps / total_env_steps,
            "safety/contact_occupancy": self.contact_occupancy_steps / total_env_steps,
            "safety/orient_occupancy": self.orient_occupancy_steps / total_env_steps,
            # Axis 3: Recovery
            "recovery/fall_count": self.fall_count,
            "recovery/success_rate": self.recovery_success_count / total_recovery_attempts,
            "recovery/mean_time_to_upright": mean_recovery_time,
            "recovery/timeout_count": self.recovery_failure_count,
            "recovery/total_attempts": self.recovery_success_count + self.recovery_failure_count,
            # Axis 4: Coupling — aggregate
            "coupling/violations_during_recovery": self.violations_during_fall,
            "coupling/violations_per_fall": self.violations_during_fall / total_falls,
            "coupling/violations_per_recovery_sec": viol_per_recovery_sec,
            "coupling/pct_recovery_with_violation": pct_recovery_with_viol,
            # Axis 4: Coupling — by type during recovery
            "coupling/torque_viol_during_recovery": self.torque_viol_during_recovery,
            "coupling/contact_viol_during_recovery": self.contact_viol_during_recovery,
            "coupling/orient_viol_during_recovery": self.orient_viol_during_recovery,
        }
