"""
Mock pose data generator for testing and development.

This module provides synthetic pose estimation data for use in development
and testing environments ONLY. The generated data mimics realistic human
pose detection outputs including keypoints, bounding boxes, and activities.

WARNING: This module uses random number generation intentionally for test data.
Do NOT use this module in production data paths.
"""

import math
import random
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Banner displayed when mock pose mode is active
MOCK_POSE_BANNER = """
================================================================================
  WARNING: MOCK POSE MODE ACTIVE - Using synthetic pose data

  All pose detections are randomly generated and do NOT represent real humans.
  For real pose estimation, provide trained model weights and real CSI data.
  See docs/hardware-setup.md for configuration instructions.
================================================================================
"""

_banner_shown = False


def _show_banner() -> None:
    """Display the mock pose mode warning banner (once per session)."""
    global _banner_shown
    if not _banner_shown:
        logger.warning(MOCK_POSE_BANNER)
        _banner_shown = True


# ---------------------------------------------------------------------------
# COCO-17 keypoint names (canonical order)
# ---------------------------------------------------------------------------
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ---------------------------------------------------------------------------
# Anatomically correct base pose templates (normalized 0-1 coordinates)
# Each template is a list of (x, y) for the 17 COCO keypoints.
# Poses are designed for a coordinate system where:
#   x: 0 = left edge, 1 = right edge
#   y: 0 = top edge,  1 = bottom edge
# ---------------------------------------------------------------------------

def _standing_pose(cx: float = 0.5, scale: float = 1.0) -> List[tuple]:
    """Generate an anatomically correct standing pose centered at cx."""
    s = scale * 0.12  # limb unit scale
    return [
        (cx, 0.15),                           # 0  nose
        (cx - 0.012, 0.13),                   # 1  left_eye
        (cx + 0.012, 0.13),                   # 2  right_eye
        (cx - 0.025, 0.145),                  # 3  left_ear
        (cx + 0.025, 0.145),                  # 4  right_ear
        (cx - s, 0.25),                       # 5  left_shoulder
        (cx + s, 0.25),                       # 6  right_shoulder
        (cx - s * 1.1, 0.38),                 # 7  left_elbow
        (cx + s * 1.1, 0.38),                 # 8  right_elbow
        (cx - s * 0.9, 0.48),                 # 9  left_wrist
        (cx + s * 0.9, 0.48),                 # 10 right_wrist
        (cx - s * 0.55, 0.52),                # 11 left_hip
        (cx + s * 0.55, 0.52),                # 12 right_hip
        (cx - s * 0.55, 0.70),                # 13 left_knee
        (cx + s * 0.55, 0.70),                # 14 right_knee
        (cx - s * 0.55, 0.88),                # 15 left_ankle
        (cx + s * 0.55, 0.88),                # 16 right_ankle
    ]


def _sitting_pose(cx: float = 0.5, scale: float = 1.0) -> List[tuple]:
    """Generate an anatomically correct sitting pose centered at cx."""
    s = scale * 0.12
    return [
        (cx, 0.22),                           # 0  nose
        (cx - 0.012, 0.20),                   # 1  left_eye
        (cx + 0.012, 0.20),                   # 2  right_eye
        (cx - 0.025, 0.215),                  # 3  left_ear
        (cx + 0.025, 0.215),                  # 4  right_ear
        (cx - s, 0.32),                       # 5  left_shoulder
        (cx + s, 0.32),                       # 6  right_shoulder
        (cx - s * 1.2, 0.44),                 # 7  left_elbow
        (cx + s * 1.2, 0.44),                 # 8  right_elbow
        (cx - s * 1.0, 0.52),                 # 9  left_wrist
        (cx + s * 1.0, 0.52),                 # 10 right_wrist
        (cx - s * 0.55, 0.55),                # 11 left_hip
        (cx + s * 0.55, 0.55),                # 12 right_hip
        (cx - s * 0.7, 0.56),                 # 13 left_knee (bent forward)
        (cx + s * 0.7, 0.56),                 # 14 right_knee
        (cx - s * 0.7, 0.72),                 # 15 left_ankle (under knee)
        (cx + s * 0.7, 0.72),                 # 16 right_ankle
    ]


# ---------------------------------------------------------------------------
# Animation helpers — smooth sinusoidal motion for realistic movement
# ---------------------------------------------------------------------------

class _PersonAnimator:
    """Animates a single person with smooth, anatomically plausible motion."""

    def __init__(
        self,
        person_id: int,
        activity: str,
        center_x: float,
        scale: float = 1.0,
        walk_speed: float = 1.0,
        walk_range: float = 0.20,
    ):
        self.person_id = person_id
        self.activity = activity
        self.center_x = center_x
        self.scale = scale
        self.walk_speed = walk_speed
        self.walk_range = walk_range
        self._phase_offset = random.uniform(0, 2 * math.pi)
        self._breath_offset = random.uniform(0, 2 * math.pi)

    def get_keypoints(self, t: float) -> List[Dict[str, Any]]:
        """Return animated keypoints at time t (seconds since epoch)."""
        phase = t * self.walk_speed + self._phase_offset

        if self.activity == "walking":
            return self._walking_keypoints(phase, t)
        elif self.activity == "sitting":
            return self._sitting_keypoints(phase, t)
        else:
            return self._standing_keypoints(phase, t)

    def _standing_keypoints(self, phase: float, t: float) -> List[Dict[str, Any]]:
        base = _standing_pose(self.center_x, self.scale)
        animated = []
        breath = math.sin(t * 0.8 + self._breath_offset) * 0.005
        sway_x = math.sin(phase * 0.3) * 0.008
        sway_y = math.cos(phase * 0.25) * 0.004

        for i, (bx, by) in enumerate(base):
            x = bx + sway_x
            y = by + sway_y
            # Breathing: shoulders and torso move slightly
            if i in (5, 6, 7, 8):  # shoulders and elbows
                y += breath
            # Slight arm swing
            if i in (9, 10):  # wrists
                x += math.sin(phase * 0.5) * 0.01 * (1 if i == 9 else -1)
            animated.append(self._make_keypoint(i, x, y))

        return animated

    def _walking_keypoints(self, phase: float, t: float) -> List[Dict[str, Any]]:
        # Walking: oscillate center_x, animate legs and arms in gait cycle
        walk_x = self.center_x + math.sin(phase * 0.4) * self.walk_range
        base = _standing_pose(walk_x, self.scale)
        animated = []

        gait = phase * 2.0  # gait cycle frequency
        stride = 0.04 * self.scale  # how far legs swing
        arm_swing = 0.025 * self.scale
        bob = math.sin(gait * 2) * 0.008  # vertical bobbing

        for i, (bx, by) in enumerate(base):
            x, y = bx, by + bob

            # Head slight bob
            if i <= 4:
                y += bob * 0.5

            # Arm swing (opposite to legs)
            if i == 7:   # left_elbow
                x += math.sin(gait) * arm_swing
                y += math.cos(gait) * 0.01
            elif i == 8:  # right_elbow
                x -= math.sin(gait) * arm_swing
                y -= math.cos(gait) * 0.01
            elif i == 9:  # left_wrist
                x += math.sin(gait) * arm_swing * 1.5
                y += math.cos(gait) * 0.015
            elif i == 10: # right_wrist
                x -= math.sin(gait) * arm_swing * 1.5
                y -= math.cos(gait) * 0.015

            # Leg gait
            if i == 13:  # left_knee
                x += math.sin(gait) * stride * 0.5
                y -= abs(math.sin(gait)) * 0.02
            elif i == 14: # right_knee
                x -= math.sin(gait) * stride * 0.5
                y -= abs(math.cos(gait)) * 0.02
            elif i == 15: # left_ankle
                x += math.sin(gait) * stride
                y -= abs(math.sin(gait)) * 0.03
            elif i == 16: # right_ankle
                x -= math.sin(gait) * stride
                y -= abs(math.cos(gait)) * 0.03

            animated.append(self._make_keypoint(i, x, y))

        return animated

    def _sitting_keypoints(self, phase: float, t: float) -> List[Dict[str, Any]]:
        base = _sitting_pose(self.center_x, self.scale)
        animated = []
        breath = math.sin(t * 0.7 + self._breath_offset) * 0.004
        fidget = math.sin(phase * 0.2) * 0.005

        for i, (bx, by) in enumerate(base):
            x, y = bx, by
            # Breathing
            if i in (5, 6):
                y += breath
            # Hand fidgeting
            if i in (9, 10):
                x += fidget * (1 if i == 9 else -1)
                y += math.cos(phase * 0.3) * 0.005
            # Slight head movement
            if i <= 4:
                x += math.sin(phase * 0.15) * 0.006
            animated.append(self._make_keypoint(i, x, y))

        return animated

    def _make_keypoint(self, idx: int, x: float, y: float) -> Dict[str, Any]:
        # Clamp to valid range
        x = max(0.02, min(0.98, x))
        y = max(0.02, min(0.98, y))
        # Core joints get higher confidence; extremities slightly lower
        if idx <= 4:
            conf = random.uniform(0.82, 0.95)
        elif idx <= 10:
            conf = random.uniform(0.75, 0.92)
        else:
            conf = random.uniform(0.65, 0.88)
        return {
            "name": KEYPOINT_NAMES[idx],
            "x": round(x, 5),
            "y": round(y, 5),
            "confidence": round(conf, 3),
        }

    def get_bounding_box(self, keypoints: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute tight bounding box from keypoints."""
        xs = [kp["x"] for kp in keypoints]
        ys = [kp["y"] for kp in keypoints]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        margin = 0.02
        return {
            "x": max(0.0, x_min - margin),
            "y": max(0.0, y_min - margin),
            "width": min(1.0, (x_max - x_min) + 2 * margin),
            "height": min(1.0, (y_max - y_min) + 2 * margin),
        }


# ---------------------------------------------------------------------------
# Persistent animator pool (survives across calls to generate_mock_poses)
# ---------------------------------------------------------------------------

_animators: Optional[List[_PersonAnimator]] = None


def _get_animators(max_persons: int = 2) -> List[_PersonAnimator]:
    """Return (or create) the persistent animator pool."""
    global _animators
    if _animators is None:
        configs = [
            {"person_id": 0, "activity": "walking",  "center_x": 0.30, "scale": 1.0,  "walk_speed": 1.0, "walk_range": 0.18},
            {"person_id": 1, "activity": "standing", "center_x": 0.68, "scale": 0.95, "walk_speed": 0.6, "walk_range": 0.0},
        ]
        _animators = [_PersonAnimator(**cfg) for cfg in configs[:max_persons]]
    return _animators[:max_persons]


# ---------------------------------------------------------------------------
# Public API (drop-in replacement for the old random generator)
# ---------------------------------------------------------------------------

def generate_mock_keypoints() -> List[Dict[str, Any]]:
    """Generate mock keypoints for a single person.

    Returns:
        List of 17 COCO-format keypoint dictionaries with name, x, y, confidence.
    """
    # Use the standing animator at center for single-person calls
    animator = _PersonAnimator(person_id=0, activity="standing", center_x=0.5)
    return animator.get_keypoints(time.time())


def generate_mock_bounding_box() -> Dict[str, float]:
    """Generate a mock bounding box for a single person.

    Returns:
        Dictionary with x, y, width, height as normalized coordinates.
    """
    x = random.uniform(0.1, 0.6)
    y = random.uniform(0.1, 0.6)
    width = random.uniform(0.2, 0.4)
    height = random.uniform(0.3, 0.5)

    return {"x": x, "y": y, "width": width, "height": height}


def generate_mock_poses(max_persons: int = 2) -> List[Dict[str, Any]]:
    """Generate mock pose detections for testing.

    Args:
        max_persons: Maximum number of persons to generate (1 to max_persons).

    Returns:
        List of pose detection dictionaries with anatomically correct,
        smoothly animated skeleton data.
    """
    _show_banner()

    animators = _get_animators(min(2, max_persons))
    t = time.time()
    poses = []

    for anim in animators:
        keypoints = anim.get_keypoints(t)
        bbox = anim.get_bounding_box(keypoints)

        # Overall confidence: average of all keypoint confidences
        avg_conf = sum(kp["confidence"] for kp in keypoints) / len(keypoints)

        pose = {
            "person_id": anim.person_id,
            "confidence": round(avg_conf, 3),
            "keypoints": keypoints,
            "bounding_box": bbox,
            "activity": anim.activity,
            "timestamp": datetime.now().isoformat(),
        }
        poses.append(pose)

    return poses


# ---------------------------------------------------------------------------
# Zone / historical / statistics helpers (unchanged from original)
# ---------------------------------------------------------------------------

def generate_mock_zone_occupancy(zone_id: str) -> Dict[str, Any]:
    """Generate mock zone occupancy data.

    Args:
        zone_id: Zone identifier.

    Returns:
        Dictionary with occupancy count and person details.
    """
    _show_banner()

    count = random.randint(0, 5)
    persons = []

    for i in range(count):
        persons.append({
            "person_id": f"person_{i}",
            "confidence": random.uniform(0.7, 0.95),
            "activity": random.choice(["standing", "sitting", "walking"]),
        })

    return {
        "count": count,
        "max_occupancy": 10,
        "persons": persons,
        "timestamp": datetime.now(),
    }


def generate_mock_zones_summary(
    zone_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate mock zones summary data.

    Args:
        zone_ids: List of zone identifiers. Defaults to zone_1 through zone_4.

    Returns:
        Dictionary with per-zone occupancy and aggregate counts.
    """
    _show_banner()

    zones = zone_ids or ["zone_1", "zone_2", "zone_3", "zone_4"]
    zone_data = {}
    total_persons = 0
    active_zones = 0

    for zone_id in zones:
        count = random.randint(0, 3)
        zone_data[zone_id] = {
            "occupancy": count,
            "max_occupancy": 10,
            "status": "active" if count > 0 else "inactive",
        }
        total_persons += count
        if count > 0:
            active_zones += 1

    return {
        "total_persons": total_persons,
        "zones": zone_data,
        "active_zones": active_zones,
    }


def generate_mock_historical_data(
    start_time: datetime,
    end_time: datetime,
    zone_ids: Optional[List[str]] = None,
    aggregation_interval: int = 300,
    include_raw_data: bool = False,
) -> Dict[str, Any]:
    """Generate mock historical pose data.

    Args:
        start_time: Start of the time range.
        end_time: End of the time range.
        zone_ids: Zones to include. Defaults to zone_1, zone_2, zone_3.
        aggregation_interval: Seconds between data points.
        include_raw_data: Whether to include simulated raw detections.

    Returns:
        Dictionary with aggregated_data, optional raw_data, and total_records.
    """
    _show_banner()

    zones = zone_ids or ["zone_1", "zone_2", "zone_3"]
    current_time = start_time
    aggregated_data = []
    raw_data = [] if include_raw_data else None

    while current_time < end_time:
        data_point = {
            "timestamp": current_time,
            "total_persons": random.randint(0, 8),
            "zones": {},
        }

        for zone_id in zones:
            data_point["zones"][zone_id] = {
                "occupancy": random.randint(0, 3),
                "avg_confidence": random.uniform(0.7, 0.95),
            }

        aggregated_data.append(data_point)

        if include_raw_data:
            for _ in range(random.randint(0, 5)):
                raw_data.append({
                    "timestamp": current_time + timedelta(seconds=random.randint(0, aggregation_interval)),
                    "person_id": f"person_{random.randint(1, 10)}",
                    "zone_id": random.choice(zones),
                    "confidence": random.uniform(0.5, 0.95),
                    "activity": random.choice(["standing", "sitting", "walking"]),
                })

        current_time += timedelta(seconds=aggregation_interval)

    return {
        "aggregated_data": aggregated_data,
        "raw_data": raw_data,
        "total_records": len(aggregated_data),
    }


def generate_mock_recent_activities(
    zone_id: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Generate mock recent activity data.

    Args:
        zone_id: Optional zone filter. If None, random zones are used.
        limit: Number of activities to generate.

    Returns:
        List of activity dictionaries.
    """
    _show_banner()

    activities = []

    for i in range(limit):
        activity = {
            "activity_id": f"activity_{i}",
            "person_id": f"person_{random.randint(1, 5)}",
            "zone_id": zone_id or random.choice(["zone_1", "zone_2", "zone_3"]),
            "activity": random.choice(["standing", "sitting", "walking", "lying"]),
            "confidence": random.uniform(0.6, 0.95),
            "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 60)),
            "duration_seconds": random.randint(10, 300),
        }
        activities.append(activity)

    return activities


def generate_mock_statistics(
    start_time: datetime,
    end_time: datetime,
) -> Dict[str, Any]:
    """Generate mock pose estimation statistics.

    Args:
        start_time: Start of the statistics period.
        end_time: End of the statistics period.

    Returns:
        Dictionary with detection counts, rates, and distributions.
    """
    _show_banner()

    total_detections = random.randint(100, 1000)
    successful_detections = int(total_detections * random.uniform(0.8, 0.95))

    return {
        "total_detections": total_detections,
        "successful_detections": successful_detections,
        "failed_detections": total_detections - successful_detections,
        "success_rate": successful_detections / total_detections,
        "average_confidence": random.uniform(0.75, 0.90),
        "average_processing_time_ms": random.uniform(50, 200),
        "unique_persons": random.randint(5, 20),
        "most_active_zone": random.choice(["zone_1", "zone_2", "zone_3"]),
        "activity_distribution": {
            "standing": random.uniform(0.3, 0.5),
            "sitting": random.uniform(0.2, 0.4),
            "walking": random.uniform(0.1, 0.3),
            "lying": random.uniform(0.0, 0.1),
        },
    }
