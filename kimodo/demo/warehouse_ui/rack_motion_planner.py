"""Geometry helpers for prompt-driven human walking motions to warehouse racks."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass


_RACK_PROMPT_RE = re.compile(r"\brack[\s_-]*(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CardinalRoute:
    positions: tuple[tuple[float, float, float], ...]
    headings: tuple[float, ...]
    approach_position: tuple[float, float, float]
    final_heading: float
    turn_degrees: tuple[int, ...]


@dataclass(frozen=True)
class RoutePrimitive:
    kind: str
    value: float


@dataclass(frozen=True)
class RackPickRequest:
    rack_name: str
    object_index: int
    shelf_number: int = 4
    target_height_m: float | None = None
    hip_height_m: float | None = None


def cardinal_rack_route_required_seconds(
    *,
    approach_position: Sequence[float],
    final_heading: float,
    fps: float,
    initial_heading: float = 0.0,
    turn_seconds_per_90: float = 0.75,
    walk_speed_m_s: float = 0.90,
    first_axis: str | None = None,
) -> float:
    """Return the minimum duration needed by ``plan_cardinal_rack_route``."""
    if fps <= 0.0 or walk_speed_m_s <= 0.0:
        raise ValueError("fps and walk_speed_m_s must be positive")

    target = tuple(float(value) for value in approach_position)
    start = (0.0, 0.0, 0.0)
    if first_axis not in (None, "x", "z"):
        raise ValueError("first_axis must be 'x', 'z', or None")

    points = [start]
    use_z_first = first_axis == "z" or (first_axis is None and abs(math.sin(final_heading)) < 0.5)
    if use_z_first:
        _append_unique_point(points, (0.0, 0.0, target[2]))
    else:
        _append_unique_point(points, (target[0], 0.0, 0.0))
    _append_unique_point(points, target)

    actions: list[tuple[str, object]] = []
    current_heading = initial_heading
    for start_point, end_point in zip(points, points[1:]):
        movement_heading = forward_route_heading(start_point, end_point)
        turn_delta = _normalize_angle(movement_heading - current_heading)
        if not math.isclose(turn_delta, 0.0, abs_tol=1e-8):
            degrees = int(round(abs(math.degrees(turn_delta))))
            if degrees not in (90, 180):
                raise ValueError(f"Cardinal route produced a non-cardinal turn: {degrees} degrees")
            actions.append(("turn", turn_delta))
        actions.append(("move", (start_point, end_point)))
        current_heading = movement_heading

    final_turn = _normalize_angle(final_heading - current_heading)
    if not math.isclose(final_turn, 0.0, abs_tol=1e-8):
        degrees = int(round(abs(math.degrees(final_turn))))
        if degrees not in (90, 180):
            raise ValueError(f"Final rack-facing turn is not cardinal: {degrees} degrees")
        actions.append(("turn", final_turn))

    turn_steps = [
        max(1, int(round(turn_seconds_per_90 * fps * abs(float(value)) / (math.pi / 2.0))))
        for kind, value in actions
        if kind == "turn"
    ]
    move_distances = [
        math.hypot(value[1][0] - value[0][0], value[1][2] - value[0][2])
        for kind, value in actions
        if kind == "move"
    ]
    move_steps = [max(1, int(round(distance / walk_speed_m_s * fps))) for distance in move_distances]
    return (sum(turn_steps) + sum(move_steps) + 1) / fps


def requested_rack_name(prompts: Sequence[str]) -> str | None:
    """Return a normalized rack name when one rack is named in one prompt."""
    if len(prompts) != 1:
        return None
    match = _RACK_PROMPT_RE.search(prompts[0])
    return f"rack_{int(match.group(1))}" if match is not None else None


def requested_base_rack_name(prompts: Sequence[str]) -> str | None:
    """Return a rack only for explicit Tara/base navigation prompts."""
    if len(prompts) != 1 or not re.search(r"\b(?:tara\s*)?base\b", prompts[0], re.IGNORECASE):
        return None
    return requested_rack_name(prompts)


def requested_base_return_rack_name(prompts: Sequence[str]) -> str | None:
    """Return a rack for explicit base-from-rack-to-origin prompts."""
    if len(prompts) != 1:
        return None
    normalized = " ".join(prompts[0].strip().lower().replace("_", " ").split())
    if "base" not in normalized or "origin" not in normalized:
        return None
    if not (normalized.startswith("return ") or " from rack" in normalized):
        return None
    return requested_rack_name(prompts)


def requested_human_return_rack_name(prompts: Sequence[str]) -> str | None:
    """Return a rack for human return prompts, excluding base commands."""
    if len(prompts) != 1:
        return None
    normalized = " ".join(prompts[0].strip().lower().replace("_", " ").split())
    if "base" in normalized or not ("origin" in normalized or "counter" in normalized):
        return None
    if not (normalized.startswith("return ") or " from rack" in normalized):
        return None
    return requested_rack_name(prompts)


def requested_rack_pick(prompts: Sequence[str]) -> RackPickRequest | None:
    """Parse ``pick object 1/2/3 from rack N shelf 1..5`` prompts."""
    if len(prompts) != 1:
        return None
    prompt = prompts[0]
    if not re.search(r"\b(?:pick|take|grab)\b", prompt, re.IGNORECASE):
        return None
    rack_name = requested_rack_name(prompts)
    object_match = re.search(r"\b(?:object|item)\s*[_-]?(\d+)\b", prompt, re.IGNORECASE)
    shelf_match = re.search(r"\bshelf\s*[_-]?(\d+)\b", prompt, re.IGNORECASE)
    if rack_name is None or object_match is None or shelf_match is None:
        return None
    object_index = int(object_match.group(1))
    shelf_number = int(shelf_match.group(1))
    if object_index not in {1, 2, 3}:
        raise ValueError("Rack object number must be 1, 2, or 3")
    if shelf_number not in {1, 2, 3, 4, 5}:
        raise ValueError("Rack shelf number must be 1, 2, 3, 4, or 5")
    hip_height_match = re.search(
        r"\b(?:hip|hips|pelvis)\s*(?:at|to|height)?\s*"
        r"(\d+(?:\.\d+)?)\s*(cm|centimeter|centimeters|m|meter|meters)\b",
        prompt,
        re.IGNORECASE,
    )
    hip_height_m = None
    if hip_height_match is not None:
        value = float(hip_height_match.group(1))
        unit = hip_height_match.group(2).lower()
        hip_height_m = value / 100.0 if unit.startswith(("cm", "centimeter")) else value
        if not 0.15 <= hip_height_m <= 1.20:
            raise ValueError("Rack pick hip height must be between 15 cm and 1.2 m")

    height_match = None if hip_height_match is not None else re.search(
        r"\b(?:at|height|target|shelf\s+height)\s*"
        r"(\d+(?:\.\d+)?)\s*(cm|centimeter|centimeters|m|meter|meters)\b",
        prompt,
        re.IGNORECASE,
    )
    target_height_m = None
    if height_match is not None:
        value = float(height_match.group(1))
        unit = height_match.group(2).lower()
        target_height_m = value / 100.0 if unit.startswith(("cm", "centimeter")) else value
        if not 0.05 <= target_height_m <= 1.50:
            raise ValueError("Rack pick target height must be between 5 cm and 1.5 m")
    return RackPickRequest(rack_name, object_index, shelf_number, target_height_m, hip_height_m)


def rack_shelf_object_position(
    rack_center: Sequence[float],
    rack_yaw: float,
    shelf_surface_height: float,
    object_index: int,
    *,
    face_normal_half_extent_m: float,
    front_edge_inset_m: float = 0.05,
    lateral_spacing_m: float = 0.15,
    object_height_m: float = 0.06,
) -> tuple[float, float, float]:
    """Return one shelf-object center in world coordinates.

    The designated customer-facing side is local +X. Objects sit 5 cm
    behind that edge and are spaced at -15, 0, and +15 cm along local Z.
    """
    if len(rack_center) != 3:
        raise ValueError("rack_center must contain X, Y, and Z")
    if object_index not in {1, 2, 3}:
        raise ValueError("object_index must be 1, 2, or 3")
    if not 0.0 <= front_edge_inset_m < face_normal_half_extent_m:
        raise ValueError("front_edge_inset_m must remain inside the shelf edge")
    local_x = face_normal_half_extent_m - front_edge_inset_m
    local_z = (object_index - 2) * lateral_spacing_m
    cos_yaw = math.cos(rack_yaw)
    sin_yaw = math.sin(rack_yaw)
    world_x = float(rack_center[0]) + cos_yaw * local_x + sin_yaw * local_z
    world_z = float(rack_center[2]) - sin_yaw * local_x + cos_yaw * local_z
    world_y = float(rack_center[1]) + shelf_surface_height + object_height_m / 2.0
    return (world_x, world_y, world_z)


def choose_safer_turn_side(
    position: Sequence[float],
    heading: float,
    x_limits: tuple[float, float],
    z_limits: tuple[float, float],
    obstacles: Sequence[tuple[float, float, float, float]],
    probe_distance: float = 0.40,
) -> str:
    """Choose the side with more free space for an in-place turnaround.

    Obstacles are axis-aligned ``(center_x, center_z, half_x, half_z)`` boxes.
    """
    x, _, z = (float(value) for value in position)
    left_vector = (-math.cos(heading), math.sin(heading))

    def clearance(probe_x: float, probe_z: float) -> float:
        boundary_clearance = min(
            probe_x - x_limits[0],
            x_limits[1] - probe_x,
            probe_z - z_limits[0],
            z_limits[1] - probe_z,
        )
        obstacle_clearances = []
        for center_x, center_z, half_x, half_z in obstacles:
            delta_x = abs(probe_x - center_x) - half_x
            delta_z = abs(probe_z - center_z) - half_z
            outside_distance = math.hypot(max(delta_x, 0.0), max(delta_z, 0.0))
            obstacle_clearances.append(
                outside_distance if delta_x > 0.0 or delta_z > 0.0 else max(delta_x, delta_z)
            )
        return min([boundary_clearance, *obstacle_clearances])

    left_probe = (x + left_vector[0] * probe_distance, z + left_vector[1] * probe_distance)
    right_probe = (x - left_vector[0] * probe_distance, z - left_vector[1] * probe_distance)
    return "left" if clearance(*left_probe) >= clearance(*right_probe) else "right"


def rack_front_approach_position(
    rack_center: Sequence[float],
    rack_yaw_rad: float,
    rack_depth_m: float,
    clearance_m: float,
) -> tuple[float, float, float]:
    """Return a ground point in front of a rack's local positive-Z face.

    The rack scene uses Y-up coordinates. ``clearance_m`` is measured from
    the outside edge of the rack, rather than from its center.
    """
    if len(rack_center) != 3:
        raise ValueError("rack_center must contain X, Y, and Z")
    if rack_depth_m <= 0.0:
        raise ValueError("rack_depth_m must be positive")
    if clearance_m < 0.0:
        raise ValueError("clearance_m must be non-negative")

    center_x, center_y, center_z = (float(value) for value in rack_center)
    center_to_target_m = rack_depth_m / 2.0 + clearance_m
    return (
        center_x + math.sin(rack_yaw_rad) * center_to_target_m,
        center_y,
        center_z + math.cos(rack_yaw_rad) * center_to_target_m,
    )


def rack_approach_pose_inside_boundary(
    rack_center: Sequence[float],
    rack_yaw_rad: float,
    rack_depth_m: float,
    clearance_m: float,
    x_limits: tuple[float, float],
    z_limits: tuple[float, float],
    required_face_sign: float | None = None,
) -> tuple[tuple[float, float, float], float]:
    """Choose a centered shelf-opening face that is inside the work area.

    The shelf-opening face spans the rack's local X/width dimension. Its
    outward normal is therefore local +Z or -Z; local X faces are the rack's
    depth/length sides and are intentionally never selected here.
    """
    center = tuple(float(value) for value in rack_center)
    face_distance = rack_depth_m / 2.0 + clearance_m
    if required_face_sign is not None and required_face_sign not in (-1.0, 1.0):
        raise ValueError("required_face_sign must be -1, 1, or None")
    face_signs = (required_face_sign,) if required_face_sign is not None else (1.0, -1.0)
    candidates = []
    for face_sign in face_signs:
        normal_x = face_sign * math.sin(rack_yaw_rad)
        normal_z = face_sign * math.cos(rack_yaw_rad)
        approach = (
            center[0] + normal_x * face_distance,
            center[1],
            center[2] + normal_z * face_distance,
        )
        inside = x_limits[0] <= approach[0] <= x_limits[1] and z_limits[0] <= approach[2] <= z_limits[1]
        if inside:
            facing_rack = math.atan2(-normal_x, -normal_z)
            distance_from_origin = math.hypot(approach[0], approach[2])
            candidates.append((distance_from_origin, -face_sign, approach, facing_rack))
    if not candidates:
        face_description = "required shelf-opening face" if required_face_sign is not None else "rack face"
        raise ValueError(f"No {face_description} has a valid approach point inside the work-area boundary")
    _, _, approach, facing_rack = min(candidates)
    return approach, facing_rack


def rack_width_side_approach_pose(
    rack_center: Sequence[float],
    rack_yaw_rad: float,
    rack_width_m: float,
    clearance_m: float,
    x_limits: tuple[float, float],
    z_limits: tuple[float, float],
    face_sign: float = 1.0,
) -> tuple[tuple[float, float, float], float]:
    """Return the centered approach to the rack's broad width side.

    This face is normal to local X. For Rack 1, local +X is the inward-facing
    broad shelf side shown in the warehouse view.
    """
    if face_sign not in (-1.0, 1.0):
        raise ValueError("face_sign must be -1 or 1")
    if rack_width_m <= 0.0 or clearance_m < 0.0:
        raise ValueError("rack_width_m must be positive and clearance_m non-negative")
    center_x, center_y, center_z = (float(value) for value in rack_center)
    normal_x = face_sign * math.cos(rack_yaw_rad)
    normal_z = -face_sign * math.sin(rack_yaw_rad)
    face_distance = rack_width_m / 2.0 + clearance_m
    approach = (
        center_x + normal_x * face_distance,
        center_y,
        center_z + normal_z * face_distance,
    )
    if not (x_limits[0] <= approach[0] <= x_limits[1] and z_limits[0] <= approach[2] <= z_limits[1]):
        raise ValueError("The requested rack width-side approach is outside the work-area boundary")
    facing_rack = math.atan2(-normal_x, -normal_z)
    return approach, facing_rack


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _append_unique_point(points: list[tuple[float, float, float]], point: tuple[float, float, float]) -> None:
    if not points or any(not math.isclose(a, b, abs_tol=1e-9) for a, b in zip(points[-1], point)):
        points.append(point)


def plan_cardinal_rack_route(
    *,
    approach_position: Sequence[float],
    final_heading: float,
    total_frames: int,
    fps: float,
    initial_heading: float = 0.0,
    turn_seconds_per_90: float = 0.75,
    walk_speed_m_s: float = 0.90,
    first_axis: str | None = None,
) -> CardinalRoute:
    """Plan axis-aligned walking with stationary 90/180-degree turns."""
    if total_frames < 2 or fps <= 0.0 or walk_speed_m_s <= 0.0:
        raise ValueError("total_frames, fps, and walk_speed_m_s must be positive")
    target = tuple(float(value) for value in approach_position)
    start = (0.0, 0.0, 0.0)

    if first_axis not in (None, "x", "z"):
        raise ValueError("first_axis must be 'x', 'z', or None")
    # Use the central aisles and avoid diagonal shortcuts. Callers may force
    # an axis order when the warehouse face geometry requires it.
    points = [start]
    use_z_first = first_axis == "z" or (first_axis is None and abs(math.sin(final_heading)) < 0.5)
    if use_z_first:
        _append_unique_point(points, (0.0, 0.0, target[2]))
    else:
        _append_unique_point(points, (target[0], 0.0, 0.0))
    _append_unique_point(points, target)

    actions: list[tuple[str, object]] = []
    current_heading = initial_heading
    turn_degrees: list[int] = []
    for start_point, end_point in zip(points, points[1:]):
        movement_heading = forward_route_heading(start_point, end_point)
        turn_delta = _normalize_angle(movement_heading - current_heading)
        if not math.isclose(turn_delta, 0.0, abs_tol=1e-8):
            degrees = int(round(abs(math.degrees(turn_delta))))
            if degrees not in (90, 180):
                raise ValueError(f"Cardinal route produced a non-cardinal turn: {degrees} degrees")
            actions.append(("turn", turn_delta))
            turn_degrees.append(degrees)
        actions.append(("move", (start_point, end_point)))
        current_heading = movement_heading

    final_turn = _normalize_angle(final_heading - current_heading)
    if not math.isclose(final_turn, 0.0, abs_tol=1e-8):
        degrees = int(round(abs(math.degrees(final_turn))))
        if degrees not in (90, 180):
            raise ValueError(f"Final rack-facing turn is not cardinal: {degrees} degrees")
        actions.append(("turn", final_turn))
        turn_degrees.append(degrees)

    available_steps = total_frames - 1
    turn_steps = [
        max(1, int(round(turn_seconds_per_90 * fps * abs(float(value)) / (math.pi / 2.0))))
        for kind, value in actions
        if kind == "turn"
    ]
    move_distances = [
        math.hypot(value[1][0] - value[0][0], value[1][2] - value[0][2])
        for kind, value in actions
        if kind == "move"
    ]
    move_steps = [max(1, int(round(distance / walk_speed_m_s * fps))) for distance in move_distances]
    required_steps = sum(turn_steps) + sum(move_steps)
    if required_steps > available_steps:
        required_seconds = cardinal_rack_route_required_seconds(
            approach_position=approach_position,
            final_heading=final_heading,
            fps=fps,
            initial_heading=initial_heading,
            turn_seconds_per_90=turn_seconds_per_90,
            walk_speed_m_s=walk_speed_m_s,
            first_axis=first_axis,
        )
        raise ValueError(
            f"Motion duration is too short for a steady normal walk; use at least {math.ceil(required_seconds)} seconds"
        )

    positions = [start]
    headings = [initial_heading]
    position = start
    heading = initial_heading
    turn_index = 0
    move_index = 0
    for kind, value in actions:
        if kind == "turn":
            steps = turn_steps[turn_index]
            turn_index += 1
            delta = float(value)
            for step in range(1, steps + 1):
                positions.append(position)
                headings.append(heading + delta * step / steps)
            heading = _normalize_angle(heading + delta)
        else:
            steps = move_steps[move_index]
            move_index += 1
            segment_start, segment_end = value
            heading = forward_route_heading(segment_start, segment_end)
            for step in range(1, steps + 1):
                ratio = step / steps
                position = tuple(
                    float(segment_start[axis] + ratio * (segment_end[axis] - segment_start[axis]))
                    for axis in range(3)
                )
                positions.append(position)
                headings.append(heading)

    while len(positions) < total_frames:
        positions.append(position)
        headings.append(heading)

    if len(positions) != total_frames:
        raise AssertionError(f"Expected {total_frames} route frames, generated {len(positions)}")
    return CardinalRoute(
        positions=tuple(positions),
        headings=tuple(headings),
        approach_position=target,
        final_heading=_normalize_angle(final_heading),
        turn_degrees=tuple(turn_degrees),
    )


def cardinal_route_primitives(route: CardinalRoute) -> tuple[RoutePrimitive, ...]:
    """Compress a per-frame route into signed turns and forward distances."""
    primitives: list[RoutePrimitive] = []
    active_kind: str | None = None
    active_value = 0.0

    def flush() -> None:
        nonlocal active_kind, active_value
        if active_kind is not None and not math.isclose(active_value, 0.0, abs_tol=1e-8):
            primitives.append(RoutePrimitive(active_kind, active_value))
        active_kind = None
        active_value = 0.0

    for frame in range(1, len(route.positions)):
        left_position = route.positions[frame - 1]
        right_position = route.positions[frame]
        distance = math.hypot(
            right_position[0] - left_position[0],
            right_position[2] - left_position[2],
        )
        heading_delta = _normalize_angle(route.headings[frame] - route.headings[frame - 1])
        kind = "forward" if distance > 1e-9 else "turn" if abs(heading_delta) > 1e-9 else None
        if kind is None:
            continue
        if kind != active_kind:
            flush()
            active_kind = kind
        active_value += distance if kind == "forward" else heading_delta
    flush()
    return tuple(primitives)


def plan_cardinal_return_route(
    *,
    start_position: Sequence[float],
    start_heading: float,
    turn_side: str,
    total_frames: int,
    fps: float,
    reverse_distance: float = 0.10,
    reverse_seconds: float = 0.50,
    turn_seconds_per_90: float = 0.75,
    walk_speed_m_s: float = 0.90,
    initial_hold_seconds: float = 0.0,
    final_hold_seconds: float = 0.0,
    first_axis: str = "x",
    target_position: Sequence[float] | None = None,
) -> CardinalRoute:
    """Plan reverse-clearance, two 90° turns, and a cardinal return to origin."""
    if turn_side not in {"left", "right"}:
        raise ValueError("turn_side must be left or right")
    if first_axis not in {"x", "z"}:
        raise ValueError("first_axis must be x or z")
    if (
        total_frames < 2
        or fps <= 0.0
        or reverse_distance < 0.0
        or (reverse_distance > 0.0 and reverse_seconds <= 0.0)
        or walk_speed_m_s <= 0.0
        or initial_hold_seconds < 0.0
        or final_hold_seconds < 0.0
    ):
        raise ValueError("Invalid return-route duration or reverse distance")
    start = tuple(float(value) for value in start_position)
    target = tuple(float(value) for value in (target_position or (0.0, 0.0, 0.0)))
    if len(target) != 3:
        raise ValueError("target_position must contain X, Y, and Z")
    forward_x, forward_z = math.sin(start_heading), math.cos(start_heading)
    backed = (
        start[0] - reverse_distance * forward_x,
        start[1],
        start[2] - reverse_distance * forward_z,
    )
    turn_sign = 1.0 if turn_side == "left" else -1.0
    opposite_heading = _normalize_angle(start_heading + turn_sign * math.pi)

    actions: list[tuple[str, object]] = []
    if reverse_distance > 0.0:
        actions.append(("backward", (start, backed)))
    actions.extend([("turn", turn_sign * math.pi / 2.0), ("turn", turn_sign * math.pi / 2.0)])
    current = backed
    current_heading = opposite_heading
    for axis in ((0, 2) if first_axis == "x" else (2, 0)):
        if axis == 0:
            if math.isclose(current[0], target[0], abs_tol=1e-9):
                continue
            next_point = (target[0], current[1], current[2])
        else:
            if math.isclose(current[2], target[2], abs_tol=1e-9):
                continue
            next_point = (current[0], current[1], target[2])
        desired = forward_route_heading(current, next_point)
        delta = _normalize_angle(desired - current_heading)
        if not math.isclose(delta, 0.0, abs_tol=1e-9):
            actions.append(("turn", delta))
        actions.append(("forward", (current, next_point)))
        current, current_heading = next_point, desired

    available_steps = total_frames - 1
    turn_values = [float(value) for kind, value in actions if kind == "turn"]
    turn_steps = [
        max(1, int(round(turn_seconds_per_90 * fps * abs(value) / (math.pi / 2.0))))
        for value in turn_values
    ]
    initial_hold_steps = int(round(initial_hold_seconds * fps))
    final_hold_steps = int(round(final_hold_seconds * fps))
    reverse_steps = max(1, int(round(reverse_seconds * fps))) if reverse_distance > 0.0 else 0
    move_values = [value for kind, value in actions if kind == "forward"]
    move_distances = [
        math.hypot(value[1][0] - value[0][0], value[1][2] - value[0][2])
        for value in move_values
    ]
    move_steps = [max(1, int(round(distance / walk_speed_m_s * fps))) for distance in move_distances]
    required_steps = initial_hold_steps + reverse_steps + sum(turn_steps) + sum(move_steps) + final_hold_steps
    if required_steps > available_steps:
        required_seconds = (required_steps + 1) / fps
        raise ValueError(
            f"Motion duration is too short for a steady normal return walk; use at least {math.ceil(required_seconds)} seconds"
        )

    positions = [start]
    headings = [start_heading]
    position, heading = start, start_heading
    for _step in range(initial_hold_steps):
        positions.append(position)
        headings.append(heading)
    turn_index = move_index = 0
    for kind, value in actions:
        if kind == "backward":
            segment_start, segment_end = value
            for step in range(1, reverse_steps + 1):
                ratio = step / reverse_steps
                position = tuple(
                    segment_start[axis] + ratio * (segment_end[axis] - segment_start[axis])
                    for axis in range(3)
                )
                positions.append(position)
                headings.append(heading)
        elif kind == "turn":
            steps = turn_steps[turn_index]
            turn_index += 1
            delta = float(value)
            for step in range(1, steps + 1):
                positions.append(position)
                headings.append(heading + delta * step / steps)
            heading = _normalize_angle(heading + delta)
        else:
            steps = move_steps[move_index]
            move_index += 1
            segment_start, segment_end = value
            heading = forward_route_heading(segment_start, segment_end)
            for step in range(1, steps + 1):
                ratio = step / steps
                position = tuple(
                    segment_start[axis] + ratio * (segment_end[axis] - segment_start[axis])
                    for axis in range(3)
                )
                positions.append(position)
                headings.append(heading)
    while len(positions) < total_frames:
        positions.append(position)
        headings.append(heading)
    if len(positions) != total_frames:
        raise AssertionError(f"Expected {total_frames} return frames, generated {len(positions)}")
    return CardinalRoute(
        positions=tuple(positions),
        headings=tuple(headings),
        approach_position=target,
        final_heading=_normalize_angle(headings[-1]),
        turn_degrees=tuple(int(round(abs(math.degrees(value)))) for value in turn_values),
    )


def forward_route_heading(
    start_position: Sequence[float],
    target_position: Sequence[float],
) -> float:
    """Return Kimodo's heading angle for forward travel from start to target.

    Kimodo uses zero radians for facing +Z, so X is passed to ``atan2`` as
    the sine component and Z as the cosine component.
    """
    if len(start_position) != 3 or len(target_position) != 3:
        raise ValueError("start_position and target_position must contain X, Y, and Z")
    delta_x = float(target_position[0]) - float(start_position[0])
    delta_z = float(target_position[2]) - float(start_position[2])
    if math.isclose(delta_x, 0.0, abs_tol=1e-9) and math.isclose(delta_z, 0.0, abs_tol=1e-9):
        raise ValueError("start_position and target_position must be different")
    return math.atan2(delta_x, delta_z)


def rack_walk_model_prompt(prompt: str, rack_name: str) -> str:
    """Return a focused, low-ambiguity locomotion prompt for rack routes."""
    rack_label = rack_name.replace("_", " ")
    return (
        "An ordinary healthy person walks at a steady normal pace with a neutral natural gait. "
        "They stand upright, look forward, and use a relaxed symmetrical arm swing. They stop at "
        "each corner, make a controlled slow turn in place, resume the same steady walking pace, "
        f"and finish standing naturally in front of {rack_label} facing it."
    )


def rack_return_model_prompt(rack_name: str) -> str:
    """Return a focused human locomotion prompt for rack-to-counter/origin routes."""
    rack_label = rack_name.replace("_", " ")
    return (
        "An ordinary healthy person walks at a steady normal pace with a neutral natural gait. "
        "They stand upright, look forward, and use a relaxed symmetrical arm swing. They stop at "
        "each corner, make a controlled slow turn in place, resume the same steady walking pace, "
        f"and finish standing naturally at the counter/origin after leaving {rack_label}."
    )
