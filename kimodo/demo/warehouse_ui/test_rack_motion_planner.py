import math
import unittest

from rack_motion_planner import (
    cardinal_route_primitives,
    choose_safer_turn_side,
    forward_route_heading,
    plan_cardinal_rack_route,
    plan_cardinal_return_route,
    rack_approach_pose_inside_boundary,
    rack_front_approach_position,
    rack_return_model_prompt,
    rack_shelf_object_position,
    rack_walk_model_prompt,
    rack_width_side_approach_pose,
    requested_rack_name,
    requested_base_rack_name,
    requested_base_return_rack_name,
    requested_human_return_rack_name,
    requested_rack_pick,
)


class RackMotionPlannerTest(unittest.TestCase):
    def test_recognizes_compact_and_spaced_rack_names(self):
        self.assertEqual(requested_rack_name(["walk from the origin to rack4"]), "rack_4")
        self.assertEqual(requested_rack_name(["Go to Rack 4"]), "rack_4")

    def test_does_not_apply_one_path_across_multiple_prompt_segments(self):
        self.assertIsNone(requested_rack_name(["stand", "walk to rack 4"]))

    def test_base_rack_prompt_is_explicit(self):
        self.assertEqual(requested_base_rack_name(["move base to rack 3"]), "rack_3")
        self.assertIsNone(requested_base_rack_name(["move to rack 3"]))
        self.assertEqual(
            requested_base_return_rack_name(["return base from rack 3 to origin"]),
            "rack_3",
        )
        self.assertEqual(
            requested_human_return_rack_name(["return from rack 3 to origin"]),
            "rack_3",
        )

    def test_parses_generic_shelf_four_pick_prompt(self):
        request = requested_rack_pick(["pick object 2 from rack 3 shelf 4"])
        self.assertIsNotNone(request)
        self.assertEqual(request.rack_name, "rack_3")
        self.assertEqual(request.object_index, 2)
        self.assertEqual(request.shelf_number, 4)

    def test_three_shelf_objects_are_15cm_apart_and_5cm_inside_front(self):
        positions = [
            rack_shelf_object_position(
                (-1.63, 0.0, -1.20),
                0.0,
                1.0144,
                object_index,
                face_normal_half_extent_m=0.17,
            )
            for object_index in range(1, 4)
        ]
        self.assertAlmostEqual(positions[0][2], -1.35)
        self.assertAlmostEqual(positions[1][2], -1.20)
        self.assertAlmostEqual(positions[2][2], -1.05)
        self.assertTrue(all(math.isclose(position[0], -1.51) for position in positions))
        rotated_center = rack_shelf_object_position(
            (0.0, 0.0, -3.43),
            -math.pi / 2.0,
            1.0144,
            2,
            face_normal_half_extent_m=0.17,
        )
        self.assertAlmostEqual(rotated_center[0], 0.0)
        self.assertAlmostEqual(rotated_center[2], -3.31)

    def test_rack_four_approach_is_in_front_of_rotated_rack(self):
        target = rack_front_approach_position(
            (0.0, 0.0, -3.43),
            math.pi / 2.0,
            rack_depth_m=0.72,
            clearance_m=0.45,
        )
        self.assertAlmostEqual(target[0], 0.81)
        self.assertAlmostEqual(target[1], 0.0)
        self.assertAlmostEqual(target[2], -3.43)

    def test_prompt_adds_natural_locomotion_instruction(self):
        prompt = rack_walk_model_prompt("Go to rack4", "rack_4")
        self.assertIn("steady normal pace", prompt)
        self.assertIn("neutral natural gait", prompt)
        self.assertIn("relaxed symmetrical arm swing", prompt)

    def test_all_rack_prompts_use_same_neutral_gait_without_style_bleed(self):
        shared_neutral_style = (
            "An ordinary healthy person walks at a steady normal pace with a neutral natural gait. "
            "They stand upright, look forward, and use a relaxed symmetrical arm swing."
        )
        for rack_number in range(1, 5):
            with self.subTest(rack=rack_number):
                rack_name = f"rack_{rack_number}"
                outbound = rack_walk_model_prompt(f"Go to rack {rack_number}", rack_name)
                returning = rack_return_model_prompt(rack_name)
                self.assertTrue(outbound.startswith(shared_neutral_style))
                self.assertTrue(returning.startswith(shared_neutral_style))
                self.assertNotIn("backward", returning.lower())

    def test_forward_heading_faces_along_the_rack_four_route(self):
        heading = forward_route_heading((0.0, 0.0, 0.0), (0.81, 0.0, -3.43))
        self.assertAlmostEqual(math.degrees(heading), 166.713, places=3)

    def test_rack_one_route_is_cardinal_inside_boundary_and_faces_rack(self):
        approach, final_heading = rack_width_side_approach_pose(
            (-1.63, 0.0, -1.20),
            0.0,
            rack_width_m=0.34,
            clearance_m=0.45,
            x_limits=(-1.8, 0.6),
            z_limits=(-3.6, 0.0),
        )
        for actual, expected in zip(approach, (-1.01, 0.0, -1.20)):
            self.assertAlmostEqual(actual, expected)
        route = plan_cardinal_rack_route(
            approach_position=approach,
            final_heading=final_heading,
            total_frames=180,
            fps=30.0,
            first_axis="z",
        )
        self.assertEqual(route.turn_degrees, (180, 90))
        self.assertAlmostEqual(route.headings[-1], -math.pi / 2.0)
        self.assertEqual(route.positions[-1], approach)
        for left, right in zip(route.positions, route.positions[1:]):
            delta_x = abs(right[0] - left[0])
            delta_z = abs(right[2] - left[2])
            self.assertFalse(delta_x > 1e-9 and delta_z > 1e-9)
            self.assertTrue(-1.8 <= right[0] <= 0.6)
            self.assertTrue(-3.6 <= right[2] <= 0.0)

    def test_rack_one_slow_route_gives_120cm_segment_two_seconds(self):
        route = plan_cardinal_rack_route(
            approach_position=(-1.01, 0.0, -1.20),
            final_heading=-math.pi / 2.0,
            total_frames=180,
            fps=30.0,
            walk_speed_m_s=0.60,
            first_axis="z",
        )
        z_motion_frames = sum(
            not math.isclose(left[2], right[2], abs_tol=1e-9)
            for left, right in zip(route.positions, route.positions[1:])
        )
        self.assertEqual(z_motion_frames, 60)

    def test_all_rack_broad_faces_have_45cm_inward_approaches(self):
        racks = {
            "rack_1": ((-1.63, 0.0, -1.20), 0.0, (-1.01, 0.0, -1.20)),
            "rack_2": ((-1.63, 0.0, -2.40), 0.0, (-1.01, 0.0, -2.40)),
            "rack_3": ((-1.10, 0.0, -3.43), -math.pi / 2.0, (-1.10, 0.0, -2.81)),
            "rack_4": ((0.00, 0.0, -3.43), -math.pi / 2.0, (0.00, 0.0, -2.81)),
        }
        for center, yaw, expected in racks.values():
            approach, _heading = rack_width_side_approach_pose(
                center,
                yaw,
                rack_width_m=0.34,
                clearance_m=0.45,
                x_limits=(-1.8, 0.6),
                z_limits=(-3.6, 0.0),
            )
            for actual, expected_value in zip(approach, expected):
                self.assertAlmostEqual(actual, expected_value)

    def test_rack_three_base_route_enters_along_z_before_cross_aisle(self):
        route = plan_cardinal_rack_route(
            approach_position=(-1.10, 0.0, -2.81),
            final_heading=-math.pi,
            total_frames=600,
            fps=30.0,
            first_axis="z",
        )
        primitives = cardinal_route_primitives(route)
        self.assertEqual([primitive.kind for primitive in primitives], ["turn", "forward", "turn", "forward", "turn"])
        self.assertAlmostEqual(primitives[1].value, 2.81)
        self.assertAlmostEqual(primitives[3].value, 1.10)
        moving_positions = [
            right
            for left, right in zip(route.positions, route.positions[1:])
            if right != left
        ]
        self.assertLess(moving_positions[0][2], 0.0)
        self.assertTrue(all(-1.8 <= pos[0] <= 0.6 and -3.6 <= pos[2] <= 0.0 for pos in route.positions))

    def test_safe_turn_side_uses_rack_and_boundary_clearance(self):
        obstacles = [
            (-1.63, -1.20, 0.17, 0.36),
            (-1.63, -2.40, 0.17, 0.36),
            (-1.10, -3.43, 0.36, 0.17),
            (0.00, -3.43, 0.36, 0.17),
        ]
        side = choose_safer_turn_side(
            (-0.91, 0.0, -1.20),
            -math.pi / 2.0,
            (-1.8, 0.6),
            (-3.6, 0.0),
            obstacles,
        )
        self.assertEqual(side, "left")

    def test_human_return_reverses_then_reaches_origin_cardinally(self):
        route = plan_cardinal_return_route(
            start_position=(-1.01, 0.0, -1.20),
            start_heading=-math.pi / 2.0,
            turn_side="left",
            total_frames=180,
            fps=30.0,
        )
        self.assertEqual(route.positions[-1], (0.0, 0.0, 0.0))
        self.assertGreater(route.positions[15][0], route.positions[0][0])
        for left, right in zip(route.positions, route.positions[1:]):
            delta_x = abs(right[0] - left[0])
            delta_z = abs(right[2] - left[2])
            self.assertFalse(delta_x > 1e-9 and delta_z > 1e-9)

    def test_human_style_return_can_turn_without_reverse_clearance(self):
        route = plan_cardinal_return_route(
            start_position=(-1.01, 0.0, -2.40),
            start_heading=-math.pi / 2.0,
            turn_side="left",
            total_frames=210,
            fps=30.0,
            reverse_distance=0.0,
        )
        self.assertEqual(route.positions[0], route.positions[15])
        self.assertEqual(route.positions[-1], (0.0, 0.0, 0.0))

    def test_rack_one_return_uses_two_seconds_for_120cm_segment(self):
        route = plan_cardinal_return_route(
            start_position=(-1.01, 0.0, -1.20),
            start_heading=-math.pi / 2.0,
            turn_side="left",
            total_frames=180,
            fps=30.0,
            reverse_distance=0.0,
            walk_speed_m_s=0.60,
        )
        z_motion_frames = sum(
            not math.isclose(left[2], right[2], abs_tol=1e-9)
            for left, right in zip(route.positions, route.positions[1:])
        )
        self.assertEqual(z_motion_frames, 60)
        self.assertEqual(route.positions[-1], (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
