"""Plan damage-aware, bottom-to-top grocery packing with OR-Tools CP-SAT."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ortools.sat.python import cp_model


DEFAULT_INVENTORY = Path(__file__).with_name("inventory.json")


class PlanningError(ValueError):
    """Raised when an order cannot be planned safely."""


@dataclass(frozen=True)
class Unit:
    product_id: str
    unit_number: int
    name: str
    rack: str
    shelf: int
    weight_grams: int
    packing_tier: int


def load_inventory(path: Path = DEFAULT_INVENTORY) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as inventory_file:
        inventory = json.load(inventory_file)
    if not isinstance(inventory, dict) or not inventory:
        raise PlanningError("Inventory must be a non-empty JSON object.")
    return inventory


def validate_and_expand_order(
    order: Mapping[str, int], inventory: Mapping[str, Mapping[str, Any]]
) -> list[Unit]:
    units: list[Unit] = []
    for product_id, quantity in order.items():
        if product_id not in inventory:
            raise PlanningError(f"Unknown inventory item: {product_id}")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise PlanningError(f"Quantity for {product_id} must be a positive integer.")
        item = inventory[product_id]
        stock = int(item["stock"])
        if quantity > stock:
            raise PlanningError(
                f"Requested {quantity} × {item['name']}, but only {stock} are in stock."
            )
        for unit_number in range(1, quantity + 1):
            units.append(
                Unit(
                    product_id=product_id,
                    unit_number=unit_number,
                    name=str(item["name"]),
                    rack=str(item["rack"]),
                    shelf=int(item["shelf"]),
                    weight_grams=int(item["weight_grams"]),
                    packing_tier=int(item["packing_tier"]),
                )
            )
    if not units:
        raise PlanningError("The customer order is empty.")
    return units


def plan_packing(
    order: Mapping[str, int], inventory: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Return the CP-SAT packing sequence from basket bottom to top."""
    units = validate_and_expand_order(order, inventory)
    model = cp_model.CpModel()
    final_position = len(units) - 1
    positions = [
        model.new_int_var(0, final_position, f"position_{unit.product_id}_{index}")
        for index, unit in enumerate(units)
    ]
    model.add_all_different(positions)

    # Hard safety constraints: fragile and temperature-sensitive tiers must be
    # above every lower tier, and heavier units within one tier stay lower.
    for left_index, left in enumerate(units):
        for right_index, right in enumerate(units):
            if left.packing_tier < right.packing_tier:
                model.add(positions[left_index] < positions[right_index])
            elif (
                left.packing_tier == right.packing_tier
                and left.weight_grams > right.weight_grams
            ):
                model.add(positions[left_index] < positions[right_index])

    # Link units to exact basket positions so CP-SAT can minimize rack changes
    # among solutions that satisfy all packing constraints.
    at_position = [
        [model.new_bool_var(f"unit_{unit_index}_at_{position}") for position in range(len(units))]
        for unit_index in range(len(units))
    ]
    for unit_index in range(len(units)):
        model.add_exactly_one(at_position[unit_index])
        model.add(
            positions[unit_index]
            == sum(position * at_position[unit_index][position] for position in range(len(units)))
        )
    for position in range(len(units)):
        model.add_exactly_one(at_position[unit_index][position] for unit_index in range(len(units)))

    rack2_at_position = []
    for position in range(len(units)):
        on_rack2 = model.new_bool_var(f"rack2_at_{position}")
        model.add(
            on_rack2
            == sum(
                at_position[unit_index][position]
                for unit_index, unit in enumerate(units)
                if unit.rack == "rack2"
            )
        )
        rack2_at_position.append(on_rack2)
    rack_switches = []
    for position in range(1, len(units)):
        switched = model.new_bool_var(f"rack_switch_at_{position}")
        model.add_abs_equality(
            switched,
            rack2_at_position[position] - rack2_at_position[position - 1],
        )
        rack_switches.append(switched)

    # Costs mirror the six available base-motion CSV durations. Constant
    # rack-1 start/counter costs can be omitted without changing the optimum.
    route_cost_centiseconds = (
        607 * rack2_at_position[0]
        + 1357 * sum(rack_switches)
        + 600 * rack2_at_position[-1]
    )
    model.minimize(route_cost_centiseconds)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise PlanningError(f"No safe packing plan found. Solver status: {status_name}")

    items = sorted(
        (
            {
                "position": solver.value(positions[index]),
                "product_id": unit.product_id,
                "unit_number": unit.unit_number,
                "name": unit.name,
                "rack": unit.rack,
                "shelf": unit.shelf,
                "weight_grams": unit.weight_grams,
                "packing_tier": unit.packing_tier,
            }
            for index, unit in enumerate(units)
        ),
        key=lambda item: item["position"],
    )
    racks = [item["rack"] for item in items]
    switch_count = sum(left != right for left, right in zip(racks, racks[1:]))
    travel_seconds = (
        (20.43 if racks[0] == "rack1" else 26.50)
        + switch_count * 13.57
        + (20.53 if racks[-1] == "rack1" else 26.53)
    )
    return {
        "status": status_name,
        "direction": "bottom_to_top",
        "rack_switches": switch_count,
        "travel_seconds": round(travel_seconds, 2),
        "items": items,
    }


def parse_order(values: list[str]) -> dict[str, int]:
    order: dict[str, int] = {}
    for value in values:
        try:
            product_id, raw_quantity = value.split("=", 1)
            quantity = int(raw_quantity)
        except ValueError as error:
            raise PlanningError(
                f"Invalid order value '{value}'. Use item=quantity, such as potato=2."
            ) from error
        order[product_id.strip().lower()] = quantity
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a safe bottom-to-top grocery sequence.")
    parser.add_argument("items", nargs="+", help="Order entries such as potato=2 tomato=2")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    try:
        plan = plan_packing(parse_order(args.items), load_inventory(args.inventory))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, PlanningError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(plan, indent=2))
        return
    print(f"Solver status: {plan['status']}")
    print("Packing sequence (bottom → top):")
    for item in plan["items"]:
        print(
            f"  {item['position'] + 1}. {item['name']} #{item['unit_number']} "
            f"from {item['rack']} shelf {item['shelf']}"
        )


if __name__ == "__main__":
    main()
