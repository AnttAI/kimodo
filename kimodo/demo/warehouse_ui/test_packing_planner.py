import unittest

from packing_planner import PlanningError, load_inventory, plan_packing


class PackingPlannerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = load_inventory()

    def test_potatoes_are_below_tomatoes(self):
        plan = plan_packing({"tomato": 2, "potato": 2}, self.inventory)
        self.assertEqual(
            [item["product_id"] for item in plan["items"]],
            ["potato", "potato", "tomato", "tomato"],
        )

    def test_heavy_and_rigid_items_are_below_delicate_items(self):
        plan = plan_packing(
            {"chips": 1, "coconut": 1, "ice_cream": 1, "coke": 1},
            self.inventory,
        )
        self.assertEqual(
            [item["product_id"] for item in plan["items"]],
            ["coconut", "coke", "chips", "ice_cream"],
        )

    def test_all_ten_products_have_valid_rack_and_shelf(self):
        plan = plan_packing({product_id: 1 for product_id in self.inventory}, self.inventory)
        self.assertEqual(len(plan["items"]), 10)
        self.assertTrue(all(item["rack"] in {"rack1", "rack2"} for item in plan["items"]))
        self.assertTrue(all(1 <= item["shelf"] <= 5 for item in plan["items"]))

    def test_rejects_out_of_stock_order(self):
        with self.assertRaisesRegex(PlanningError, "only 20"):
            plan_packing({"tomato": 21}, self.inventory)

    def test_rejects_unknown_item(self):
        with self.assertRaisesRegex(PlanningError, "Unknown inventory item"):
            plan_packing({"glass": 1}, self.inventory)


if __name__ == "__main__":
    unittest.main()
