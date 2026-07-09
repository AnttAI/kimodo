# OR-Tools retail fulfillment planner

The browser parses multi-item natural-language orders, validates stock, and
sends the structured order to Kimodo's `POST /packing-plan` endpoint. The
endpoint runs `packing_planner.py` with OR-Tools CP-SAT and returns an optimized
bottom-to-top basket sequence.

## Planning rules

- Packing tier is a hard constraint: heavy/load-bearing goods are below fragile
  goods, and ice cream is picked last to reduce melting time.
- Within one tier, heavier goods are placed lower.
- Among equally safe solutions, CP-SAT minimizes rack changes using the six
  available base-motion durations.
- The browser converts the result into rack navigation, shelf-specific pick
  instructions, counter navigation, and delivery steps.
- Inventory is defined in `inventory.json`; shelves are numbered bottom-to-top.

## Install and test

```bash
cd /home/jony/Downloads/warehouse_fulfillment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_packing_planner.py
```

The current `pick_item.csv` is still one generic arm motion. Shelf identities
are planned and displayed, but physical shelf-height picking requires a
validated motion (or lift/arm target) for each shelf before real deployment.

## Human motion to a rack

In the Kimodo robot demo, use one prompt segment such as:

```text
walk from the origin to rack 4
```

Generating a rack prompt automatically adds a dense, boundary-aware 2D-root
route. Translation is restricted to straight X/Z segments; the character
stops to make 90° or 180° turns and finishes 45 cm from the exact center of
the rack's inward-facing broad shelf face while facing it. This applies to all
four racks. Rack 1 ends at `(X=-1.01, Z=-1.20)`; Racks 3 and 4 face inward from
the lower boundary. The root route stays within the orange work-area boundary
and does not use diagonal shortcuts.

For TaraBase wheel-RPM generation, use an explicit base prompt:

```text
move base to rack 3
```

The route is converted into the calibrated forward and 90°/180° turn
primitives and written through the existing TaraBase CSV path. Base routes
enter the orange work area along Z before making a cross-aisle move, so Rack 3
does not travel laterally along the boundary.

After generating and previewing an outbound base route, generate its separate
return motion with:

```text
return base from rack 3 to origin
```

The return starts from the loaded outbound endpoint, reverses 10 cm, chooses
the safer left/right side from rack and boundary clearances, performs two 90°
turns, and follows a cardinal path back to `(0, 0)`. Save it as a separate Base
CSV after previewing it.

The same return is available for generated human motion. Generate the outbound
human rack motion first, keep it active, then use:

```text
return from rack 3 to origin
```

The human steps backward 10 cm, performs the same clearance-selected pair of
90° turns, and walks slowly along straight aisles to the origin. Use about 6
seconds for Rack 1 and 8 seconds for the longer Rack 3/4 returns.
