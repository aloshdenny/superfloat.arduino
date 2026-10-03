# Bill of materials

One sentinel node: a ridge- or pole-mounted camera station that runs off-grid.
Prices are indicative single-unit US retail (2026) and exclude shipping.

## Compute and sensing

| Part | Example | Qty | ~USD | Notes |
| --- | --- | --- | --- | --- |
| Arduino UNO Q, 2 GB | ABX00162 | 1 | 44 | QRB2210 (4x A53 @ 2.0 GHz) + STM32U585 |
| USB camera, UVC, 1080p, fixed focus | Arducam B0205 or similar | 1 | 35 | narrow FOV (60-70°) for distant smoke |
| USB-C hub with PD pass-through | any 3-port | 1 | 15 | camera on the single USB-C port |
| Thermal array, 32x24, Qwiic | Adafruit 4469 (55°) / SparkFun SEN-14844 (110°) | 1 | 75 | MCU thermal tier; optional |
| Qwiic cable, 100 mm | | 1 | 1 | Qwiic is MCU-side (`Wire1`), 3.3 V |
| Active buzzer, 5 V, 85 dB | | 1 | 2 | driven through a transistor |
| NPN transistor 2N2222 + 1 kΩ + 1N4148 | | 1 | 1 | buzzer driver on D9 |
| Momentary push button, IP67 panel | | 1 | 4 | mute, on D8 to GND |

## Power (off-grid)

| Part | Example | Qty | ~USD | Notes |
| --- | --- | --- | --- | --- |
| Solar panel, 60 W mono | | 1 | 70 | sized below |
| MPPT charge controller, 12 V 10 A | Victron 75/10 | 1 | 60 | LiFePO4 profile, load output |
| LiFePO4 battery, 12.8 V 25 Ah | | 1 | 110 | ~320 Wh, 3 days autonomy |
| Inline fuse holder + 3 A fuse | | 1 | 3 | battery -> board |
| Wiring to VIN | | 1 | 5 | UNO Q VIN accepts 7-24 V |

## Enclosure

| Part | Example | Qty | ~USD | Notes |
| --- | --- | --- | --- | --- |
| IP65 polycarbonate box, ~200x150x100 mm, clear lid | | 1 | 25 | camera looks through the lid |
| Printed mount plate + sun hood | `enclosure/sentinel_mount.scad` | 1 | 3 | PETG or ASA for UV |
| Pressure-equalising vent, M12 | | 1 | 4 | stops condensation pumping |
| Silica gel desiccant, 10 g | | 2 | 1 | replace at each service |
| Pole clamp kit, 40-60 mm | | 1 | 12 | |
| M2.5 / M3 stainless hardware | | 1 | 4 | |

**Node total:** about 480 USD with solar, about 200 USD for a mains-powered node
(UNO Q, camera, hub, thermal array, enclosure).

## Power budget

Measured figures replace these once the boards arrive (docs/roadmap.md).

| Load | Estimate |
| --- | --- |
| UNO Q, 4 cores busy ~10% of the time (one frame per 2 s) | 2.5 W |
| USB camera streaming | 1.0 W |
| MLX90640 + MCU tier + matrix | 0.3 W |
| **Average** | **~3.8 W -> ~91 Wh/day** |

- **Panel:** 91 Wh/day over 3 peak-sun-hours in winter, at about 70% end-to-end
  efficiency, needs 43 W. That gives a 60 W panel with margin.
- **Battery:** 3 days with no sun is 273 Wh. A 320 Wh LiFePO4 pack covers it at
  85% depth of discharge.
- **Night duty cycling:** smoke is invisible at night. Dropping the camera to
  one frame per 10 s and leaning on the thermal tier cuts the night-time load
  roughly in half. This will be measured in stage 2.
