// Wildfire sentinel: internal mount plate, IR window ring and sun hood.
//
// The plate drops into a standard IP65 junction box (inner floor ~190 x 140 mm,
// ~100 mm deep) and carries the UNO Q on standoffs. The camera and the MLX90640
// sit on raised platforms facing the clear lid (+z), tilted cam_tilt degrees
// downward. The box is pole-mounted with the lid facing the horizon and the
// plate's +y edge at the top.
//
// Polycarbonate is opaque to long-wave infrared, so the thermal sensor looks
// through its own window: drill a 16 mm hole in the lid in front of it and
// clamp 0.1 mm HDPE film (the material of PIR sensor lenses, ~80% transmissive
// at 8-14 um) or a germanium window under the printed ring. The hood clips over
// the lid edge to keep low sun and rain streaks off both windows.
//
// Render one part at a time:
//   openscad -D 'part="plate"'  -o plate.stl  sentinel_mount.scad
//   openscad -D 'part="window"' -o window.stl sentinel_mount.scad
//   openscad -D 'part="hood"'   -o hood.stl   sentinel_mount.scad
// Print in PETG or ASA (PLA creeps in a sun-heated box), 0.2 mm layers.

part = "plate";           // "plate" | "window" | "hood" | "assembly"

/* [Box] */
plate_w = 186;            // fit to the box's inner floor, minus clearance
plate_d = 136;
plate_t = 3;
corner_r = 6;

/* [UNO Q] UNO form factor; hole positions from the UNO R3 outline (mm) */
board_w = 68.6;
board_d = 53.4;
board_holes = [[13.97, 2.54], [15.24, 50.8], [66.04, 7.62], [66.04, 35.56]];
standoff_h = 8;
standoff_d = 6;
m25_clear = 2.7;          // M2.5 self-tapping into the standoff: use 2.2
board_pos = [12, 70];

/* [Camera] platform for a 32 x 32 mm UVC board camera, lens toward the lid */
cam_board = 32;
cam_holes = 28;           // square hole pattern, M2
cam_tilt = 8;             // degrees below horizontal: ridge lines sit low
cam_pos = [120, 95];      // platform centre on the plate
mount_h = 70;             // platform height: lens 5-10 mm behind a lid ~95 mm up

/* [Thermal] MLX90640 breakout (Adafruit 25.4 x 17.8 mm), sensor toward the lid */
th_w = 25.4;
th_d = 17.8;
th_holes = [[2.5, 2.5], [22.9, 2.5]];
th_pos = [164, 95];

/* [IR window] clamp ring for HDPE film or a germanium window over a 16 mm lid hole */
win_hole = 16;
win_od = 36;
win_t = 3;

/* [Hood] */
hood_depth = 60;
hood_wall = 2.4;
lid_w = 200;
lid_h = 150;

$fn = 40;
embed = 1;                // parts sink this far into the plate so the union is manifold

module rounded_plate(w, d, t, r) {
    hull() for (x = [r, w - r], y = [r, d - r]) translate([x, y, 0]) cylinder(r = r, h = t);
}

module standoff(h, od, id) {
    difference() {
        cylinder(d = od, h = h);
        translate([0, 0, -0.1]) cylinder(d = id, h = h + 0.2);
    }
}

module uno_mount() {
    translate(board_pos) for (p = board_holes)
        translate([p[0], p[1], plate_t - embed]) standoff(standoff_h + embed, standoff_d, 2.2);
}

// A platform on four legs, tilted cam_tilt degrees about x so the optical axis
// points slightly below the horizon when the box is pole-mounted.
module tilted_platform(pos, w, d) {
    translate([pos[0], pos[1], 0]) {
        translate([0, 0, mount_h]) rotate([cam_tilt, 0, 0]) difference() {
            translate([-w / 2, -d / 2, -3]) cube([w, d, 3]);
            children();
        }
        for (dx = [-1, 1], dy = [-1, 1]) hull() {
            translate([dx * (w / 2 - 3), dy * (d / 2 - 3), plate_t - embed]) cylinder(d = 6, h = 1);
            translate([0, 0, mount_h]) rotate([cam_tilt, 0, 0])
                translate([dx * (w / 2 - 3), dy * (d / 2 - 3), -3]) cylinder(d = 6, h = 3);
        }
    }
}

module camera_bracket() {
    tilted_platform(cam_pos, cam_board + 8, cam_board + 8) {
        translate([-6, -9, -4]) cube([12, 18, 5]);  // pass-through for the USB lead
        for (dx = [-1, 1], dy = [-1, 1])
            translate([dx * cam_holes / 2, dy * cam_holes / 2, -4]) cylinder(d = 1.8, h = 5);
    }
}

module thermal_bracket() {
    tilted_platform(th_pos, th_w + 6, th_d + 10) {
        translate([-th_w / 2 + 2, -th_d / 2 + 2, -4]) cube([th_w - 4, 6, 5]);  // Qwiic leads
        for (p = th_holes) translate([p[0] - th_w / 2, p[1] - th_d / 2, -4]) cylinder(d = 2.7, h = 5);
    }
}

// Clamps the IR window material over the lid hole; three M3 screws, silicone
// bead under the film for the seal.
module window_ring() {
    difference() {
        cylinder(d = win_od, h = win_t);
        translate([0, 0, -0.1]) cylinder(d = win_hole - 2, h = win_t + 0.2);
        for (a = [0, 120, 240]) rotate([0, 0, a]) translate([win_od / 2 - 5, 0, -0.1]) cylinder(d = 3.4, h = win_t + 0.2);
    }
}

module cable_slots() {
    for (x = [60, 100]) translate([x, -0.1, -0.1]) cube([16, 10, plate_t + 0.2]);
}

module plate() {
    difference() {
        rounded_plate(plate_w, plate_d, plate_t, corner_r);
        cable_slots();
        // fixing holes into the box floor bosses
        for (x = [8, plate_w - 8], y = [8, plate_d - 8]) translate([x, y, -0.1]) cylinder(d = 4.2, h = plate_t + 0.2);
        // lighten
        for (i = [0:3]) translate([20 + i * 22, 20, -0.1]) cylinder(d = 14, h = plate_t + 0.2);
    }
    uno_mount();
    camera_bracket();
    thermal_bracket();
}

// Three-sided visor that clips over the lid flange.
module hood() {
    difference() {
        cube([lid_w + 2 * hood_wall, hood_depth, lid_h * 0.45]);
        translate([hood_wall, -0.1, hood_wall]) cube([lid_w, hood_depth + 0.2, lid_h]);
    }
    // clip lip
    for (x = [0, lid_w + hood_wall]) translate([x, 0, 0]) cube([hood_wall, 6, lid_h * 0.45 + 4]);
}

if (part == "plate") plate();
else if (part == "window") window_ring();
else if (part == "hood") hood();
else {
    plate();
    translate([-7, plate_d + 4, 0]) rotate([90, 0, 0]) %hood();
}
