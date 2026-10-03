// Wildfire sentinel: internal mount plate and sun hood.
//
// The plate drops into a standard IP65 junction box (inner floor ~190 x 140 mm)
// and carries the UNO Q on standoffs, with a camera bracket and the MLX90640
// side by side behind the clear lid. The hood clips over the lid edge to keep
// low sun and rain streaks off the camera window.
//
// Render one part at a time:
//   openscad -D 'part="plate"' -o plate.stl sentinel_mount.scad
//   openscad -D 'part="hood"'  -o hood.stl  sentinel_mount.scad
// Print in PETG or ASA (PLA creeps in a sun-heated box), 0.2 mm layers.

part = "plate";           // "plate" | "hood" | "assembly"

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

/* [Camera] bracket for a 32 x 32 mm UVC board camera, lens forward */
cam_board = 32;
cam_holes = 28;           // square hole pattern, M2
cam_tilt = 8;             // degrees below horizontal: ridge lines sit low
cam_pos = [120, 100];

/* [Thermal] MLX90640 breakout (Adafruit 25.4 x 17.8 mm) */
th_w = 25.4;
th_d = 17.8;
th_holes = [[2.5, 2.5], [22.9, 2.5]];
th_pos = [158, 100];

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

// A tilted wall the camera board screws onto, lens through a window.
module camera_bracket() {
    wall_h = cam_board + 10;
    translate([cam_pos[0], cam_pos[1], plate_t - embed]) {
        rotate([90 - cam_tilt, 0, 0]) difference() {
            translate([-5, 0, 0]) cube([cam_board + 10, wall_h, 4]);
            translate([cam_board / 2, wall_h / 2 + 2, -0.1]) cylinder(d = 14, h = 5);  // lens
            for (dx = [-1, 1], dy = [-1, 1])
                translate([cam_board / 2 + dx * cam_holes / 2, wall_h / 2 + 2 + dy * cam_holes / 2, -0.1])
                    cylinder(d = 1.8, h = 5);
        }
        // foot that ties the tilted wall into the plate
        translate([-5, -14, 0]) cube([cam_board + 10, 16, 3 + embed]);
    }
}

module thermal_bracket() {
    translate([th_pos[0], th_pos[1], plate_t - embed]) {
        rotate([90 - cam_tilt, 0, 0]) difference() {
            cube([th_w + 6, th_d + 14, 3]);
            translate([3 + th_w / 2, 7 + th_d / 2, -0.1]) cylinder(d = 10, h = 4);  // sensor window
            for (p = th_holes) translate([3 + p[0], 7 + p[1], -0.1]) cylinder(d = 2.7, h = 4);
        }
        translate([0, -14, 0]) cube([th_w + 6, 16, 3 + embed]);
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
else if (part == "hood") hood();
else {
    plate();
    translate([-7, plate_d + 4, 0]) rotate([90, 0, 0]) %hood();
}
