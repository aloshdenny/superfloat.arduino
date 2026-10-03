# Wiring

```
                 +---------------------------- UNO Q ----------------------------+
 12 V battery -->| VIN (7-24 V)  via 3 A fuse                                     |
                 |                                                                |
 USB-C hub ----->| USB-C   <- camera (UVC), hub also passes PD when on mains      |
                 |                                                                |
 MLX90640 ------>| Qwiic (MCU I2C4 = Wire1, 3.3 V)                                |
                 |                                                                |
 buzzer driver <-| D9  ---[1k]--- B 2N2222  C --- buzzer(-)   buzzer(+) --- 5V    |
                 |                           E --- GND        1N4148 across buzzer |
 mute button --->| D8  --- button --- GND   (INPUT_PULLUP, active low)            |
                 +----------------------------------------------------------------+
```

- The MCU's I/O is 3.3 V. Do not connect anything to the 1.8 V high-speed
  header on the Qualcomm side.
- Qwiic on the UNO Q is wired to the MCU, not to Linux. That is why the thermal
  tier runs on the STM32 and reaches Linux only through the Bridge.
- A transistor drives the buzzer because a 5 V active buzzer draws more than a
  GPIO should source. The diode protects against the coil's back-EMF.
- Pin numbers live at the top of `uno_q/wildfire-sentinel/sketch/sketch.ino`
  (`BUZZER_PIN`, `MUTE_PIN`).
