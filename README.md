# VisionNerfGun

## Table of Contents
* [Overview ](#Overview)
* [Description ](#Description)
* [Tools](#Tools)
* [How to use](#How-to-use)
* [Future improvements](#Future-improvements)
* [License](#License)
* [Credits](#Credits)

## Overview 
We created VisionNerfGun, the vision based Nerf sentry gun, with the automatic aim, and controll panel to use it with keyboard. Plus basic camera calibration software.

## Description 
Equipment we used:
* STM32F407VG microcontroller 
* 1x Servo motor sg90
* 2x Servo motors mg996r
* Creative hd 720p camera
* NerfGun Stryfe

## Tools 
* We use [STM32CubeMX](https://www.st.com/en/development-tools/stm32cubemx.html) as Code generator , and Eclipse with [SW4STM32](https://www.st.com/en/development-tools/sw4stm32.html)
* Vision part is created with [Python 3.7](https://www.python.org/downloads/release/python-370/) + [OpenCv](https://opencv.org/) using [Pycharm](https://www.jetbrains.com/pycharm/)

## How to use
* Download VisionNerfGun
* Plug-in device and load the C program on it.
* Connect servos to power source, and connect pan servo to PD12 pin, tilt servo to PD14 pin, and trigger servo to PD15 
* Check on what port you are using Virtual ComPort, and change it in the Python project
* Use the objdetection.py program, and select:
  * (c) for calibrating your camera,
  * (m) for controll panel for steering manualy,
  * (a) for automatic use.

## Future improvements
* Custom NerfGun
* Stronger servo to work with trigger 

## License
MIT License 

## Credits 
* [Andrzej Skrobak](https://github.com/SomeonePL)
* [Przemysław Woźny](https://github.com/DalduK)

The project was conducted during the Microprocessor Lab course held by the [Institute of Control and Information Engineering](http://www.cie.put.poznan.pl/index.php?lang=en), [Poznan University of Technology](https://www.put.poznan.pl/en).
Supervisor: [Tomasz Mańkowski](https://github.com/Tomasz-Mankowski)
