# Agnirath-Application-AE24B022

Assumptions :
perfect weather conditions
constant wind speed always opposing the motion of the car
air density = constant = 1.2kg/m^3
constant slope
no limit on acceleration
constant motor temp, so no change in motor efficiency. Same goes for solar panels.

Model description :
The objective of the model is to maximize the total distance traveled during the race. The race is assumed to start at 6:00 AM and end at 6:00 PM, giving a continuous 12-hour window of operation with no stops in between.

At the start, the vehicle’s battery is fully charged, and the only source of additional energy during the race is solar power received by the onboard panels. No external charging or refueling is allowed once the race begins.

The vehicle’s speed can be adjusted every 5 minutes, which sets the time resolution for the optimization. This interval is fine enough to capture meaningful changes in velocity and energy use, while still keeping the simulation computationally manageable.
