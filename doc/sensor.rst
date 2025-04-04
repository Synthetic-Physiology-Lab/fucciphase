# Adding a new sensor

Steps to add a new sensor.
Here, for example, the PIP-FUCCI sensor (TODO insert citation).
Check Figure 3 of the paper to get an idea of the sensor.

We followed the code structure of the FUCCI-SA sensor.
The number of fluorophores is the same but the
phases are different. PIP-FUCCI distinguishes
G1, S, G2/M


In the function `get_phase`, we implement the new logic:
there is a G1 sensor (first channel) and a S sensor (second channel).
If G1 is on or S is on but not the other, we are in the respective phase.
If none is on, we are in the S phase, if both are on, we are in G2/M.
This logic is different from the FUCCI(SA) sensor!

For now, we have not implemented the characteristic shape
of the PIP-FUCCI sensor. Thus, the code for the percentage
estimate and the intensity prediction is tagged with
a `NotImplementedError`.
