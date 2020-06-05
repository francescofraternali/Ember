from Pible_func import *

SC_volt = 3.2
light = 500

PIR_on_off = 1
event = 4
temp_polling_min = 60
next_wake_up_time = 60 # min


SC_volt, energy_prod, energy_used = Pible_func.Energy(SC_volt, light, PIR_on_off, temp_polling_min, next_wake_up_time, event)

print(energy_prod, energy_used)
exit()
