# Ember
Ember: Energy Management of Battery-less Event Detection Sensors with Reinforcement Learning

Using Deep Reinforcement Learning to detect events. Ember can detect both PIR events and temperature, humidity, pressure and light (THPL) events. The goal of the agent is to control the sensors (i.e. turning on/off the sensors) to catch events while saving energy whenever possible. In the given example, Ember will try to catch THPL events. The agent simulate the Pible mote to maximize its reward. A THPL event happend whenever the temperature, humidity, pressure and light vary by more than ±0.1C, ±1%, ±100Pa, and ±50 lux respectively.

Modify parameters on settings.json.

Set "PIR_or_thpl" to "0" to detect PIR or to "1" to detect THPL events

Set "SS_mode": "1", to test the SS Mode

# Installation
- Install conda using https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart
- conda create --name rayrllib python=3.6
- conda activate rayrllib
- pip install numpy
- pip install gym
- pip install psutil
- pip install ray[tune]
- pip install ray[rllib]==0.8.2
- pip install pandas
- pip install matplotlib
- pip install opencv-python
- pip install requests
- pip install lz4
- pip install setproctitle
- pip install tensorflow
- sudo apt-get install sshpass
- git clone https://github.com/francescofraternali/Ember.git
- run main.py
- Enjoy!
