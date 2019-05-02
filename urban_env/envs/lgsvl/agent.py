from enum import Enum

from .geometry import Vector, Transform, BoundingBox
from .sensor import Sensor


class AgentType(Enum):
  EGO = 1
  NPC = 2
  PEDESTRIAN = 3


class VehicleControl:
  def __init__(self):
    self.steering = 0.0     # [-1..+1]
    self.throttle = 0.0     # [0..1]
    self.breaking = 0.0     # [0..1]
    self.reverse = False
    self.handbrake = False

    # optional
    self.headlights = None         # int, 0=off, 1=low, 2=high beams
    self.windshield_wipers = None  # int, 0=off, 1-3=on
    self.turn_signal_left = None   # bool
    self.turn_signal_right = None  # bool


class AgentState:
  def __init__(self, transform = Transform(), velocity = Vector(), angular_velocity = Vector()):
    self.transform = transform
    self.velocity = velocity
    self.angular_velocity = angular_velocity

  @property
  def position(self):
    return self.transform.position

  @property
  def rotation(self):
    return self.transform.rotation

  @property
  def speed(self):
    return math.sqrt(
      self.velocity.x * self.velocity.x +
      self.velocity.y * self.velocity.y +
      self.velocity.z * self.velocity.z)

  @staticmethod
  def from_json(j):
    return AgentState(
      Transform.from_json(j["transform"]),
      Vector.from_json(j["velocity"]),
      Vector.from_json(j["angular_velocity"]),
    )

  def to_json(self):
    return {
      "transform": self.transform.to_json(),
      "velocity": self.velocity.to_json(),
      "angular_velocity": self.angular_velocity.to_json(),
    }

  def __repr__(self):
    return str({
      "transform": str(self.transform),
      "velocity": str(self.velocity),
      "angular_velocity": str(self.angular_velocity),
    })


class Agent:
  def __init__(self, uid, remote, simulator):
    self.uid = uid
    self.remote = remote
    self.simulator = simulator

  @property
  def state(self):
    j = self.remote.command("agent/get_state", {"uid": self.uid})
    return AgentState.from_json(j)

  @state.setter
  def state(self, state):
    j = state.to_json()
    self.remote.command("agent/set_state", {
      "uid": self.uid,
      "state": state.to_json()
    })

  @property
  def transform(self):
    return self.state.transform

  @property
  def bounding_box(self):
    j = self.remote.command("agent/get_bounding_box", {"uid": self.uid})
    return BoundingBox.from_json(j)

  def __eq__(self, other):
    return self.uid == other.uid

  def __hash__(self):
    return hash(self.uid)

  def on_collision(self, fn):
    self.remote.command("agent/on_collision", {"uid": self.uid})
    if self not in self.simulator.callbacks:
      self.simulator.callbacks[self] = {}
    if "collision" not in self.simulator.callbacks[self]:
      self.simulator.callbacks[self]["collision"] = set()
    self.simulator.callbacks[self]["collision"].add(fn)

  def on_area_reached(self, position, distance, fn):
    raise NotImplementedError()

  @staticmethod
  def create(simulator, remote, uid, agent_type):
    if agent_type == AgentType.EGO:
      return EgoVehicle(uid, remote, simulator)
    elif agent_type == AgentType.NPC:
      return NpcVehicle(uid, remote, simulator)
    elif agent_type == AgentType.PEDESTRIAN:
      return Pedestrian(uid, remote, simulator)
    else:
      raise ValueError("unsupported agent type")


class Vehicle(Agent):
  def __init__(self, uid, remote, simulator):
    super().__init__(uid, remote, simulator)

  def set_fixed_speed(self, fixed, speed):
    raise NotImplementedError()

  def on_stop_line(self, fn):
    raise NotImplementedError()


class EgoVehicle(Vehicle):
  def __init__(self, uid, remote, simulator):
    super().__init__(uid, remote, simulator)

  @property
  def bridge_connected(self):
    raise NotImplementedError()

  def connect_bridge(self, address, port):
    raise NotImplementedError()

  def get_sensors(self):
    j = self.remote.command("vehicle/get_sensors", {"uid": self.uid})
    return [Sensor.create(self.remote, sensor) for sensor in j]

  def apply_control(self, control, sticky = False):
    args = {
      "uid": self.uid,
      "sticky": sticky,
      "control": {
        "steering": control.steering,
        "throttle": control.throttle,
        "breaking": control.breaking,
        "reverse": control.reverse,
        "handbrake": control.handbrake,
      }
    }
    if control.headlights is not None:
      args["control"]["headlights"] = control.headlights
    if control.windshield_wipers is not None:
      args["control"]["windshield_wipers"] = control.windshield_wipers
    if control.turn_signal_left is not None:
      args["control"]["turn_signal_left"] = control.turn_signal_left
    if control.turn_signal_right is not None:
      args["control"]["turn_signal_right"] = control.turn_signal_right
    self.remote.command("vehicle/apply_control", args)


class NpcVehicle(Vehicle):
  def __init__(self, uid, remote, simulator):
    super().__init__(uid, remote, simulator)

  def follow(self, waypoints, loop = False):
    raise NotImplementedError()

  def follow_closest_lane(self, follow, max_speed):
    self.remote.command("vehicle/follow_closest_lane", {"uid": self.uid, "follow": follow, "max_speed": max_speed})

  def on_waypoint_reached(self, fn):
    raise NotImplementedError()

  def on_lane_change(self, fn):
    raise NotImplementedError()


class Pedestrian(Agent):
  
  def walk_randomly(self, enable):
    raise NotImplementedError()

  def follow(self, waypoints, loop = False):
    raise NotImplementedError()

  def on_waypoint_reached(self, fn):
    raise NotImplementedError()
