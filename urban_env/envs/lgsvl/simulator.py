from .remote import Remote
from .agent import Agent, AgentType, AgentState
from .geometry import Vector, Transform

from collections import namedtuple

RaycastHit = namedtuple("RaycastHit", "distance point normal")

class Simulator:

  def __init__(self, address = "localhost", port = 8181):
    self.remote = Remote(address, port)
    self.agents = {}
    self.callbacks = {}
    self.stopped = False

  def load(self, scene):
    self.remote.command("simulator/load_scene", {"scene": scene})
    self.agents.clear()
    self.callbacks.clear()

  @property
  def version(self):
    return self.remote.command("simulator/version")

  @property
  def current_scene(self):
    return self.remote.command("simulator/current_scene")

  @property
  def current_frame(self):
    return self.remote.command("simulator/current_frame")

  @property
  def current_time(self):
    return self.remote.command("simulator/current_time")

  def reset(self):
    self.remote.command("simulator/reset")
    self.agents.clear()
    self.callbacks.clear()

  def stop(self):
    self.stopped = True

  def run(self, time_limit = 0.0):
    self._process("simulator/run", {"time_limit": time_limit})

  def step(self, frames = 1, framerate = 30.0):
    raise NotImplementedError()

  def _process_events(self, events):
    self.stopped = False
    for ev in events:
      agent = self.agents[ev["agent"]]
      if agent in self.callbacks:
        callbacks = self.callbacks[agent]
        event_type = ev["type"]
        if event_type in callbacks:
          for fn in callbacks[event_type]:
            if event_type == "collision":
              fn(agent, self.agents[ev["other"]], Vector.from_json(ev["contact"]))
            if self.stopped:
              return

  def _process(self, cmd, args):
    j = self.remote.command(cmd, args)
    while True:
      if j is None:
        return
      if "events" in j:
        self._process_events(j["events"])
        if self.stopped:
          break
      j = self.remote.command("simulator/continue")

  def add_agent(self, name, agent_type, state = AgentState()):
    args = {"name": name, "type": agent_type.value, "state": state.to_json()}
    uid = self.remote.command("simulator/add_agent", args)
    agent = Agent.create(self, self.remote, uid, agent_type)
    self.agents[uid] = agent
    return agent

  def remove_agent(self, agent):
    self.remote.command("simulator/remove_agent", {"uid": agent.uid})
    del self.agents[agent.uid]
    if agent in self.callbacks:
      del self.callbacks[agent]

  def get_agents(self):
    return self.agents.values()

  @property
  def weather(self):
    raise NotImplementedError()

  @weather.setter
  def weather(self, state):
    raise NotImplementedError()

  @property
  def time_of_day(self):
    raise NotImplementedError()

  @time_of_day.setter
  def time_of_day(self, t):
    raise NotImplementedError()

  def get_spawn(self):
    spawns = self.remote.command("simulator/get_spawn")
    return [Transform.from_json(spawn) for spawn in spawns]

  def map_to_gps(self, transform):
    raise NotImplementedError()

  def map_from_gps(self, gps):
    raise NotImplementedError()

  def map_point_on_lane(self, point):
    j = self.remote.command("map/point_on_lane", {"point": point.to_json()})
    return Transform.from_json(j)

  def raycast(self, origin, direction, layer_mask = -1, max_distance = float("inf")):
    hit = self.remote.command("simulator/raycast", {
      "origin": origin.to_json(),
      "direction": direction.to_json(),
      "layer_mask": layer_mask,
      "max_distance": max_distance
    })
    if hit is None:
      return None
    return RaycastHit(hit["distance"], Vector.from_json(hit["point"]), Vector.from_json(hit["normal"]))
