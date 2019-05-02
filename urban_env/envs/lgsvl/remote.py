import threading
import websockets
import asyncio
import json
import logging

class Remote(threading.Thread):

  def __init__(self, host, port):
    super().__init__(daemon=True)
    self.endpoint = "ws://{}:{}".format(host, port)
    self.lock = threading.Lock()
    self.cv = threading.Condition()
    self.data = None
    self.sem = threading.Semaphore(0)
    self.start()
    self.sem.acquire()
    
    logger = logging.getLogger('websockets')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

  def run(self):
    self.loop = asyncio.new_event_loop()                
    asyncio.set_event_loop(self.loop)
    self.loop.run_until_complete(self.process())

  async def process(self):
    self.websocket = await websockets.connect(self.endpoint, compression=None)
    self.sem.release()

    while True:
      try:
        data = await self.websocket.recv()
      except Exception as e:
        with self.cv:
          self.data = {"error": str(e)}
          self.cv.notify()
        break       
      with self.cv:
        self.data = json.loads(data)
        self.cv.notify()

  def command(self, name, args = {}):
    data = json.dumps({"command": name, "arguments": args})
    asyncio.run_coroutine_threadsafe(self.websocket.send(data), self.loop)

    with self.cv:
      self.cv.wait_for(lambda: self.data is not None)
      data = self.data
      self.data = None

    if "error" in data:
      raise Exception(data["error"])
      #self.run() # re-connect

    return data["result"]
