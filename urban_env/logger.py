######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author:   Pinaki Gupta
#######################################################################

from __future__ import division, print_function
import ray


class Loggable(object):
    """
        Implements an object whose metrics can be logged through
        time and accessed as a pandas DataFrame.
    """
    def dump(self):
        """
            Update an internal log of object data.
        """
        raise Exception('Not implemented.')

    def get_log(self):
        """
            Cast the object's internal log into a pandas DataFrame.

        :return: the DataFrame containing the object's log
        """
        raise Exception('Not implemented.')


def test():
    from urban_env.vehicle.dynamics import Vehicle
    r = None
    v = Vehicle(r, [0, 0], 0, 20)
    v.dump()
    v.dump()
    v.dump()
    v.dump()
    print(v.get_log())


if __name__ == '__main__':
    test()
