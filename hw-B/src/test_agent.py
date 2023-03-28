import numpy as np

from agentClass import TQAgent

def test_read_state():
    board1 = np.ones((4, 4)) * (-1)
    assert(TQAgent.state_id(board1, 0) == "00000000000000000")

    board2 = np.ones((4, 4))
    assert(TQAgent.state_id(board2, 0) == "11111111111111110")
