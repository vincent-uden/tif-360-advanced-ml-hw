import numpy as np

from agentClass import TQAgent

def test_read_state():
    board1 = np.ones((4, 4)) * (-1)
    assert TQAgent.calc_state_id(board1, 0) == 0

    board2 = np.ones((4, 4))
    assert TQAgent.calc_state_id(board2, 0) == int("0b001111111111111111", 2)

    board3 = np.zeros((4, 4))
    board3[1, 1] = 1
    assert TQAgent.calc_state_id(board3, 1) == int("0b010000000000100000", 2)

    board4 = np.zeros((4, 4))
    board4[1, 1] = 1
    assert TQAgent.calc_state_id(board4, 2) == int("0b100000000000100000", 2)
