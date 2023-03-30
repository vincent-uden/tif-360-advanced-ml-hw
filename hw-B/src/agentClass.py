import datetime
import numpy as np
import h5py  # type: ignore
import tensorflow as tf  # type: ignore

from typing import Callable, Literal, List

from gameboardClass import TGameBoard


class TQAgent:
    q_file: h5py.File
    q_table: h5py.Dataset

    # Agent for learning to play tetris using Q-learning
    def __init__(self, alpha: float, epsilon: float, episode_count: int):
        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode = 0
        self.episode_count = episode_count

        self.episode_reward = 0
        self.plot_path = f"./cache/plots/{datetime.datetime.now().isoformat()}.csv"
        self.reward_tots = np.zeros(episode_count)

    def fn_init(self, gameboard: TGameBoard):
        self.gameboard = gameboard

        now = datetime.datetime.now()
        self.q_file = h5py.File(f"./cache/q_table/{now.isoformat()}.hdf5", "x")
        self.state_count = np.power(
            2, self.gameboard.N_col * self.gameboard.N_row + len(self.gameboard.tiles)
        )
        self.action_count = self.gameboard.N_col * 4
        # This shouldn't be able to fail since q_file should be unique due to
        # the time-stamp file name
        self.q_table = self.q_file.create_dataset(
            "q_table", (self.state_count, self.action_count), dtype=np.dtype(float)
        )
        # Optimistic initialisation
        self.q_table[:, :] = 0.0

        # Rewards???

    def fn_load_strategy(self, strategy_file: str):
        # The Q-table is stored as a S x A matrix.

        # S is the unique integer identifying the board state. It's calculated
        # using the board from top left to bottom right as a bit-sequence.

        # A is 28, the total amount of possible actions for each board state
        self.q_file = h5py.File(strategy_file, "r")
        self.q_table = self.q_file["q_table"]

    def fn_read_state(self):
        self.state_id = self.calc_state_id(
            self.gameboard.board, self.gameboard.cur_tile_type
        )

    @staticmethod
    def calc_state_id(board: np.ndarray, tile_type: int) -> int:
        flat = board.flatten()
        output = 0
        for i, x in enumerate(flat):
            if x > 0:
                output += 2**i

        if tile_type == 1 or tile_type == 3:
            output += 2**flat.size
        if tile_type == 2 or tile_type == 3:
            output += 2 ** (flat.size + 1)

        return output

    def fn_select_action(self):
        is_valid = False
        while not is_valid:
            if np.random.random_sample() < self.epsilon:
                # Random action
                self.action_taken = np.random.randint(0, self.action_count)
            else:
                # Greedy action
                max_val = np.nanmax(self.q_table[self.state_id, :])
                self.action_taken = np.random.choice(
                    np.nonzero(self.q_table[self.state_id, :] == max_val)[0]
                )

            # Actions are stored in a row of the Q-Table as (col, rot) in the
            # following order:
            #   (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), ...
            col = self.action_taken // 4
            rot = self.action_taken % 4

            # What do we do if it is invalid? Set to NaN?
            is_valid = self.gameboard.fn_move(col, rot)

            if not is_valid:
                self.q_table[self.state_id, self.action_taken] = np.nan

    def fn_reinforce(self, old_state_id: int, reward: float):
        # The last action performed was moving the block into place before
        # dropping it. Therefore self.action_taken does what we want and we do
        # not need an addtional self.old_action_taken
        self.q_table[old_state_id, self.action_taken] += self.alpha * (
            reward
            + np.nanmax(self.q_table[self.state_id, :])
            - self.q_table[old_state_id, self.action_taken]
        )

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1
            if self.episode % 100 == 0:
                print(
                    "episode "
                    + str(self.episode)
                    + "/"
                    + str(self.episode_count)
                    + " (reward: ",
                    str(
                        np.sum(
                            self.reward_tots[range(self.episode - 100, self.episode)]
                        )
                    ),
                    ")",
                )
            if self.episode % 1000 == 0:
                saveEpisodes = {
                    1000,
                    2000,
                    5000,
                    10000,
                    20000,
                    50000,
                    100000,
                    200000,
                    500000,
                    1000000,
                }
                if self.episode in saveEpisodes:
                    # TODO: Decide on what data needs to be saved for plotting
                    #       and strategy retrieval
                    with open(self.plot_path, "a") as f:
                        f.write(f"{self.episode},{self.episode_reward}\n")

                    self.episode_reward = 0
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data
                    # files for plotting of the rewards and the Q-table can be
                    # used to test how the agent plays

            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            old_state_id = self.state_id

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state_id, reward)


class ReplayFrame:
    def __init__(
        self,
        old_state: np.ndarray,
        action_taken: int,
        reward_obtained: float,
        next_state: np.ndarray,
    ):
        self.old_state = old_state
        self.action_taken = action_taken
        self.reward_obtained = reward_obtained
        self.next_state = next_state


class TDQNAgent:
    state: np.ndarray
    # Agent for learning to play tetris using Q-learning
    def __init__(
        self,
        alpha: float,
        epsilon: float,
        epsilon_scale: float,
        replay_buffer_size: int,
        batch_size: int,
        sync_target_episode_count: int,
        episode_count: int,
    ):
        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_scale = epsilon_scale
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.sync_target_episode_count = sync_target_episode_count
        self.episode = 0
        self.episode_count = episode_count

        self.replay_buffer: List[ReplayFrame] = [ReplayFrame(0,0,0,0)] * replay_buffer_size
        self.replay_frame = 0

    def fn_init(self, gameboard):
        self.gameboard = gameboard

        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.Input(
                shape=(gameboard.N_col * gameboard.N_col + len(gameboard.tiles)),
            )
        )
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(16, activation="relu"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),
            loss=tf.keras.losses.MeanSquaredError()
        )
        self.q_network = model
        self.target_network = tf.keras.clone_model(self.q_network)

        self.action_count = self.gameboard.N_col * 4

    def fn_load_strategy(self, strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        self.state = np.zeros(self.gameboard.board.size + 4)
        self.state[0:self.gameboard.board.size] = self.gameboard.board.copy().flatten()
        self.state[self.gameboard.board.size + self.gameboard.cur_tile_type] = 1

    def fn_select_action(self):
        is_valid = False
        r = np.random.random_sample()
        i = 0
        y_hat = self.q_network(self.state)
        reward_order = np.argsort(y_hat)
        while not is_valid:
            if r < self.epsilon:
                # Random action
                self.action_taken = np.random.randint(0, self.action_count)
            else:
                # Greedy action
                assert(i < y_hat.size)
                self.action_taken = y_hat[reward_order[i]]
                i += 1

            col = self.action_taken // 4
            rot = self.action_taken % 4

            is_valid = self.gameboard.fn_move(col, rot)


    def fn_reinforce(self, batch_indices: List[int]):
        batch_x = np.zeros((len(batch_indices), 4))
        batch_y = np.zeros((len(batch_indices), self.action_count))
        for i, n in enumerate(batch):
            batch_x[i,:] = self.replay_buffer[n].old_state
            batch_y[i,:] # This formula is in pdf TODO
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last
        # action, last reward, new state)

        # Calculate the loss function by first, for each old state, use the
        # Q-network to calculate the values Q(s_old,a), i.e. the estimate of
        # the future reward for all actions a

        # Then repeat for the target network to calculate the value \hat
        # Q(s_new,a) of the new state (use \hat Q=0 if the new state is
        # terminal)

        # This function should not return a value, the Q table is stored as an
        # attribute of self

        # Useful variables:
        # The input argument 'batch' contains a sample of quadruplets used to
        # update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1
            if self.episode % 100 == 0:
                print(
                    "episode "
                    + str(self.episode)
                    + "/"
                    + str(self.episode_count)
                    + " (reward: ",
                    str(
                        np.sum(
                            self.reward_tots[range(self.episode - 100, self.episode)]
                        )
                    ),
                    ")",
                )
            if self.episode % 1000 == 0:
                saveEpisodes = [
                    1000,
                    2000,
                    5000,
                    10000,
                    20000,
                    50000,
                    100000,
                    200000,
                    500000,
                    1000000,
                ]
                if self.episode in saveEpisodes:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and (
                    (self.episode % self.sync_target_episode_count) == 0
                ):
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets
                self.fn_reinforce(batch)


class THumanAgent:
    def fn_init(self, gameboard):
        self.episode = 0
        self.reward_tots = [0]
        self.gameboard = gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self, pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots = [0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(
                            self.gameboard.tile_x,
                            (self.gameboard.tile_orientation + 1)
                            % len(self.gameboard.tiles[self.gameboard.cur_tile_type]),
                        )
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(
                            self.gameboard.tile_x - 1, self.gameboard.tile_orientation
                        )
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(
                            self.gameboard.tile_x + 1, self.gameboard.tile_orientation
                        )
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode] += self.gameboard.fn_drop()
