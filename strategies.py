# strategies.py
from abc import ABC, abstractmethod
import numpy as np

class RewardStrategy(ABC):
    @abstractmethod
    def assign_rewards(self, planning_frames, battle_frames, max_player_health, max_enemy_health):
        """
        Assign rewards to frames based on the strategy.

        :param planning_frames: List of planning frames.
        :param battle_frames: List of battle frames.
        :param max_player_health: Maximum player health observed.
        :param max_enemy_health: Maximum enemy health observed.
        :return: Tuple of (processed_planning_frames, processed_battle_frames)
        """
        pass

class DefaultStrategy(RewardStrategy):
    def __init__(self, window_size=100):
        self.window_size = window_size

    def assign_rewards(self, planning_frames, battle_frames, max_player_health, max_enemy_health):
        all_planning_frames = planning_frames
        all_battle_frames = battle_frames

        # Calculate total damage dealt by player and enemy for each battle phase
        rounds = self.segment_into_rounds(all_planning_frames, all_battle_frames)
        
        for round_data in rounds:
            planning = round_data['planning']
            battle = round_data['battle']

            if not battle:
                continue

            # Calculate total reward and punishment
            total_reward = sum(frame.get('reward', 0) for frame in battle)
            total_punishment = sum(frame.get('punishment', 0) for frame in battle)

            # Determine winner based on total damage dealt
            if total_reward > total_punishment:
                is_winner = True  # Player wins
            elif total_reward < total_punishment:
                is_winner = False  # Enemy wins
            else:
                # Tie-breaker based on final health values
                last_battle_frame = battle[-1]
                player_health_last = last_battle_frame.get('player_health', 0)
                enemy_health_last = last_battle_frame.get('enemy_health', 0)
                is_winner = player_health_last >= enemy_health_last

            # Assign flat reward based on is_winner to planning frames
            round_reward = 1.0 if is_winner else -1.0

            for frame in planning:
                frame['assigned_reward'] = round_reward

            # Compute cumulative rewards for battle frames
            rewards = np.array([frame.get('reward', 0) for frame in battle], dtype=np.float32)
            punishments = np.array([frame.get('punishment', 0) for frame in battle], dtype=np.float32)
            net_rewards = rewards - punishments

            # Modify net rewards based on is_winner
            win_reward = 1.0 if is_winner else -1.0
            battle_rewards = net_rewards.copy()
            battle_rewards += win_reward  # Adjust rewards based on outcome

            # Linear decay weights
            weights = np.array([(self.window_size - k) / self.window_size for k in range(self.window_size)], dtype=np.float32)

            cumulative_rewards = []
            num_battle_frames = len(battle)
            for t in range(num_battle_frames):
                # Determine the window range
                start = t
                end = min(t + self.window_size, num_battle_frames)
                window_size = end - start
                window_weights = weights[:window_size]
                # Compute net reward
                net_reward = np.sum(net_rewards[start:end] * window_weights)
                cumulative_rewards.append(net_reward)

            # Assign cumulative rewards to battle frames
            for idx, frame in enumerate(battle):
                frame['assigned_reward'] = cumulative_rewards[idx]

        return all_planning_frames, all_battle_frames

    def segment_into_rounds(self, planning_frames, battle_frames):
        """
        Placeholder method. Implement round segmentation if needed.
        Currently returns a single round containing all frames.
        """
        return [{'planning': planning_frames, 'battle': battle_frames}]

class DodgeStrategy(RewardStrategy):
    def __init__(self, punishment_value=-0.5, window_size=100):
        self.punishment_value = punishment_value
        self.window_size = window_size

    def assign_rewards(self, planning_frames, battle_frames, max_player_health, max_enemy_health):
        all_planning_frames = planning_frames
        all_battle_frames = battle_frames

        #print battle frame count

        # Identify frames where the player got hit
        hit_frames = [i for i, frame in enumerate(all_battle_frames) if frame.get('punishment', 0) > 0]

        # Assign a flat reward to planning frames
        for frame in all_planning_frames:
            frame['assigned_reward'] = 0.0  # Or another logic as needed
            
        for hit_idx in hit_frames:
            # Punish frames leading up to the hit
            start = max(0, hit_idx - self.window_size)
            for i in range(start, hit_idx):
                if i < len(all_battle_frames):
                    decay = (hit_idx - i) / self.window_size
                    all_battle_frames[i]['assigned_reward'] = self.punishment_value * decay
            # Punish the frame where the hit occurred
            all_battle_frames[hit_idx]['assigned_reward'] = self.punishment_value

        # Assign a positive reward to frames where the player didn't get hit
        for i, frame in enumerate(all_battle_frames):
            if 'assigned_reward' not in frame:
                all_battle_frames[i]['assigned_reward'] = 1.0


        return all_planning_frames, all_battle_frames

class DamageStrategy(RewardStrategy):
    def __init__(self, reward_value=1.0, window_size=100):
        self.reward_value = reward_value
        self.window_size = window_size

    def assign_rewards(self, planning_frames, battle_frames, max_player_health, max_enemy_health):
        all_planning_frames = planning_frames
        all_battle_frames = battle_frames

        # Identify frames where the player did damage
        damage_frames = [i for i, frame in enumerate(all_battle_frames) if frame.get('reward', 0) > 0]

        for dmg_idx in damage_frames:
            # Reward frames leading up to the damage
            start = max(0, dmg_idx - self.window_size)
            for i in range(start, dmg_idx):
                if i < len(all_battle_frames):
                    decay = (i - start + 1) / self.window_size
                    all_battle_frames[i]['assigned_reward'] = self.reward_value * decay
            # Reward the frame where damage was done
            all_battle_frames[dmg_idx]['assigned_reward'] = self.reward_value

        # Assign a flat reward to planning frames
        for frame in all_planning_frames:
            frame['assigned_reward'] = 0.0  # Or another logic as needed

        return all_planning_frames, all_battle_frames

class AggressiveStrategy(RewardStrategy):
    def __init__(self, reward_value=1.0, punishment_value=-0.5, window_size=100):
        self.reward_value = reward_value
        self.punishment_value = punishment_value
        self.window_size = window_size

    def assign_rewards(self, planning_frames, battle_frames, max_player_health, max_enemy_health):
        all_planning_frames = planning_frames
        all_battle_frames = battle_frames

        # Identify frames where the player did damage
        damage_frames = [i for i, frame in enumerate(all_battle_frames) if frame.get('reward', 0) > 0]

        for dmg_idx in damage_frames:
            # Reward frames leading up to the damage
            start = max(0, dmg_idx - self.window_size)
            for i in range(start, dmg_idx):
                if i < len(all_battle_frames):
                    decay = (i - start + 1) / self.window_size
                    all_battle_frames[i]['assigned_reward'] = self.reward_value * decay
            # Reward the frame where damage was done
            all_battle_frames[dmg_idx]['assigned_reward'] = self.reward_value

        # Punish frames where the player avoids combat or doesn't deal damage
        for i, frame in enumerate(all_battle_frames):
            if frame.get('reward', 0) == 0 and 'assigned_reward' not in frame:
                all_battle_frames[i]['assigned_reward'] = self.punishment_value

        # Assign a flat reward to planning frames
        for frame in all_planning_frames:
            frame['assigned_reward'] = 0.0  # Or another logic as needed

        return all_planning_frames, all_battle_frames
