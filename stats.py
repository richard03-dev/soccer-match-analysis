import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd

class PlayerStats:
    def __init__(self):
        self.player_positions = {}
        self.player_distances = {}
        self.field_zones = {}

    def update_position(self, track_id, position, frame_shape):
        if track_id not in self.player_positions:
            self.player_positions[track_id] = []
            self.player_distances[track_id] = 0
            self.field_zones[track_id] = np.zeros((5, 5))

        if self.player_positions[track_id]:
            last_position = self.player_positions[track_id][-1]
            distance = euclidean(last_position, position)
            self.player_distances[track_id] += distance

        self.player_positions[track_id].append(position)
        zone_x, zone_y = int(position[0] // (frame_shape[1] / 5)), int(position[1] // (frame_shape[0] / 5))
        self.field_zones[track_id][zone_y, zone_x] += 1

    def get_stats(self):
        stats = []
        for track_id in self.player_positions.keys():
            frequent_zones = np.where(self.field_zones[track_id] > np.percentile(self.field_zones[track_id], 80))
            stats.append({
                'Player ID': track_id,
                'Distance Traveled (px)': round(self.player_distances[track_id], 2),
                'Frequent Zones': list(zip(frequent_zones[0], frequent_zones[1]))
            })
        return pd.DataFrame(stats)
