import cv2
from tracking import SoccerTracker
from stats import PlayerStats

class SoccerAnalysis:
    def __init__(self, video_path):
        self.video_path = video_path
        self.tracker = SoccerTracker()
        self.stats_tracker = PlayerStats()

    def run_analysis(self):
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            tracks = self.tracker.process_frame(frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_ltrb()
                position = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                self.stats_tracker.update_position(track_id, position, frame.shape)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imshow('Soccer Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(self.stats_tracker.get_stats())

if __name__ == "__main__":
    video_path = '.mp4'
    analysis = SoccerAnalysis(video_path)
    analysis.run_analysis()
