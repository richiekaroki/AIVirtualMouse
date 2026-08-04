from hand_motion.ai.landmark_classifier import LandmarkClassifier
import json

c = LandmarkClassifier()
report = c.train_from_directory("motion_data")
print("Training report:")
for k, v in report.items():
    print(f"  {k}: {v}")
c.save_model()
print("Model saved.")

with open("motion_data/point_sample.json") as f:
    data = json.load(f)
lm = data["frames"][10]["landmarks"]
result = c.predict(lm)
print(f"Prediction: {result['gesture']} (confidence: {result['confidence']:.2f}, method: {result['method']})")
