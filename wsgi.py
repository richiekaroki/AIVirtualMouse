import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from hand_motion.web.app import create_app, socketio

app = create_app(data_dir="motion_data")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
