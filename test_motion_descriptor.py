from MotionDescriptor import MotionDescriptor

# Create descriptor
md = MotionDescriptor()

# Simulate hand landmarks (21 points with [id, x, y])
fake_landmarks = [[i, 100 + i*10, 200 + i*5] for i in range(21)]

# Simulate finger states (POINT gesture)
fingers = [0, 1, 0, 0, 0]

# Create descriptor
descriptor = md.create_descriptor(fake_landmarks, fingers, frame_shape=(480, 640))

print("Descriptor created successfully!")
print(f"Primitive: {descriptor['primitive']}")
print(f"Handshape: {descriptor['handshape_code']}")
print(f"Timestamp: {descriptor['timestamp']}")
print(f"Features: {descriptor['features']}")

# Test saving
md.save_sequence("motion_data/test_gesture.json", "test_point")

print("\n✓ All tests passed!")