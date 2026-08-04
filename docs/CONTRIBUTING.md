# Contributing to Hand Motion Interpretation Pipeline

Thank you for your interest in contributing! This project aims to build foundational infrastructure for sign language accessibility technology.

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in Issues
- Use the bug report template
- Include: OS, Python version, steps to reproduce, expected vs. actual behavior

### Suggesting Features

- Check if the feature has been suggested
- Explain the use case and why it matters for sign language/accessibility
- Consider whether it aligns with the project's focus on infrastructure

### Code Contributions

**Before coding:**

1. Fork the repository
2. Create a feature branch (`feature/your-feature-name`)
3. Check existing issues or create one describing what you'll work on

**While coding:**

- Follow existing code style (see below)
- Add docstrings to new functions/classes
- Update relevant documentation
- Test your changes thoroughly

**Code Style:**

- Python 3.13+ compatible
- Follow PEP 8 guidelines
- Use type hints for new functions
- Use descriptive variable names
- Add comments for complex logic
- Document assumptions and limitations

**Pull Request Process:**

1. Update README.md if adding features
2. Update docs/CHANGELOG.md with your changes
3. Ensure all existing functionality still works (`python -m pytest tests/ -x -q`)
4. Write a clear PR description explaining what and why
5. Link to any related issues

### Documentation Contributions

Documentation improvements are always welcome!

- Fix typos or unclear explanations
- Add examples or use cases
- Improve installation instructions
- Translate documentation (if applicable)

### Community Guidelines

**Be Respectful:**
- Assume good intent
- Be constructive in feedback
- Remember this is volunteer work

**Accessibility Focus:**
- Consider deaf community perspectives
- Think about usability for non-technical users
- Prioritize semantic accuracy over visual polish

**Technical Quality:**
- Test on real data
- Consider edge cases
- Document limitations honestly

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/AIVirtualMouse.git
cd AIVirtualMouse

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -x -q

# Start web server
python -m src.hand_motion.web.app
# Open http://localhost:8000
```

## Project Structure

```
src/hand_motion/
├── detection.py          # Hand tracking (MediaPipe/cvzone)
├── descriptor.py         # Core abstraction (32 primitives)
├── analyzer.py           # Analysis toolkit
├── ai/
│   ├── landmark_classifier.py   # ML classifier
│   ├── gesture_recognizer.py    # CNN+LSTM (not yet wired)
│   └── inference_engine.py      # Frame buffering
├── web/
│   ├── app.py            # Flask + SocketIO server
│   └── templates/        # Web UI
└── apps/                 # Desktop applications
```

## Areas Where Contributions Are Especially Valuable

### High Priority

1. **Sign language expertise** — Linguistic validation, annotation tools
2. **Deaf community connections** — User testing, feedback
3. **ML model improvement** — Better accuracy, more gestures
4. **3D animation** — Rigging, motion retargeting, Three.js/Blender integration

### Medium Priority

1. **Testing** — Unit tests, integration tests, edge case handling
2. **Documentation** — Examples, tutorials, video guides
3. **Performance** — Optimization, profiling, memory management
4. **UI/UX** — Interface improvements, accessibility features

### Always Welcome

1. **Bug fixes** — Any size, any module
2. **Code cleanup** — Refactoring, removing duplication
3. **Example code** — Demonstrating features

## Questions?

- Open an issue with the "question" label
- Email: <karokirichard522@gmail.com>
- Be patient — this is a personal project with limited maintainer time

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in relevant documentation
- Mentioned in release notes

## Code of Conduct

**In short: Be kind, be professional, focus on accessibility.**

This project exists to serve the deaf community and advance accessibility technology. All contributors should approach this work with respect, humility, and a commitment to building systems that empower rather than exclude.

---

Thank you for helping make sign language technology more accessible!
