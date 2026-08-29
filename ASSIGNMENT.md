# inviol Image Analyser Assignment

Welcome to the inviol Image Analyser technical assignment. This project evaluates your ability to build a production-ready computer vision API service for workplace health and safety monitoring.

## Dev Environment Setup

1. Install [UV](https://docs.astral.sh/uv/)
2. Set up local environment: `uv sync`
3. Run the development server: `uv run uvicorn inviol_image_analyser_assignment:app --reload`
4. Visit the docs page at `http://localhost:8000/docs` to test

---

## Assignment

### Overview

Build a FastAPI-based image analysis service that uses computer vision to assess workplace health and safety risks. Your service should accept image uploads, run object detection inference using a pretrained model, apply safety rules to detected objects, and return a structured risk assessment.

**Time allocation:** Approximately 4-8 hours

**Key decisions you need to make:**
- Which pretrained object detection model/library to use (e.g., YOLOv8, Detectron2, Transformers, an API)
- What safety rules to implement (e.g., proximity between workers and machinery)
- How to calculate and justify risk ratings
- How to structure your code for maintainability and testability

A basic skeleton exists in this repository to help you get started, but you'll need to design the core analysis logic, and service architecture for any new features.

### Core Features

These features are core and form the basis of the assignment:

1. **Image Upload Endpoint**
   - Enhance the existing `/analyse` POST endpoint in [app.py](src/inviol_image_analyser_assignment/app.py) to produce analysis results

2. **Object Detection**
   - Integrate a pretrained computer vision model for object detection
   - Detect relevant objects in workplace scenes (e.g., people, vehicles, equipment, PPE)
   - Choose an appropriate model/library and justify your choice in code comments or documentation
   - You may run inference with a model inside the API, or send to an external service

3. **Safety Rule Engine**
   - Implement at least 2 meaningful safety rules that analyze detected objects
   - Examples:
     - Person near forklift/heavy machinery (proximity-based risk)
     - Person not wearing required PPE (helmet, vest) in construction zone
     - People in restricted areas
     - Vehicles operating near pedestrian zones
     - Anything else that sounds fun
   - Rules should be configurable and well-documented

4. **Risk Assessment Response**
   - Extend the `AnalysisResult` model in [analysis_result.py](src/inviol_image_analyser_assignment/models/analysis_result.py)
   - Include overall risk rating (e.g., 0-10 scale or Low/Medium/High)
   - Provide detailed findings: what objects were detected, which rules were breached, and any other information
   - Return structured JSON that a frontend could easily consume

5. **Code Quality**
   - Follow modern Python 3.13+ best practices (type hints, async/await where appropriate)
   - Use the existing code style (Ruff is configured in [pyproject.toml](pyproject.toml))
   - Write clean, maintainable code with appropriate separation of concerns
   - Include docstrings for key functions and classes
   - Update documentation as appropriate

### Optional Features

Choose zero or more bonus features to demonstrate additional skills:

1. **Video Analysis**
   - Accept video file uploads
   - Process frames at intervals (e.g., every 1 second)
   - Aggregate risk assessments across frames
   - Return temporal analysis (risk over time) or create rules which rely on multiple frames/motion

2. **Confidence Scoring**
   - Include confidence scores from the ML model in your response
   - Filter out low-confidence detections appropriately
   - Explain how confidence affects risk ratings

3. **Visualization**
   - Return annotated images showing detected objects and risk areas
   - Use bounding boxes, labels, and color-coding for risk levels
   - Could return base64-encoded image or save to temporary storage

4. **Caching/Performance Optimization**
   - Implement model loading optimization (lazy loading, singleton pattern)
   - Add response caching for identical images
   - Include performance metrics in response (inference time)

5. **Testing**
   - Write pytest tests for your safety rules
   - Test edge cases (empty images, no detections, etc.)
   - Mock the ML model for fast unit tests

6. **Configuration Management**
   - Externalize safety rule thresholds (YAML/JSON config file)
   - Allow runtime configuration of risk weights
   - Environment-based settings (dev vs production) or a configuration database

7. **Extended API Features**
   - Batch processing endpoint (multiple images)
   - Streaming endpoint for real-time camera feeds
   - Historical analysis comparison

8. **Containerization**
   - Create a Dockerfile for the service
   - Optimize image size (multi-stage builds, appropriate base image)
   - Include compose.yaml or a devcontainer definition for easy local development
   - Document deployment instructions

9. **Improved API Input Validation**
   - Accept image files (JPEG, PNG) via multipart/form-data
   - Validate file types and implement reasonable size limits
   - Handle errors gracefully (invalid files, corrupted images, etc.)

### Criteria

Your submission will be evaluated on the following criteria during our follow-up discussion:

#### 1. Technical Implementation
- **Model Selection:** Is the chosen model appropriate for the use case? Performance vs accuracy tradeoffs?
- **Code Architecture:** Clean separation of concerns (API layer, service layer, models)?
- **Error Handling:** Robust handling of edge cases and failure modes?
- **Performance:** Reasonable response times for API requests?

#### 2. Safety Rule Logic
- **Rule Quality:** Are the safety rules meaningful and realistic?
- **Risk Calculation:** Is the risk rating methodology logical and well-justified?
- **Configurability:** Can rules be easily modified or extended?
- **Documentation:** Are rules clearly explained?

#### 3. Code Quality
- **Readability:** Clean, self-documenting code with clear naming?
- **Type Safety:** Proper use of type hints and Pydantic models?
- **Best Practices:** Following Python idioms and modern patterns?
- **Tooling Compliance:** Code passes Ruff linting and Pyright type checking?

#### 4. API Design
- **Response Structure:** Well-designed, intuitive JSON responses?
- **Validation:** Proper input validation and meaningful error messages?
- **Documentation:** Clear API behavior?

#### 5. Bonus Features
- Implementation of optional features demonstrates initiative and depth

#### 6. Discussion & Decisions
During the follow-up meeting, be prepared to discuss:
- Why you chose your specific model and library
- Trade-offs you considered (speed vs accuracy, complexity vs functionality)
- How you would scale this service for production
- What you would improve with more time
- How you would handle model updates or multiple model versions
- Security considerations for a production deployment

---

## Submission

1. Complete your implementation a repository using this as a template, and include commit history
2. Ensure the service runs locally
3. Include a brief summary of your approach (can be added to this README or separate doc)
4. List any additional dependencies you added and why
5. Provide example request(s) showing how to test your implementation

**Note:** Focus on demonstrating your engineering skills rather than achieving perfect ML accuracy. We're more interested in your code architecture, decision-making process, and ability to build production-ready services than in state-of-the-art CV performance.

Good luck!
