# models/Text.py

class_names_5 = [
    "Neutral",
    "Enjoyment",
    "Confusion",
    "Fatigue",
    "Distraction"
]

class_names_with_context_5 = [
    "A student shows a neutral learning state in a classroom.",
    "A student shows enjoyment while learning in a classroom.",
    "A student shows confusion during learning in a classroom.",
    "A student shows fatigue during learning in a classroom.",
    "A student shows distraction and is not focused in a classroom."
]

class_descriptor_5 = [
    "A student looks neutral and calm in class, with a relaxed face and steady gaze, quietly watching the lecture or reading notes.",
    "A student shows enjoyment while learning, with a gentle smile and bright eyes, appearing engaged and interested in the lesson.",
    "A student looks confused in class, with furrowed eyebrows and a puzzled expression, focusing on the material as if trying to understand.",
    "A student appears fatigued in class, with drooping eyelids and yawning, head slightly lowered, showing low energy.",
    "A student is distracted in class, frequently looking away from the lesson, scanning around, and not paying attention to learning materials."
]

prompt_ensemble_5 = [
  # 0 Neutrality
  [
    "A photo of a student with a neutral facial expression, relaxed eyebrows, and calm eyes.",
    "A photo of a student looking calm and composed while learning, with a relaxed face.",
    "A photo of a student with a natural, neutral look, no strong emotion on the face."
  ],
  # 1 Enjoyment
  [
    "A photo of a student showing enjoyment while learning, with a gentle smile and bright eyes.",
    "A photo of a student looking happy and engaged, smiling slightly while studying.",
    "A photo of a student with cheerful eyes and an upturned mouth, enjoying the lesson."
  ],
  # 2 Confusion
  [
    "A photo of a student with a puzzled expression, furrowed eyebrows, and slightly squinting eyes.",
    "A photo of a student staring at the material with uncertainty, as if struggling to understand.",
    "A photo of a student showing confusion while learning, with a tense face and slightly open mouth."
  ],
  # 3 Fatigue
  [
    "A photo of a student who looks fatigued, with drooping eyelids, and a low-energy face.",
    "A photo of a student looking sleepy or tired while studying, with heavy eyes and a slack mouth.",
    "A photo of a student yawning or having half-closed eyes, showing fatigue during learning."
  ],
  # 4 Distraction
  [
    "A photo of a student who is distracted, with a wandering gaze and unfocused eyes.",
    "A photo of a student looking away from the learning material, attention clearly not on the lesson.",
    "A photo of a student with an inattentive expression, eyes drifting away and not concentrating."
  ],
]
