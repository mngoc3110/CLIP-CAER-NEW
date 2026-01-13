class_names_5 = [
    "Neutral (student in class).",
    "Enjoyment (student in class).",
    "Confusion (student in class).",
    "Fatigue (student in class).",
    "Distraction (student in class)."
]

class_names_with_context_5 = [
    "A student shows a neutral learning state in a classroom.",
    "A student shows enjoyment while learning in a classroom.",
    "A student shows confusion during learning in a classroom.",
    "A student shows fatigue during learning in a classroom.",
    "A student shows distraction and is not focused in a classroom."
]

class_descriptor_5_only_face = [
    "A student has a neutral face with relaxed mouth, open eyes, and calm eyebrows.",
    "A student looks happy with a slight smile, bright eyes, and relaxed eyebrows.",
    "A student looks confused with furrowed eyebrows, a puzzled look, and slightly open mouth.",
    "A student looks tired with drooping eyelids, frequent yawning, and a sleepy face.",
    "A student looks distracted with unfocused eyes and a wandering gaze away from the lesson."
]

class_descriptor_5_only_body = [
    "A student sits still with an upright posture and hands on the desk, showing a neutral learning state.",
    "A student leans slightly forward with an open, engaged posture, showing enjoyment in learning.",
    "A student tilts the head and leans in, hand on chin, showing confusion while trying to understand.",
    "A student slouches with shoulders dropped and head lowered, showing fatigue during class.",
    "A student shifts around, turns away from the desk, or looks sideways, showing distraction and low focus."
]

class_descriptor_5 = [
    "A student looks neutral and calm in class, with a relaxed face and steady gaze, quietly watching the lecture or reading notes.",
    "A student shows enjoyment while learning, with a gentle smile and bright eyes, appearing engaged and interested in the lesson.",
    "A student looks confused in class, with furrowed eyebrows and a puzzled expression, focusing on the material as if trying to understand.",
    "A student appears fatigued in class, with drooping eyelids and yawning, head slightly lowered, showing low energy.",
    "A student is distracted in class, frequently looking away from the lesson, scanning around, and not paying attention to learning materials."
]

# Prompt Ensemble for RAER (5 classes)
# Each inner list contains multiple descriptions for a single class.
# prompt_ensemble_5 = [
#     [   # Neutral
#         "A photo of a student with a neutral expression.",
#         "A photo of a student sitting still and watching the lecture.",
#         "A photo of a student with a calm face and neutral body posture."
#     ],
#     [   # Enjoyment
#         "A photo of a student showing enjoyment while learning.",
#         "A photo of a student with a happy face and a slight smile.",
#         "A photo of a student who looks engaged and interested in the lesson."
#     ],
#     [   # Confusion
#         "A photo of a student who is confused.",
#         "A photo of a student with a puzzled look and furrowed eyebrows.",
#         "A photo of a student staring at the material as if trying to understand."
#     ],
#     [   # Fatigue
#         "A photo of a student who appears fatigued or sleepy.",
#         "A photo of a student with drooping eyelids or yawning.",
#         "A photo of a student showing low energy with a lowered head."
#     ],
#     [   # Distraction
#         "A photo of a student who is distracted from learning.",
#         "A photo of a student looking away from the lesson or checking a phone.",
#         "A photo of a student with a wandering gaze and unfocused eyes."
#     ]
# ]

#chỉnh prompt ensemble prompt cho phù hợp với mục đích nhận diện trạng thái học tập vì tắt crop body
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

  # 2 Confusion (face/cognitive struggle mạnh hơn)
  [
    "A photo of a student with a puzzled expression, furrowed eyebrows, and slightly squinting eyes.",
    "A photo of a student staring at the material with uncertainty, as if struggling to understand.",
    "A photo of a student showing confusion while learning, with a tense face and slightly open mouth."
  ],

  # 3 Fatigue
  [
    "A photo of a student who looks fatigued, with drooping eyelids and a low-energy face.",
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
class_descriptor_8 = [
    'A person who is feeling neutral.',
    'A person who is feeling happy.',
    'A person who is feeling sad.',
    'A person who is feeling surprise.',
    'A person who is feeling fear.',
    'A person who is feeling disgust.',
    'A person who is feeling anger.',
    'A person who is feeling contempt.'
]

class_names_8 = [
    'Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'
]

class_names_7 = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

class_descriptor_7 = [
    'A person who is feeling neutral.',
    'A person who is feeling happy.',
    'A person who is feeling sad.',
    'A person who is feeling surprise.',
    'A person who is feeling fear.',
    'A person who is feeling disgust.',
    'A person who is feeling anger.'
]

class_names_with_context_7 = [
    'A person shows neutral emotion.',
    'A person shows happy emotion.',
    'A person shows sad emotion.',
    'A person shows surprise emotion.',
    'A person shows fear emotion.',
    'A person shows disgust emotion.',
    'A person shows anger emotion.'
]

class_descriptor_7_only_face = [
    'The face of a person who is feeling neutral.',
    'The face of a person who is feeling happy.',
    'The face of a person who is feeling sad.',
    'The face of a person who is feeling surprise.',
    'The face of a person who is feeling fear.',
    'The face of a person who is feeling disgust.',
    'The face of a person who is feeling anger.'
]

class_descriptor_7_only_body = [
    'The body of a person who is feeling neutral.',
    'The body of a person who is feeling happy.',
    'The body of a person who is feeling sad.',
    'The body of a person who is feeling surprise.',
    'The body of a person who is feeling fear.',
    'The body of a person who is feeling disgust.',
    'The body of a person who is feeling anger.'
]

class_names_with_context_8 = [
    'A person shows neutral emotion.',
    'A person shows happy emotion.',
    'A person shows sad emotion.',
    'A person shows surprise emotion.',
    'A person shows fear emotion.',
    'A person shows disgust emotion.',
    'A person shows anger emotion.',
    'A person shows contempt emotion.'
]

class_descriptor_8_only_face = [
    'The face of a person who is feeling neutral.',
    'The face of a person who is feeling happy.',
    'The face of a person who is feeling sad.',
    'The face of a person who is feeling surprise.',
    'The face of a person who is feeling fear.',
    'The face of a person who is feeling disgust.',
    'The face of a person who is feeling anger.',
    'The face of a person who is feeling contempt.'
]

class_descriptor_8_only_body = [
    'The body of a person who is feeling neutral.',
    'The body of a person who is feeling happy.',
    'The body of a person who is feeling sad.',
    'The body of a person who is feeling surprise.',
    'The body of a person who is feeling fear.',
    'The body of a person who is feeling disgust.',
    'The body of a person who is feeling anger.',
    'The body of a person who is feeling contempt.'
]