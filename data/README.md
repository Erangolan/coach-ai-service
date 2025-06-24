# Training Data Structure

## Directory Structure
```
data/
├── exercises/           # Exercise directory
│   ├── exercise_name/   # Name of the exercise (e.g., squat, deadlift, pushup)
│   │   ├── good/       # Good examples of the exercise
│   │   └── bad/        # Bad examples of the exercise
│   └── ...
└── documentation/       # Documentation and explanations
    ├── exercise_name/   # Folder for each exercise
    │   ├── good_examples.md    # Explanation of good examples
    │   └── bad_examples.md     # Explanation of bad examples
    └── ...
```

## Guidelines for Uploading Videos

### File Name Format
- Format: `YYYY-MM-DD_exercise_name_description.mp4`
- Example: `2024-03-20_squat_deep_good_form.mp4`

### Video Requirements
1. Minimum quality: 720p
2. Good lighting
3. Clear shooting angle
4. Duration: 5-15 seconds
5. Background as clean as possible

### Documentation
For each exercise, create two documentation files:
1. `good_examples.md` - Explanation of what is considered a good movement
2. `bad_examples.md` - Explanation of common mistakes

## Adding a New Exercise

1. Create a new folder in `exercises/` with the exercise name
2. Create `good/` and `bad/` folders inside the exercise folder
3. Create a new folder in `documentation/` with the exercise name
4. Create documentation files `good_examples.md` and `bad_examples.md`
5. Upload the appropriate videos to the `good/` and `bad/` folders
6. Update the documentation accordingly

## Example Documentation

### good_examples.md
```markdown
# Good Examples for [Exercise Name]

## Key Points for Good Movement
1. [Point 1]
2. [Point 2]
3. [Point 3]

## Example Videos
- [Video 1]: [Description]
- [Video 2]: [Description]

## Tips
- [Tip 1]
- [Tip 2]
```

### bad_examples.md
```markdown
# Common Mistakes in [Exercise Name]

## Common Mistakes
1. [Mistake 1]
   - Explanation
   - How to avoid
   - How to correct

2. [Mistake 2]
   - Explanation
   - How to avoid
   - How to correct

## Example Videos
- [Video 1]
- [Video 2]
``` 