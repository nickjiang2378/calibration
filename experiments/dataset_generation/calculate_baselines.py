import json

with open("calibration_preferences.jsonl") as f:
    data_calibration = [json.loads(line) for line in f]
with open("data_gemini/ood_dpo.jsonl") as f:
    data_ood = [json.loads(line) for line in f]
with open("data_gemini/test_dpo.jsonl") as f:
    data_test = [json.loads(line) for line in f]

calibrations = dict()
for question in data_calibration:
  calibrations[question["qid"]] = question

ood_scores = [0, 0]
for question in data_ood:
  if question["qid"] not in calibrations:
    continue


  calibration = calibrations[question["qid"]]
  if calibration[f"p{question['chosen']}"] is not None:
    ood_scores[1] += 1
    if calibration[f"p{question['chosen']}"] > 0.8:
      ood_scores[0] += 1

test_scores = [0, 0]
for question in data_test:
  if question["qid"] not in calibrations:
    continue

  calibration = calibrations[question["qid"]]
  if calibration[f"p{question['chosen']}"] is not None:
    test_scores[1] += 1
    if calibration[f"p{question['chosen']}"] > 0.8:
      test_scores[0] += 1

print(f"OOD scores: {ood_scores}, {ood_scores[0] / ood_scores[1] if ood_scores[1] > 0 else 0}")
print(f"Test scores: {test_scores}, {test_scores[0] / test_scores[1] if test_scores[1] > 0 else 0}")