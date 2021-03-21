import numpy as np

def validate_submission(submission, sample_submission):
    if not isinstance(submission, dict):
      print("Submission should be dict")
      return False

    if not submission.keys() == sample_submission.keys():
      print("Submission keys don't match")
      return False

    for key in submission:
      sv = submission[key]
      ssv = sample_submission[key]
      if not len(sv) == len(ssv):
        print(f"Submission lengths of {key} doesn't match")
        return False

    for key, sv in submission.items():
      if not all(isinstance(x, (np.int32, np.int64, int)) for x in list(sv)):
        print(f"Submission of {key} is not all integers")
        return False

    print("All tests passed")
    return True
